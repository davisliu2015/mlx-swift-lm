# Qwen3.5/3.6 MTP D2 多 draft 投机解码方案

> 本文档记录在 mlx-swift-lm 中，为 Qwen3.5/3.6（**VLM 与纯文本 LLM 均已支持**）的 MTP 投机解码从
> **D1（预测 1 个 token）** 扩展到 **D2（预测 2 个 token）** 的完整方案，重点解决 DeltaNet
> （线性注意力）递归状态在"部分接受"时如何精确、廉价地恢复。给出两种可切换的恢复实现
> （**快照方案 D** / **变换链方案 C**）及其在 M3 Pro 上的实测对比。
>
> 前置阅读：`Qwen35_MTP.md`（MTP 架构与 D1 实现）。

## 目录

- [1. 背景：为什么 D2 需要额外处理](#1-背景为什么-d2-需要额外处理)
- [2. 三种接受情形与状态点](#2-三种接受情形与状态点)
- [3. DeltaNet 状态的仿射性（方案 C 的数学基础）](#3-deltanet-状态的仿射性方案-c-的数学基础)
- [4. 方案 D：完整状态快照](#4-方案-d完整状态快照)
- [5. 方案 C：变换链折叠](#5-方案-c变换链折叠)
- [6. 两方案的切换开关（foldOnly，用户可配置 + 热更新）](#6-两方案的切换开关foldonly用户可配置--热更新)
- [7. 实测数据（M3 Pro）](#7-实测数据m3-pro)
- [8. VLM 与 LLM 的支持差异](#8-vlm-与-llm-的支持差异)
- [9. 关键文件与 API 清单](#9-关键文件与-api-清单)

---

## 1. 背景：为什么 D2 需要额外处理

D1 每轮只预测 1 个 draft token，verify 后要么接受（拿 draft + bonus 共 2 个），要么拒绝
（拿修正 token 1 个）。DeltaNet 的递归状态只需一个"confirmed 后"的快照即可回滚。

D2 每轮迭代 MTP head 两次，生成 `d1`、`d2` 两个 draft，verify 时 backbone 一次吃
`[confirmed, d1, d2]`（`nConfirmed = 1`）。产出取决于命中情况：

| 情形 | 概率(实测) | 产出 token | DeltaNet 状态需求 |
|---|---|---|---|
| **全接受** d1✅ d2✅ | ~30% | d1 + d2 + bonus = 3 | cache 主状态即最终态，无需恢复 |
| **部分接受** d1✅ d2❌ | ~44% | d1 + 修正 = 2 | **需恢复到"走完 d1 后"的中间态** |
| **拒绝** d1❌ | ~26% | 修正 = 1 | 回滚到"confirmed 后"base state |

**唯一的难点是"部分接受"**：此时 cache 里的 DeltaNet 状态已经走完了 d1 **和** d2
（verify 一次性 forward 了两个 draft 位置），但我们只接受 d1，必须把状态"退回到只走完 d1"的中间态
`S_d1`。而 DeltaNet 是递归状态（非 KV，不能按 token 裁剪），无法像 attention 的 KV cache 那样
`trim(1)` 了事。

三个状态点：

```
S_confirmed ──走d1──► S_d1 ──走d2──► S_d2(= verify 后 cache 主状态)
     ▲                   ▲                    ▲
   reject 回到这        partial 需要退到这     full accept 用这
```

---

## 2. 三种接受情形与状态点

- **`S_confirmed`**：处理完 confirmed token 后的状态。verify 的 Chunk1 自然产出，存入
  `MambaCache.rollbackState`。reject 时 `rollback()` 恢复它。
- **`S_d1`**：走完 d1（未走 d2）的中间态。**只有 partial accept 需要**。这是本方案的核心。
- **`S_d2`**：走完 d1、d2 的状态，就是 verify forward 后 cache 的主状态（`state[1]`）。
  full accept 直接用，无需任何恢复。

对 attention（KV cache）而言，partial/reject 都只是 `trim(n)` 掉多写的 draft 位置，成本 O(1)。
问题全部集中在 48 个 DeltaNet 层的递归状态上。

---

## 3. DeltaNet 状态的仿射性（方案 C 的数学基础）

Gated DeltaNet 单步状态更新对上一状态是**仿射变换**：

```
S_t = A_t · S_{t-1} + C_t
其中  A_t = diag(g_t) − β_t · k_t · k_tᵀ     （g = 衰减门，β = 写入强度，k = key）
      C_t = β_t · k_t · v_tᵀ                 （v = value）
```

关键推论：给定 `S_confirmed` 和第 t 步的 `(k_t, v_t, g_t, β_t)`，可以**只用这几个轻量算子**
重放出 `S_d1`，无需保存整份 `S_d1`。这就是方案 C 用 ~几 MB 的算子替代 ~150 MB 完整快照的依据。

`g` 的计算：`g = exp(−exp(aLog) · softplus(a + dtBias))`，见 `computeGatedDeltaG`。
`β = sigmoid(b)`。存储时直接存算好的 `g`、`β`（而非原始 `a`、`b` + 每层 `aLog`/`dtBias`），
便于折叠时把 48 层堆到 batch 维一次 kernel 完成。

---

## 4. 方案 D：完整状态快照

**思路**：verify 的 Chunk2 逐 draft 处理时，走完每个 draft 就存一份完整状态
`[convState, ssmState]`。partial accept 时 O(1) 拷贝恢复。

- 存储：`MambaCache.draftSnapshots: [[MLXArray]]`，`draftSnapshots[0]` = 走完 d1 后。
- 恢复：`MambaCache.rollbackToDraft(0)` 直接把 `state` 设为 `draftSnapshots[0]`。
- 成本：每份快照 48 层 × SSM state ≈ **~150 MB**；恢复只是引用赋值 + 一次 eval，实测 **~0.6 ms/次**。
- 代价在"存"而非"恢复"：这份快照在 verify 时就要落地，占 decode 阶段的内存带宽。

## 5. 方案 C：变换链折叠

**思路**：不存完整快照，只存轻量算子；partial accept 时从 `S_confirmed` 出发用算子折叠出 `S_d1`。

- 存储：`MambaCache.draftOperators: [[MLXArray]]`，每个元素
  `[kNormed_i, v_i, g_i, β_i, convSnap_i]`，体积 ~几 MB。
- 恢复：`foldPartialAccept(cache:draftIndex:)`（VLM/LLM 顶层模型均实现）：
  1. 收集所有 DeltaNet 层（VLM 48 层 / LLM 也是全部 linear 层）的 `(k, v, g, β, base state)`；
  2. **堆叠到 batch 维**，一次 `gatedDeltaUpdateWithG` kernel 完成全部层折叠（**优化点 1**，
     避免逐层多次 kernel launch）；
  3. 切回各层写入 cache（`applyFolded`）。
- 成本：折叠是一次真实的 SSM kernel，实测 **~9 ms/次**（比快照恢复 ~0.7~1.3 ms/次 贵约一个量级），
  换来的是每轮 verify **不必**落地 ~150 MB 快照的内存带宽。是否划算取决于机器的算力/带宽比。

**数值正确性**：C 折叠结果与 D 快照严格相等——VLM/LLM 均实测 partial accept 最大相对误差
**0.00e+00**（完全一致，见第 7 节自检）。

## 6. 两方案的切换开关（`foldOnly`，用户可配置 + 热更新）

`foldOnly` 是 **`MambaCache` 的实例属性**（不再是全局 `static`），因此可作为 smlx 等上层应用的
**用户可配置项**，且**支持热更新**：

```swift
// 拿到某次生成的 cache 数组后，对所有 DeltaNet 层的 MambaCache 设置：
for c in cache {
    (c as? MambaCache)?.foldOnly = wantFoldOnly
}
```

- `foldOnly = false`（默认）：Chunk2 同时存 `draftSnapshots`(D) + `draftOperators`(C)。
  用于方案 D，或做 C vs D 数值自检时的对比基准。
- `foldOnly = true`：Chunk2 **跳过 150 MB 快照**，只存轻量 operators。方案 C 的最优形态。
- 恢复路径由调用方选择：`rollbackToDraft(0)`（D）或 `foldPartialAccept(cache:draftIndex:0)`（C）。

**热更新语义**：`foldOnly` 在**每轮 verify 的 Chunk2 开头被读取**，用于决定这一轮是否落地快照。
因此模型**跑到一半改这个值，下一轮 verify 立即生效**，无需重建 cache 或重载模型。
`copy()` 会一并复制 `foldOnly`，保证 cache 克隆后配置不丢。

`full accept` / `reject` 两种情形**都不触发折叠**，无额外开销：full 用主状态、reject 用 `rollback()`。
只有 partial accept 才走恢复逻辑。

---

## 7. 实测数据（M3 Pro）

测试代码：`mlx_proj/test_mtp_v2/`（Release 构建，Metal kernel 需 xcodebuild）。
`./run.sh vlm`（默认）跑 VLM，`./run.sh llm` 跑纯文本 LLM。

### 7.1 VLM（`Qwen3.8-27B-MXFP4-VL-MTP`，prompt ≈ 657 token，maxTokens = 512）

| 方案 | temp=0 轮1 | temp=0 轮2 | temp=0.7 |
|---|---|---|---|
| Baseline（关闭 MTP，纯自回归） | 8.59 | 8.63 | 8.60 |
| D1（MTP 1-draft） | 12.28 | 12.44 | 11.70 |
| D2-快照（方案 D） | 12.56 | 12.68 | 13.06 |
| D2-变换（方案 C，foldOnly 最优） | 12.10 | 12.47 | 12.80 |

（单位 tok/s）相对 Baseline：D1 约 **+36~44%**，D2 约 **+46~52%**。

### 7.2 LLM（`samwang0041/Qwen3.6-27B-MLX-4bit-MTP` 纯文本，M3 Pro）

| 方案 | temp=0 轮1 | temp=0 轮2 | temp=0.7 |
|---|---|---|---|
| Baseline（关闭 MTP，纯自回归） | 8.06 | 8.21 | 8.17 |
| D1（MTP 1-draft） | 13.35 | 13.46 | 11.55 |
| D2-快照（方案 D） | 15.37 | 14.76 | 13.41 |
| D2-变换（方案 C，foldOnly 最优） | 14.70 | 14.74 | 12.57 |

（单位 tok/s）命中率(temp=0)：d1 draft **88%**、d2 draft 54%，均每轮接受 2.41 token。

- 相对 Baseline：D1 约 **+64~66%**，D2 约 **+80~91%**。
- 相对 D1：D2 约 **+10~16%**。

### 7.3 两模型的 MTP 头量化差异（重要）

VLM 与 LLM 用的是**两个不同的发布模型**，MTP 头量化状态相反：

| | VLM `davisliu/Qwen3.8-27B-MXFP4-VL-MTP` | LLM `samwang0041/Qwen3.6-27B-MLX-4bit-MTP` |
|---|---|---|
| 主干量化 | MXFP4 | 4bit (MLX) |
| **MTP 头** | **未量化（fp16，仅 `.weight`）** | **4bit 量化（`.weight`+`.scales`+`.biases`）** |
| Baseline | ~8.6 t/s | ~8.1 t/s |
| d1 命中率 | ~77% | ~88% |

**为什么 LLM 的相对提升更高？** 主要是**命中率更高（88% vs 77%）**——投机解码命中率越高，
一次前向多出的 token 越多、赚头越大。这更可能源于模型本身/MTP 头训练质量，而非量化
（量化只会拖低命中率）。此外"相对提升 %"会被较慢的 Baseline 放大；按**绝对速度**看两者
更接近（VLM D2 ~12.5 t/s、LLM D2 ~15 t/s）。

### 7.4 结论

1. **MTP 本身是主要收益来源**：相对纯自回归 Baseline，VLM 上 D1 +36~44%、LLM 上 D1 +64~66%。
2. **D2 相对 D1 稳定正收益**：VLM temp=0 约 +2% / temp=0.7 约 +11.7%；LLM 约 +10~16%。
   命中率越高、全接受越多，D2 优势越明显。
3. **C vs D 数值完全一致**（0 误差，VLM/LLM 均实测），但**折叠耗时 ~9 ms/次远高于快照恢复
   ~0.7~1.3 ms/次**。partial accept 占比约 1/3，因此 **M3 Pro 上 C 比 D 慢约 0.1~6.3%**。
4. **选型**：M3 Pro 上优先 **D2-快照**（最快、最简）。方案 C 省 ~150 MB/轮快照带宽，其价值在
   **算力更强 / 内存带宽更受限**的机器（折叠算力代价被稀释、省带宽收益放大）可能反超 D，
   需在目标机器实测。
5. 更深的 D3/D4 已验证为负收益（MTP head 预测深度不足，d3/d4 命中率极低），不在本方案内。

---

## 8. VLM 与 LLM 的支持差异

VLM 与 LLM **均已完整支持 D2（快照 D + 变换 C）**：

| 能力 | VLM 版 `MLXVLM/Models/Qwen35.swift` | LLM 版 `MLXLLM/Models/Qwen35.swift` |
|---|---|---|
| D1（1-draft） | ✅ | ✅ |
| D2 全接受 / 拒绝 | ✅ | ✅ |
| **D2 部分接受（S_d1 恢复）** | ✅（快照 D + 变换 C） | ✅（快照 D + 变换 C） |
| `draftSnapshots` / `draftOperators` | ✅ 逐 draft 填充 | ✅ 逐 draft 填充 |
| `foldPartialAccept` | ✅ | ✅（`Qwen35Model` 转发到 `Qwen35TextModel`） |
| `foldOnly`（实例属性 + 热更新） | ✅ | ✅（共用 `MambaCache.foldOnly`） |
| MTP 权重加载 | 需 `loadMTPWeights()`（绕过 MXFP4 包装） | sanitize 阶段已加载，无需额外调用 |
| draft 位置对齐 | `mtpForward(positionOffset:)` 显式（M-RoPE） | cache offset 隐式递增，自动对齐 |

**LLM 版实现要点**：
1. `Qwen35GatedDeltaNet` 的 Chunk2 改为 `for i in 0..<nDraft` 逐 draft 循环，逐个 `gatedDeltaUpdate`
   并存 `draftSnapshots`(foldOnly 时跳过) + `draftOperators`；Chunk1（confirmed）清空上一轮存储。
2. `Qwen35TextModel.foldPartialAccept` 遍历所有 linear 层、堆 batch 维一次折叠、`applyFolded` 回写；
   顶层 `Qwen35Model.foldPartialAccept` 转发到它。
3. 该模型 config 为 VLM 布局（字段在 `text_config` 内、权重前缀 `language_model.*`）但无 vision，
   `model_type=qwen3_5` 由 `LLMModelFactory` + `Qwen35Configuration`(从 text_config 解析) 加载。

### ⚠️ 协议派发坑（务必注意）

`foldPartialAccept` **必须声明为 `MTPCapableModel` 协议 body 里的协议要求**，而不能只在协议
`extension` 里给默认实现 `{ false }`。

- 若只在 extension 提供默认实现、未在协议 body 声明为要求，则通过 `MTPCapableModel` **协议类型**
  （如 `model as? MTPCapableModel`）调用时会**静态派发到扩展默认实现 `{ false }`**，具体模型
  （`Qwen35Model` / `Qwen35`）的真实折叠实现被完全绕过。
- 现象：折叠耗时 0.000 ms、`state[1]` 从不更新、C vs D 自检出现大误差（曾实测 1.94）。
- 修复：把声明放进协议 body（走 witness table 动态派发），扩展只留默认实现。
- 历史教训：本轮把 test 的 `model` 统一成协议类型后暴露此坑；此前 VLM 用具体类型调用侥幸没踩到。

---

## 9. 关键文件与 API 清单

**`Libraries/MLXLMCommon/KVCache.swift` — `MambaCache`**
- `rollbackState: [MLXArray]?` — confirmed 后 base state（reject 用）。
- `draftSnapshots: [[MLXArray]]` — 方案 D，逐 draft 完整状态快照。
- `draftOperators: [[MLXArray]]` — 方案 C，逐 draft 轻量算子 `[k,v,g,β,convSnap]`。
- `foldOnly: Bool = false`（public 实例属性）— C 最优开关，用户可配置、支持热更新（每轮 Chunk2 读取）。
- `rollback()` — 恢复到 `S_confirmed`。
- `rollbackToDraft(_ i:)` — 恢复到 `draftSnapshots[i]`（方案 D 的 partial 恢复）。
- `applyFolded(_:)` — 写入折叠结果（方案 C 的 partial 恢复）。
- `copy()` — 深拷贝三个新字段 + `foldOnly`。

**`Libraries/MLXLMCommon/GatedDelta.swift`**
- `computeGatedDeltaG(_:_:_:)`（public）— 计算衰减门 `g`。
- `gatedDeltaUpdateWithG(q:k:v:g:beta:state:mask:)`（public）— 直接吃 `g`/`β` 的状态更新，
  供多层堆 batch 维一次折叠。VLM/LLM 共用同一份。

**`Libraries/MLXLMCommon/LanguageModel.swift` — `MTPCapableModel` 协议**
- `foldPartialAccept(cache:draftIndex:) -> Bool`（**协议 body 要求** + extension 默认实现 `{ false }`）。
  ⚠️ 声明必须在协议 body 内，否则协议类型调用会静态派发到默认实现（见第 8 节协议派发坑）。

**`Libraries/MLXVLM/Models/Qwen35.swift`（VLM）**
- `GatedDeltaNet.callAsFunction` — Chunk2 逐 draft 循环，填充 snapshots/operators；
  Chunk1 清空上一轮存储；`cache.foldOnly` 时跳过 snapshots。
- `Qwen35.foldPartialAccept(cache:draftIndex:)`（public）— 全层 batch 折叠（优化点 1）。
- `Qwen35.loadMTPWeights()` — 注入 MTP 权重（绕过 MXFP4 包装）。
- `mtpForward(_:hiddenStates:cache:positionOffset:)` — draft 位置对齐。

**`Libraries/MLXLLM/Models/Qwen35.swift`（LLM）**
- `Qwen35GatedDeltaNet` Chunk2 逐 draft 循环，填充 snapshots/operators；`cache.foldOnly` 时跳过快照。
- `Qwen35TextModel.foldPartialAccept(cache:draftIndex:)`（public）— 堆 batch 维一次折叠、`applyFolded` 回写。
- `Qwen35Model.foldPartialAccept(...)` — 顶层转发到 `Qwen35TextModel`。
- MTP 权重在 sanitize 阶段加载，无需额外调用。

**测试**：`mlx_proj/test_mtp_v2/Sources/MTPv2Bench/MTPv2Bench.swift`
- 入口参数 `[llm|vlm] [modelPath]`；用 `LLMModelFactory` / `VLMModelFactory` 分别加载；
  VLM 加载后调 `loadMTPWeights()`，LLM 无需。
- `runBaseline`（关闭 MTP 纯自回归）、`runD1`（库现有 MTP）、`runD2`（快照 D / 变换 C / 自检）；
  三者面向 `MTPCapableModel` 协议，`foldOnly` 经 `(c as? MambaCache)?.foldOnly = ...` 设置。
