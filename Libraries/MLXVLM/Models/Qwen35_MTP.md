# Qwen3.5/3.6 MTP (Multi-Token Prediction) 支持文档

> 本文档记录了在 mlx-swift-lm 中为 Qwen3.5/3.6 VLM 模型实现 MTP 投机解码的完整过程，
> 包括架构分析、7 个关键 bug 的根因与修复，VLM 与 LLM 实现的对比，以及 StreamingKVCache 与 M-RoPE 的兼容性分析。

## 目录

- [1. MTP 架构概述](#1-mtp-架构概述)
- [2. 核心数据流](#2-核心数据流)
- [3. 遇到的 7 个问题及修复](#3-遇到的-7-个问题及修复)
- [4. VLM 与 LLM 实现对比](#4-vlm-与-llm-实现对比)
- [5. StreamingKVCache × M-RoPE 兼容性分析](#5-streamingkvcache--m-rope-兼容性分析)
- [6. 关键文件清单](#6-关键文件清单)

---

## 1. MTP 架构概述

Qwen3.5/3.6 的 MTP 头是一个**单 transformer 层**，用于在投机解码中快速预测下一个 token，
避免每次都运行完整的 64 层 backbone。

### 架构图

```
输入:
  hidden_states  [B, L, 5120]   ← backbone 最后一层的 hidden state (POST-norm)
  token_ids      [B, L]          ← 当前 token (主模型刚采样的)

流程:
  tokenEmb  = embedTokens(token_ids)              → [B, L, 5120]
  eNormed   = pre_fc_norm_embedding(tokenEmb)      → GemmaRMSNorm
  hNormed   = pre_fc_norm_hidden(hidden_states)    → GemmaRMSNorm
  combined  = concat([eNormed, hNormed], axis=-1)  → [B, L, 10240]
  x         = fc(combined)                         → [B, L, 5120]
  x         = MTPBlock(x)                          → attention + MLP + residuals
  x         = norm(x)                              → GemmaRMSNorm
  logits    = lmHead(x)                            → [B, L, 248320] (共享 lm_head)

输出: 下一个 token 的 logits
```

### 关键设计决策

| 设计 | 说明 |
|------|------|
| **Concat 顺序** | `[embedding, hidden]` — embedding 在前 (vLLM `qwen3_next_mtp` 确认) |
| **Norm 类型** | 所有 norm 层使用 **GemmaRMSNorm** (`x * (1+w) / rms(x)`) |
| **Attention** | 复用 backbone 的 `Attention` 类 (gated + M-RoPE)，维度 24h/4kv/256d |
| **Position IDs** | 从主模型 KV cache offset 动态计算 (非 MTP 自身 cache offset) |
| **Hidden States** | 使用 prepare 阶段的 hidden_{P-1} (POST-norm)，非 forwardWithHiddenStates 的 hidden_P |

---

## 2. 核心数据流

### 2.1 推测解码主循环 (Evaluate.swift)

```
prepare:
  1. 主模型处理完整 prompt → hidden_{P-1}, logits_P
  2. 采样 token_P → emit
  3. lastHiddenStates = hidden_{P-1}  ← 从 prepare 获取 (不调用 forwardWithHiddenStates)
  4. generateDraft: MTP(hidden_{P-1}, token_P) → draft_{P+1}

speculateRound (每个迭代):
  5. verify: forwardWithHiddenStates([token_P, draft_{P+1}]) → 首次处理 token_P
  6. 主模型预测 verifyToken (position 0 logits)
  7. if verifyToken == draftToken:
       ACCEPT → emit draft + bonus → generateDraft(cacheCommit)
     else:
       REJECT → rollback cache → emit verifyToken → generateDraft(nil)
```

### 2.2 Accept 后的 Cache Commit

当 draft 被接受时，MTP 需要将已接受的 token 提交到自己的 KV cache：

```
mtpForwardWithCommit:
  1. commit:  (hidden_at_confirmed, accepted_draft_token) → 更新 MTP KV cache
  2. current: (hidden_at_draft, bonus_token) → 完整 forward → 产生新 draft
```

### 2.3 Reject 后的 Cache Rollback

当 draft 被拒绝时：

```
1. 主模型 KV cache: trim(1) → 移除 draft 位置的 KV
2. GatedDeltaNet (MambaCache): rollback() → 恢复到 snapshot
3. MTP KV cache: 不 trim（MTP cache 只记录已确认的 token）
```

---

## 3. 遇到的 7 个问题及修复

### 问题 1: 权重加载 — `layers[0]` 数组子模块

**症状**: MTP 权重加载后 `layers[0]` 的权重值不匹配

**根因**: `Module.update()` 不处理数组子模块 (`layers[0]`)，`NestedItem` 的数组结构被跳过

**修复**: 先整体 `update`，再单独 `update` `layers[0]`:

```swift
// Qwen35.swift - loadMTPWeights()
try? mtp.update(parameters: ModuleParameters(item: nested), verify: .none)

// 单独 update layers[0]
if case .dictionary(let tree) = nested,
   case .array(let layerItems) = tree["layers"] ?? .none,
   layerItems.count > 0 {
    try? mtp.layers[0].update(
        parameters: ModuleParameters(item: layerItems[0]), verify: .none)
}
```

### 问题 2: Position IDs — MTP 用错误的 M-RoPE 位置

**症状**: MTP 输出 prob ≈ 0.001 (近均匀分布)，draft 全部错误

**根因**: `MTPBlock` 传 `positionIds: nil` 给 Attention，导致 MTP 用自身 KV cache offset (0,1,2...) 做 M-RoPE。但实际文本位置是 prompt_length, prompt_length+1, ... (如 25, 26, 27...)

**修复**: 从主模型 KV cache 动态读取位置，计算正确的 3D M-RoPE position IDs:

```swift
// Qwen35.swift - mtpForward()
let mtpPosition = mainCache[faIdx].offset  // 主模型 cache offset
let startPos = mtpPosition - L              // MTP 处理的 hidden states 的起始位置

// 构建 3D position IDs [3, B, L] (text, height, width)
var base = MLXArray(0 ..< L) + delta
let positionIds = broadcast(base[.newAxis, 0..., 0...], to: [3, B, L])
```

关键：用 `_mainCache` 引用而非静态值，因为 rejection 时 `trim(1)` 会改变 offset。

### 问题 3: Concat 顺序 — embedding 和 hidden 的顺序

**症状**: MTP 输出与主模型不相关 (cos ≈ 0)

**根因**: 最初写成 `[hNormed, eNormed]` (hidden 在前)，但 vLLM 参考实现确认正确顺序是 `[inputs_embeds, hidden_states]` (embedding 在前)

**修复**:

```swift
// vLLM qwen3_next_mtp 确认: torch.cat([inputs_embeds, hidden_states], dim=-1)
let combined = MLX.concatenated([eNormed, hNormed], axis: -1)
```

### 问题 4: Hidden State — 用 hidden_P 而非 hidden_{P-1}

**症状**: MTP 输入不匹配，输出与主模型不相关

**根因**: `forwardWithHiddenStates(token_P)` 返回 hidden_P (处理 token_P 之后的 hidden state)，但 MTP 架构要求 `hidden_t + token_{t+1} → predict token_{t+2}`，即需要 hidden_{P-1} (处理 token_P 之前的 hidden state)

**修复**: 不调用 `forwardWithHiddenStates`，直接使用 prepare 阶段的 hidden states:

```swift
// Evaluate.swift - prepare (.logits case)
// MTP 需要 hidden_{P-1}（prepare 时的 hidden state），不是 hidden_P
if let allHidden = model.lastForwardHiddenStates {
    let seqLen = allHidden.dim(1)
    lastHiddenStates = allHidden[0..., (seqLen - 1)..., 0...]
}
```

同时添加 `MTPCapableModel.lastForwardHiddenStates` 协议属性，由 Qwen35 实现。

### 问题 5: 重复处理 token_P

**症状**: token_P 被 `forwardWithHiddenStates` 和 verify 各处理一次，导致 KV cache 冲突

**根因**: 在 prepare 中调用 `forwardWithHiddenStates(token_P)` 处理 token_P，然后 verify 又处理 `[token_P, draft]`，token_P 被处理两次

**修复**: 去掉 prepare 中的 `forwardWithHiddenStates` 调用，让 verify 首次处理 token_P:

```swift
// 之前: forwardWithHiddenStates(token_P) → hidden_P (token_P 被处理)
// 之后: 直接用 prepare 的 hidden_{P-1}，token_P 在 verify 中首次处理
```

### 问题 6: PRE-norm vs POST-norm

**症状**: 改为 PRE-norm 后 cos 从 ≈0 变为 ≈-0.5 (反相关)

**根因**: 误以为 MTP 需要 PRE-norm hidden states，但 vLLM 参考实现确认主模型返回 POST-norm hidden states 给 MTP。MTP 的 `pre_fc_norm_hidden` 会再次 normalize (double-norm 是正确行为)

**修复**: 恢复 POST-norm:

```swift
// Model.callAsFunction: return norm(hiddenStates)  // POST-norm
// LanguageModel: _lastHiddenStates = hiddenStates   // POST-norm 给 MTP
```

### 问题 7: GemmaRMSNorm — 最终根因 ⭐

**症状**: MTP 输出与主模型**反相关** (cos ≈ -0.5)，所有修复都验证生效但输出仍然不对

**根因**: Qwen3.5/3.6 的所有 norm 层使用 **GemmaRMSNorm** (`x * (1+w) / rms(x)`)，而非标准 RMSNorm (`x * w / rms(x)`)。checkpoint 中的权重 `w` 训练时以 `1+w` 为有效缩放因子。

关键证据 — `pre_fc_norm_embedding` 权重 **全为负值** (mean=-0.44):

| Norm 层 | checkpoint 权重 (w) | 标准 RMSNorm (=w) | GemmaRMSNorm (=1+w) | 结果 |
|---------|---------------------|--------------------|--------------------|------|
| pre_fc_norm_embedding | mean=**-0.44** | **-0.44** (符号翻转!) | **0.56** (正确) | ❌→✅ |
| pre_fc_norm_hidden | mean=**-0.17** | **-0.17** (部分翻转) | **0.83** (正确) | ❌→✅ |
| input_layernorm | mean=**0.04** | **0.04** (25x太小) | **1.04** (正确) | ❌→✅ |
| post_attention_layernorm | mean=**0.21** | **0.21** (5x太小) | **1.21** (正确) | ❌→✅ |

用标准 RMSNorm 时，`pre_fc_norm_embedding` 的有效权重为 **-0.44** 而非 **0.56**，导致 embedding 方向**完全翻转** → MTP 收到的输入是反的 → 输出反相关。

**修复**: 创建 `GemmaRMSNorm` 类，替换所有 MTP norm 层:

```swift
final class GemmaRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float

    init(dimensions: Int, eps: Float = 1e-6) {
        _weight.wrappedValue = MLXArray.zeros([dimensions])  // 初始化为 0 (有效初始值 = 1+0 = 1)
        self.eps = eps
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let rms = sqrt(mean(x * x, axis: -1, keepDims: true) + eps)
        return x / rms * (1.0 + weight)  // 关键: (1 + w) 而非 w
    }
}
```

### 问题 7.5 (附带): mtpForwardWithCommit 返回维度错误

**症状**: accept 后 crash — `SmallVector out of range`

**根因**: 默认实现 `logits[0..., -1, 0...]` 用 `-1` 删除了中间维度 (3D→2D)，然后 `[.newAxis, .newAxis]` 没有正确恢复 3D

**修复**: 用 range slice 保持维度:

```swift
// 修复前: logits[0..., -1, 0...][.newAxis, .newAxis]  // 丢失维度
// 修复后:
let lastIdx = logits.dim(1) - 1
return logits[0..., lastIdx ..< (lastIdx + 1), 0...]  // 保持 3D [1, 1, vocab]
```

---

## 4. VLM 与 LLM 实现对比

| 方面 | VLM (`MLXVLM/Qwen35.swift`) | LLM (`MLXLLM/Qwen35.swift` + `MTPHead.swift`) |
|------|------------------------------|------------------------------------------------|
| **Norm 类型** | `GemmaRMSNorm` (自定义类，运行时处理) | `RMSNorm` (标准) + `sanitize` 时权重 +1 |
| **GemmaRMSNorm 处理** | 运行时：`x/rms(x) * (1+w)` | 加载时：`w → w+1`，然后用标准 RMSNorm | 
| **Attention 类** | 复用 backbone `Attention` (gated + M-RoPE) | 专用 `Qwen35Attention` (标准 RoPE) |
| **Position IDs** | 从主模型 cache 动态计算 3D M-RoPE | 不需要显式 position IDs (标准 RoPE 从 cache offset 推导) |
| **mtpForward 签名** | `mtpForward(tokenIds, hiddenStates, cache, positionIds)` | `mtpForward(tokenIds, hiddenStates, cache)` (无 positionIds) |
| **mtpForwardWithCommit** | 用协议默认实现 (concat 2 tokens) | **自定义优化实现** (分开处理 commit + current，省一次 lm_head) |
| **Hidden States 对齐** | 不需要 (从 prepare 获取) | `alignedHidden = hiddenStates[0..., (hiddenLen - tokenLen)..., 0...]` (手动对齐) |
| **Vision 支持** | 有 (pixel_values, image_grid_thw 等) | 无 |
| **M-RoPE** | 需要 (3D position IDs: text/height/width) | 不需要 (纯文本用 1D RoPE) |
| **ropeDeltas** | 需要 (图像位置偏移) | 不需要 |

### 4.1 两种 GemmaRMSNorm 处理方式的对比

| | LLM 版本 (sanitize) | VLM 版本 (GemmaRMSNorm 类) |
|---|---|---|
| **方法** | 加载时 `w → w + 1`，用标准 RMSNorm | 保持原始 `w`，用 `GemmaRMSNorm` 类 |
| **公式** | `x/rms(x) * (w+1)` | `x/rms(x) * (1+w)` |
| **数学等价** | ✅ 相同 | ✅ 相同 |
| **修改范围** | 所有 norm 权重（含主模型） | 只改 MTP 的 norm 层 |
| **优点** | 无需自定义类，改动小 | 不修改权重，更直观 |
| **缺点** | 修改了原始权重值 | 需要自定义 `GemmaRMSNorm` 类 |

### 4.2 LLM 版本的优化: 自定义 mtpForwardWithCommit

LLM 版本有一个**自定义的 `mtpForwardWithCommit`**，比协议默认实现更高效：

```swift
// LLM 版本: 分开处理 commit 和 current
// Step 1: Commit (只更新 KV cache，跳过 norm + lm_head)
let commitEmb = model.embedTokens(commitToken.reshaped(1, 1))
// ... 只过 MTPBlock，不过 norm + lm_head

// Step 2: Current (完整 forward)
let tokenEmb = model.embedTokens(tokenIds)
// ... 完整 forward 包括 norm + lm_head
```

**优势**: 省一次 `lm_head` 矩阵乘 (hidden_size × vocab_size = 5120 × 248320)，
对 27B 模型每次 accept 节省约 2.5 GFLOPs。

VLM 版本目前用协议默认实现，未来可以借鉴 LLM 版本的优化。

---

## 5. StreamingKVCache × M-RoPE 兼容性分析

### 5.1 StreamingKVCache 工作原理

`StreamingKVCache` 采用 **Attention Sink + 滑动窗口** 策略：

1. 永久保留开头的少量 **sink token**（`keep` 个，如 system prompt 或前几个 token）
2. 只保留最近的 `windowSize` 个 token
3. **关键操作**：淘汰旧 token 后，对幸存窗口 token 的 key 做**均匀 RoPE 旋转移位**（`applyUniformRoPEShift`），把位置平移回 `[keep, keep+windowSize)` 区间

核心数学恒等式（`StreamingKVCacheTests.swift` 验证）：
```
R(δ) · R(p) = R(p + δ)
```
即：对已烘焙的 key 施加均匀旋转，等价于在新位置重新烘焙。

### 5.2 M-RoPE 的工作原理

VLM 使用 **M-RoPE**（multidimensional rotary position embedding），位置 IDs 是 3D 的：
```
positionIds = [3, batch, seq]   ← [text_position, image_height_position, image_width_position]
```

不同频率维度使用不同的位置分量（由 `mropeSection: [11, 11, 10]` 控制）：
- 前 11 个基础频率 → `text_position`
- 中间 11 个 → `image_height`
- 后 10 个 → `image_width`

### 5.3 关键分析

`applyUniformRoPEShift` 对**所有** rotary 维度施加相同的旋转角度 δ。这个操作：

**对标准 1D RoPE 成立** ✅：
```
每个维度 i: angle_i = δ / θ_i
所有维度用同一个 δ → uniform shift 正确
```

**对 M-RoPE 需要分情况讨论**：

#### 情况 A: 纯文本生成 (text_position = height = width)

所有位置分量相同，M-RoPE ≈ 1D RoPE：
```
positionIds = [3, 1, 1] = [[p], [p], [p]]   ← 三个维度相等
→ 所有频率维度用同一个位置 p
→ uniform shift 正确 ✅
```

#### 情况 B: 图像 tokens 在 sink 区域（保留不淘汰）

```
Sink: [image_token_0, image_token_1, ..., system_prompt, text_0, text_1]  ← 混合 3D 位置
Window: [text_k, text_{k+1}, ..., text_n]                                   ← 纯文本 (3D 相同)

evict 后:
- Sink tokens: 不移动，位置不变 ✅
- Window tokens: 纯文本 → shift 正确 ✅
```

#### 情况 C: 图像 tokens 在 window 区域内（会被淘汰 / 移位）

```
Window: [image_token, ..., text_0, text_1]

evict 后如果 image tokens 幸存：
→ shift 会改变 height/width 维度的位置（但这些维度不应跟随 text 维度 shift）
→ ⚠️ 产生偏差
```

但实际场景中**图像几乎总是在 conversation 开头**（system prompt 或第一轮 user message），
属于 sink 区域或最早被淘汰的内容，不会出现在需要 shift 的 window 中。

### 5.4 量化分析

即使极端情况下图像 tokens 需要 shift，偏差也很有限：

| 参数 | 值 |
|------|-----|
| Head dim | 256 |
| `partialRotaryFactor` | 0.25 |
| Rotary dims | 256 × 0.25 = 64 |
| M-RoPE 影响的频率分量 | 32 个（每个分量对应 2 个 rotary dim） |
| M-RoPE 影响的 rotary dims | 64 (全部 rotary) |
| **受潜在 shift 偏差影响的比例** | 64/256 = 25% |
| 但 text 分量 (11/32) shift 正确 | 11/32 = 34% 正确 |
| 实际偏差比例 | **~41%** 的 rotary dimensions |
| 偏差占全部 head dim 的比例 | 64×0.41/256 ≈ **10%** |

### 5.5 `ropeDeltas` 和 `precomputedPositionIds` 的处理

StreamingKVCache eviction 后，还需要处理两个位置相关状态：

#### ropeDeltas

```swift
// Qwen35.swift: ropeDeltas = 图像 tokens 引入的位置偏移
var delta = MLXArray(cacheOffset) + (ropeDeltas ?? 0)
```

eviction 后 `cacheOffset` 减少了 `evictCount`，`ropeDeltas` 也需要相应调整：
- 如果被淘汰的 tokens 不涉及图像 → `ropeDeltas` 不变
- 如果被淘汰的 tokens 包含图像 → `ropeDeltas` 需要重新计算

#### precomputedPositionIds

prepare 阶段计算的 `precomputedPositionIds` 会被缓存到 `state` 中，用于后续 forward。
eviction 后这些预计算的位置 IDs 不再有效，需要重新计算或增量更新。

### 5.6 结论

| 场景 | StreamingKVCache 兼容性 | 说明 |
|------|:---:|------|
| 纯文本对话 | ✅ 完全兼容 | M-RoPE = 1D RoPE，shift 正确 |
| 图像在 sink 区域 | ✅ 兼容 | Sink 不 shift，window 纯文本 |
| 图像在 window 区域且被淘汰 | ✅ 无影响 | 被淘汰了就无所谓 |
| 图像在 window 区域且幸存 | ⚠️ 有偏差 | 图像相关维度 shift 不正确 (~10% dims) |
| 多轮图像 (每轮有不同图) | ⚠️ 需谨慎 | `ropeDeltas` 和 `precomputedPositionIds` 需额外处理 |

**推荐策略**：
1. 将 system prompt 和第一轮（含图片）设为 sink（`keep = 第一轮 token 数`）
2. 后续纯文本对话在 window 内流转
3. 如果必须支持多轮图像，每次 evict 后需要重新计算 `ropeDeltas` 和 `precomputedPositionIds`

**StreamingKVCache 文档中的 "⚠️ 不建议" 标记**可以更新为："纯文本和 sink-protected 图像场景下支持，需额外处理 ropeDeltas 和 positionIds 缓存"。

---

## 6. 关键文件清单

| 文件 | 修改内容 |
|------|----------|
| `MLXVLM/Models/Qwen35.swift` | GemmaRMSNorm 类、MTPBlock、MTPModule、mtpForward (position IDs)、forwardWithHiddenStates、lastForwardHiddenStates、prepare (_mainCache)、loadMTPWeights |
| `MLXLMCommon/Evaluate.swift` | MTPSpeculativeTokenIterator: prepare (.logits case 用 lastForwardHiddenStates)、speculateRound、generateDraft |
| `MLXLMCommon/LanguageModel.swift` | MTPCapableModel 协议 + lastForwardHiddenStates 属性 + mtpForwardWithCommit 默认实现修复 (维度保持) |
| `MLXLLM/Models/Qwen35.swift` | LLM 版本 (sanitize 中 shouldShiftNormWeights 处理 GemmaRMSNorm) |
| `MLXLLM/Models/MTPHead.swift` | LLM MTP 组件 (标准 RMSNorm + 标准 RoPE) |

---

## 参考实现

- **vLLM**: `vllm/model_executor/models/qwen3_next_mtp.py` — 确认 concat 顺序 [embed, hidden]、GemmaRMSNorm
- **vLLM**: `vllm/model_executor/models/qwen3_next.py` — 确认 Qwen3NextDecoderLayer、Qwen3NextRMSNorm = GemmaRMSNorm
- **vLLM**: `vllm/model_executor/layers/layernorm.py` — GemmaRMSNorm 公式 `x * (1 + w) / rms(x)`
- **mlx-qwen-mtp**: https://github.com/quivent/mlx-qwen-mtp — 确认架构 `hidden_t + token_{t+1} → token_{t+2}`、concat [embed, hidden]
- **SGLang**: `--speculative-algo NEXTN` — 推测解码配置参考
- **StreamingLLM**: Xiao et al. 2023 — Attention Sink + 滑动窗口策略基础
