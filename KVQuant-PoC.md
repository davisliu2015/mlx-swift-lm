# KV Cache 8-bit 量化 PoC —— 库层实现说明（mlx-swift-lm fork）

> 状态：**PoC 完成，暂停封存**（2026-08-30）
> 分支：`feature/quantize_v1`
> 涉及文件：`Libraries/MLXLMCommon/KVCache.swift`

本文档记录本 fork 中为「KV cache 8-bit 量化」所做的库层改动、设计取舍、验证进展，以及暂停原因。上层 app（smlx）的对应说明见 smlx 仓库的 `KVQuant-PoC.md`。

---

## 1. 目标

在保留 StreamingLLM 滑动窗口淘汰（sink + window + RoPE 重索引）语义的前提下，把 KV cache 从全精度压到 8-bit，理论上省约 50% 的 KV 显存。

难点：现有 `QuantizedKVCache` 是普通量化 cache，**不支持 evict / RoPE-shift**；而 `StreamingKVCache` 支持滑窗淘汰但不量化。两者能力互斥，需要一个「既量化、又能滑窗淘汰」的新类型。

---

## 2. 核心实现

### 2.1 新增 `QuantizedStreamingKVCache`

`StreamingKVCache` 的量化子类，同时实现 `QuantizedKVCacheProtocol`：

- **存储/attention 引擎**：内部持有一个 `QuantizedKVCache`（字段 `quant`），所有 `update` 和 attention 都走它。父类的明文 `keys`/`values` 平时保持 `nil`，只在 evict 期间短暂反量化使用。
- **attention 透明**：因为实现了 `QuantizedKVCacheProtocol`，`attentionWithCacheUpdate` 会自动路由到 `quantizedScaledDotProductAttention`，**模型侧零改动**。
- **可从现有 `StreamingKVCache` 构造**（`init(from:bits:groupSize:)`），用于运行中原地替换。

### 2.2 淘汰策略（方式 1：正确性优先）

`evict(tokenCount:)` 采用最简单且正确的策略：

1. 把当前有效区间**全量反量化**回明文；
2. 委托父类的明文 `evict(tokenCount:)`（**原样复用**其全部 RoPE / M-RoPE 重索引逻辑）；
3. 对存活 token **重新量化**。

代价：evict 期间峰值内存会短暂涨到全精度 KV 大小（但 evict 是低频操作），evict 之间常驻内存保持量化。**未来优化**：只反量化 window 内的 key，做增量量化。

### 2.3 `maybeQuantizeKVCache` 分派修正

- `StreamingKVCache` 必须在 `KVCacheSimple` **之前**判断（前者是后者子类），转换成 `QuantizedStreamingKVCache` 而非普通 `QuantizedKVCache`（否则会丢掉 evict / RoPE-shift 能力）。
- `QuantizedStreamingKVCache` 不是 `QuantizedKVCache` 的子类，需在 guard 里**显式排除**，避免对已量化的 streaming cache 二次量化。

### 2.4 访问级别放宽

为让子类复用父类逻辑，把 `StreamingKVCache` 若干成员从 `private` 放宽为 `internal`：
`keep` / `windowSize`（改为 `internal(set)`）、`ropeDimensions` / `ropeBase` / `ropeTraditional` / `ropeScale` / `mropeSection`，并新增 `adoptImageMask(from:)` 复制图像 token 掩码。

### 2.5 `copy()` / `trim()` / `innerState()`

- `copy()`：重建量化引擎并复制 `state` / `metaState` / imageMask —— 保证 prompt cache 的持久化/复用正确。
- `trim(n)`：只改 offset（O(1)），与明文 cache 一致，**MTP 部分接受回滚不产生额外量化开销**。

---

## 3. 临时自检代码（`SMLX_KVQUANT_SELFCHECK`）

`evict()` 内有一段数值自检，**仅当环境变量 `SMLX_KVQUANT_SELFCHECK=1` 时启用**：

- 校验「反量化 → RoPE 旋转 → 重量化」这条链的数值误差；
- 对比重量化前后 K 的相对误差 `relErr`，打印 `TEMP-KVQUANT-SELFCHECK evict=... K relErr=... [OK/HIGH]`（阈值 0.05）。

所有临时代码都用 `⚠️ TEMP-KVQUANT-SELFCHECK BEGIN/END ⚠️` 包裹，方便 grep + 整段删除。

---

## 4. 验证进展

| 项 | 状态 |
|---|---|
| 文本路径量化收发 | ✅ 正常（8-bit / groupSize=64 / start=0）|
| prompt cache 复用（copy/state） | ✅ 正常（`cacheHit=true`，offset 同步正确）|
| 与 MTP（D1）共存 | ✅ 无回滚报错，量化 cache 快照/state 配合正常 |
| **evict 数值正确性（`K relErr`）** | ⚠️ **未验证**（自检代码就绪，但未跑到真实 evict）|
| VLM（M-RoPE 视觉路径）量化 | ❌ 未覆盖 |

---

## 5. 性能观察（配合 smlx 实测）

- **量化本身对纯自回归解码几乎零开销**（无 MTP 时 ON≈OFF）。
- **量化不影响 MTP 在代码/数学类场景的加速**（D1 下量化≈不量化）。
- **自然语言场景 + MTP 下量化明显变慢**。根因：`QuantizedKVCache.updateQuantized` 每次写入是「反量化+拼接+重量化整段」，成本约 O(当前 cache 总长)，而非 O(新增 token)；MTP 在自然语言下接受率低 → verify 步数多 → 高频触发这条 O(总长) 的重量化，被放大。**不是** rollback/trim 被放大（trim 只改 offset）。

---

## 6. 暂停原因

1. **目标与手段错配**：诉求是「提速」，但量化本质是「用算力换显存」，最好情况只能做到「不拖速地省显存」，给不了速度提升。
2. **自然语言场景有明显速度回退**（见第 5 节）。
3. **evict 正确性未验证**，VLM 未覆盖。
4. **真实显存收益未实测**（省 50% 仍是理论推算，尚不确定显存是否为瓶颈）。

结论：先**封存**，代码保留在 `feature/quantize_v1` 分支，待「显存确为瓶颈」时再捡起。

---

## 7. 后续 TODO（捡起时）

- [ ] 跑真实 evict，抓 `K relErr` 确认数值正确性。
- [ ] 实测量化 ON/OFF 的真实进程内存，坐实显存收益。
- [ ] 优化 `updateQuantized`：改为只量化新增 token 的增量追加（O(新增)），消除自然语言 + MTP 的高频重量化开销。
- [ ] evict 优化：只反量化 window 内 key，避免全量反量化。
- [ ] 扩展到 VLM（M-RoPE）路径。
- [ ] 转正：删除所有 `TEMP-KVQUANT-SELFCHECK` 临时代码，量化参数改为 per-model 正式配置。
