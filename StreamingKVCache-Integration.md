# StreamingKVCache 上层接入指南（给 smlx）

本文面向 **上层调用方（smlx）**，说明如何使用 `mlx-swift-lm` 中新增的
`StreamingKVCache`，在 **不做全量 re-prefill** 的前提下，让对话可以「无限」进行下去而不会
撞上 context size 上限、也不会因为 RoPE 外推导致质量崩塌。

> 代码位置：`Libraries/MLXLMCommon/KVCache.swift`
> 单测：`Tests/MLXLMTests/StreamingKVCacheTests.swift`（8 个用例，纯数值自验证，无需下载模型）

---

## 1. 它解决什么问题

普通的 `KVCacheSimple` 是**只增不减**的：它的 `offset` 同时承担「有效长度 / 物理槽位 /
RoPE 位置」三重身份，所以**无法丢弃早期 token**。当对话长度逼近模型 context 上限时，传统做法
是把旧消息裁掉后**重新 prefill**，代价极高。

`RotatingKVCache` 虽然能丢中间 token，但它保留每个幸存 token 的**原始绝对 RoPE 位置**，
于是 query 到旧 token 的距离会越拉越大，最终超出模型训练时见过的范围（RoPE 外推 → 质量崩塌）。

`StreamingKVCache` 采用 **StreamingLLM（Attention Sink + 滑动窗口）** 策略：

- 永久保留开头的少量 **sink token**（通常是 system prompt / 最初几个 token）；
- 只保留最近的 `windowSize` 个 token；
- **关键**：淘汰旧 token 后，把幸存窗口 token 的 RoPE 位置**整体平移**回一段**有界、连续**的
  区间。逻辑位置永远不超过 `keep + windowSize`，模型始终处于「分布内」，可以一直聊下去；
- 整个过程**不重新 prefill**，只对保留下来的 key 做一次 O(N) 的均匀旋转。

布局始终是连续的：`[ sink (≤keep) | window (≤windowSize) ]`。

---

## 2. 适用范围（务必先确认）

`StreamingKVCache` 依赖一个数学事实：均匀旋转一个已经「烘焙」过位置的 key，等价于把它在新位置
重新烘焙（`R(δ·θ)·R(p·θ) = R((p+δ)·θ)`）。这个等式**只对标准 / 线性缩放的 RoPE 成立**。

| RoPE 类型 | 是否支持 | 说明 |
|---|---|---|
| `default`（标准 RoPE，如 Qwen3、未启用动态缩放的 Llama） | ✅ 支持 | 推荐 |
| `linear`（线性缩放） | ✅ 支持 | 传入 `ropeScale = 1/factor` |
| `yarn` / `longrope` / `llama3`（动态/分段缩放） | ❌ 不支持 | 有效频率依赖序列长度区间，旋转不再保位置不变 |
| `mrope`（多模态，Qwen2-VL 等） | ⚠️ 不建议 | 位置语义复杂，未验证 |

另外它是 **有损** 的：被淘汰的 token 彻底丢失，后续注意力再也看不到它们。请在产品层面接受
「远期上下文遗忘、保留 system prompt + 最近窗口」这一行为。

---

## 3. 构造参数

```swift
public init(
    keep: Int = 4,           // 永久保留的 sink token 数（StreamingLLM 建议 ~4）
    windowSize: Int,         // 淘汰后保留的最近 token 数（滑动窗口大小）
    ropeDimensions: Int,     // RoPE 旋转维度，通常等于模型的 head_dim
    ropeBase: Float,         // RoPE theta base，即 config 的 rope_theta
    ropeTraditional: Bool = false,  // 是否为 interleaved(GPT-J) 布局，绝大多数模型为 false
    ropeScale: Float = 1.0   // 位置缩放：无缩放为 1.0；linear 缩放传 1/factor
)
```

- `capacity = keep + windowSize`：淘汰后缓存最多保留这么多 token，也是逻辑位置的上界。
- 这些 RoPE 参数**必须和模型实际使用的 RoPE 完全一致**，否则平移后的 key 与模型预期不符。
  它们来自模型的 `config.json`：`head_dim`、`rope_theta`、`rope_scaling`。

---

## 4. 标准接入流程

### 4.1 为每一层创建一个 cache（关键）

模型每层都有独立的 KV cache，数量等于 `kvHeads.count`（即层数）。所以要构造一个**数组**，
每个元素都是一个 `StreamingKVCache`，配置相同：

```swift
import MLXLMCommon

/// 从模型 config 派生 RoPE 参数（以 Qwen3 为例）
func makeStreamingCache(
    numLayers: Int,
    headDim: Int,
    ropeTheta: Float,
    ropeScaling: [String: StringOrNumber]?,
    keep: Int = 4,
    windowSize: Int
) -> [KVCache] {
    // 线性缩放：scale = 1/factor；否则 1.0
    var ropeScale: Float = 1.0
    if let s = ropeScaling, s["type"] == .string("linear"),
       let factor = s["factor"]?.asFloat() {
        ropeScale = 1 / factor
    }

    return (0 ..< numLayers).map { _ in
        StreamingKVCache(
            keep: keep,
            windowSize: windowSize,
            ropeDimensions: headDim,
            ropeBase: ropeTheta,
            ropeTraditional: false,
            ropeScale: ropeScale
        )
    }
}
```

> `numLayers` 可用模型的 `kvHeads.count` 获取（模型实现了 `KVCacheDimensionProvider`）。
> `headDim` / `ropeTheta` / `ropeScaling` 来自该模型的 Configuration（即 `config.json`）。

### 4.2 把自定义 cache 注入生成流程

`TokenIterator` 和 `generate(...)` 都接受可选的 `cache: [KVCache]?`，传入即可，**无需改模型代码**：

```swift
let cache = makeStreamingCache(
    numLayers: model.kvHeads.count,
    headDim: config.headDim,
    ropeTheta: config.ropeTheta,
    ropeScaling: config.ropeScaling,
    keep: 4,
    windowSize: 4096
)

// 方式 A：直接用 TokenIterator
let iterator = try TokenIterator(
    input: input, model: model, cache: cache, parameters: params)

// 方式 B：用高层 generate（同一个 cache 数组贯穿多轮复用）
let stream = try generate(
    input: input, cache: cache, parameters: params, context: context)
```

### 4.3 在「安全边界」显式淘汰

**`update(keys:values:)` 只负责追加**（和 `KVCacheSimple` 完全一致），保证解码热路径简单正确。
淘汰是一个**显式动作**，由你在安全点调用——典型时机是**每轮对话结束、下一次 prefill 之前**：

```swift
// 一轮生成结束后、准备处理下一条用户消息之前：
for c in cache {
    (c as? StreamingKVCache)?.evictToWindow()
}
```

`evictToWindow()` 的语义：

- 若当前有效 token 数 `≤ capacity`：**空操作**；
- 否则：丢弃 `[keep, keep+evict)` 这段最旧的非 sink token，把 `[keep+evict, valid)` 的幸存
  窗口 token **整体下移** `evict` 个位置（key 做均匀 RoPE 旋转，value 不含位置信息、原样保留），
  sink token `[0, keep)` **原封不动**。完成后 `offset == capacity`。

也可以先判断再调用：

```swift
if let s = c as? StreamingKVCache, s.needsEviction {
    s.evictToWindow()
}
```

> ⚠️ 不要在解码每一步都调 `evictToWindow()`（虽然单测验证了它在逐 token 循环里也能保持
> 位置有界且数值稳定），更自然的做法是在**轮与轮之间**淘汰一次，减少不必要的旋转。

---

## 5. `keep` / `windowSize` 怎么选

- `keep`：StreamingLLM 论文经验值 **4** 即可显著优于「无 sink 的纯窗口」。如果你希望整段
  system prompt 都作为 sink 永久保留，就把 `keep` 设为 **system prompt 的 token 数**。
- `windowSize`：保留多少最近上下文，越大越「记得久」，但显存和注意力计算也越多。
  建议 `keep + windowSize` 留出余量、**小于模型训练 context**（例如模型支持 32k，可设
  `windowSize = 8192~16384`）。

> 一个实用约定：`keep` = system prompt 长度；`windowSize` = 你愿意为「最近对话」付出的显存预算。

---

## 6. 序列化（持久化 / 恢复）

`StreamingKVCache` 已接入框架的 prompt cache 存取，可与 `savePromptCache` /
`loadPromptCache` 一起用，`metaState` 会完整保存配置（`keep / windowSize / rope 参数`）：

```swift
try savePromptCache(url: url, cache: cache, metadata: [:])
let (restored, meta) = try loadPromptCache(url: url)
// restored[i] 仍是配置完好的 StreamingKVCache，可继续 update / evictToWindow
```

---

## 7. 一句话总结调用顺序

1. 按层数构造 `[StreamingKVCache(...)]`，RoPE 参数来自模型 config；
2. 把这个数组通过 `cache:` 传给 `TokenIterator` / `generate`，**跨轮复用同一个数组**；
3. 每轮生成结束后、下一轮 prefill 之前，对每个 cache 调一次 `evictToWindow()`；
4. 仅适用于标准 / 线性 RoPE，且接受「远期上下文有损遗忘」。

---

## 8. 正确性如何保证（可机械复现）

`Tests/MLXLMTests/StreamingKVCacheTests.swift` 用框架自身的 `MLXFast.RoPE` 构造 ground truth，
断言数值等价，全部自包含、无需下载模型：

```bash
swift test --filter StreamingKVCacheTests
```

覆盖点：

- `uniformShiftMatchesRebaking`：核心恒等式 —— 均匀平移 == 在新位置重新烘焙；
- `evictionProducesCorrectlyReindexedCache`：端到端淘汰结果与「第一性原理」重建一致；
- `sinkTokensAreUntouched`：sink 跨淘汰逐位不变；
- `noEvictionWhenWithinCapacity`：未超容量时为空操作；
- `repeatedEvictionStaysBounded`：长解码循环中位置始终 ≤ capacity 且无 NaN；
- `serializationRoundTripPreservesConfigAndState` / `copyIsIndependent`：序列化与拷贝语义正确。
