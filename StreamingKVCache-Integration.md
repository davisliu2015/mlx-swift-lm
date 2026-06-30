# StreamingKVCache 上层接入指南（给 smlx）

本文面向 **上层调用方（smlx）**，说明如何使用 `mlx-swift-lm` 中新增的
`StreamingKVCache`，在 **不做全量 re-prefill** 的前提下，让对话可以「无限」进行下去而不会
撞上 context size 上限、也不会因为 RoPE 外推导致质量崩塌。

> 代码位置：`Libraries/MLXLMCommon/KVCache.swift`
> 单测：`Tests/MLXLMTests/StreamingKVCacheTests.swift`（11 个用例，纯数值自验证，无需下载模型）

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
淘汰是一个**显式动作**，由你在安全点调用——典型时机是**每轮对话结束、下一次 prefill 之前**。

框架提供两种淘汰方式：

#### A. `evict(tokenCount:)` — 精确控制丢弃数量（推荐）

上层计算好要丢多少个 token（对齐到消息边界后），直接告诉 cache：

```swift
// 返回实际淘汰的 token 数（可能被 clamp 到 evictableCount）
let evicted = (c as? StreamingKVCache)?.evict(tokenCount: tokensToEvict) ?? 0
```

- `evictableCount` 属性：当前可淘汰的非 sink token 总数（`offset - keep`）。
- 如果请求数超过 `evictableCount`，自动 clamp，不会 crash。
- 淘汰后 `offset` 减少 `evicted`（不一定等于 `capacity`）。

#### B. `evictToWindow()` — 一步砍到 capacity（懒人版）

不关心消息边界，直接砍到 `capacity = keep + windowSize`：

```swift
for c in cache {
    (c as? StreamingKVCache)?.evictToWindow()
}
```

等价于 `evict(tokenCount: offset - capacity)`。

---

### 4.4 消息边界对齐（上层主动控制裁切）

**`evict(tokenCount:)` 是 token 级原语**，它不知道"消息"的概念。如果直接用
`evictToWindow()`，窗口边界可能恰好落在一条消息中间，导致半截消息留在 cache 里，理解出偏差。

**推荐做法：上层（smlx）维护每条消息的 token 起止位置，裁切时对齐到消息边界。**

```swift
/// 消息 token 范围示例
struct MessageTokenRange {
    let messageIndex: Int
    let tokenStart: Int   // 在完整 prompt token 序列中的起始位置
    let tokenEnd: Int     // 在完整 prompt token 序列中的结束位置（不含）
}

/// 上层淘汰逻辑
func evictAlignedToMessageBoundary(
    cache: [KVCache],
    messageRanges: [MessageTokenRange],
    keep: Int  // sink token 数
) -> Int {  // 返回丢弃的完整消息数
    guard let first = cache.first as? StreamingKVCache,
          first.needsEviction else { return 0 }

    // 需要丢弃的最小 token 数
    let minEvict = first.offset - first.capacity

    // 向后对齐到消息边界：找到「丢完后，第一条保留消息的 tokenStart」
    var tokensToEvict = 0
    var messagesDropped = 0
    for range in messageRanges {
        // 跳过 sink 区域内的消息（它们被 keep 保护，不会被丢）
        if range.tokenEnd <= keep { continue }
        let msgTokens = range.tokenEnd - max(range.tokenStart, keep)
        if tokensToEvict + msgTokens <= minEvict || tokensToEvict < minEvict {
            tokensToEvict += msgTokens
            messagesDropped += 1
        } else {
            break
        }
    }
    // 确保至少丢够 minEvict 个 token（宁可多丢一条完整消息，也不截断）
    if tokensToEvict < minEvict {
        // 需要多丢一条消息来覆盖
        // ...根据实际情况继续累加
    }

    // 对所有层执行相同的淘汰
    for c in cache {
        (c as? StreamingKVCache)?.evict(tokenCount: tokensToEvict)
    }

    // 从消息列表中同步删除被丢弃的消息
    // messageRanges.removeFirst(messagesDropped)
    return messagesDropped
}
```

**关键**：淘汰后，下一次构建 prompt 时，**不要再包含被丢弃的消息**。因为框架的
`LLMModel.prepare()` 靠 `cache.offset` 做"盲跳"（跳过前 `offset` 个 token），它假设
prompt 的前 `offset` 个 token 与 cache 中的 KV 完全一致。如果 smlx 仍然带上已被丢弃的
消息，token 序列就跟 cache 内容对不上，会产生错误输出。

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
3. 每轮生成结束后、下一轮 prefill 之前：
   - （推荐）算好要丢几条完整消息的 token 数，调 `evict(tokenCount:)`；
   - （简单版）直接调 `evictToWindow()` 一刀切到 capacity；
4. **同步删除被丢弃的消息**，下次构建 prompt 不再包含它们；
5. 仅适用于标准 / 线性 RoPE，且接受「远期上下文有损遗忘」。

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
- `evictExactTokenCount`：`evict(tokenCount:)` 精确裁切指定数量，结果与 ground truth 一致；
- `evictClampsToAvailable`：请求超额时自动 clamp 到可用量，不 crash；
- `evictToWindowMatchesExplicitEvict`：`evictToWindow()` 等价于 `evict(tokenCount: overflow)`；
- `serializationRoundTripPreservesConfigAndState` / `copyIsIndependent`：序列化与拷贝语义正确。
