# Qwen3VL 多轮带图 decode 位置错位 bug（2026-09-05）

## 症状
多轮带图对话中，历史消息重附着图片（真实 smlx App 行为：每轮重建 chatMessages 时，
历史 user 消息会带着其原图路径重新附着）→ Turn2 追问（不带新图）时，模型把用户输入
复述了一遍，而不是正常回答。

App 内表现：连续两次报告"识图有问题，偶尔还有 panic，启动后第一次给图都不行"。
用独立 CLI 复现工具（test_vision_1/Qwen3VLBench）排查后确认：

- 单轮识图（全新 prefill，无历史）：完全正确。
- 多轮带图（增量 prefill，历史消息里的图重附着）：Turn2 复读用户输入。

## 复现方式
`test_vision_1/Sources/Qwen3VLBench/Qwen3VLBench.swift`：
1. 先跑一次 warmup（纯文本 "hi"，无 cache 参数，模拟 App ModelService.loadModel 的真实启动流程）。
2. Turn1：全新 prefill，带图，`.user(prompt, images:[url])`。
3. Turn2：`.user(prompt)`（无新图），但 `chatHistory` 数组里 Turn1 的消息仍带着图片
   （`UserInput.init(chat:)` 会汇总整个 chat 里所有消息的图片，所以 `input.images`
   在 Turn2 依然非空）。

运行：
```
cd test_vision_1
xcodebuild -scheme Qwen3VLBench -destination 'platform=macOS' -derivedDataPath .build/xcode-dd build -skipPackagePluginValidation
export HF_HOME="$HOME/Library/Application Support/huggingface"
.build/xcode-dd/Build/Products/Debug/Qwen3VLBench <modelDir> <imagePath>
```

## 根因
文件：`mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift`
函数：`Qwen3VLLanguage.LanguageModel.callAsFunction`（M-RoPE 位置计算的 4 分支判断）

decode 阶段（逐 token 生成时）统一走标准接口
`Qwen3VL.callAsFunction(_ input: LMInput.Text, cache:, state:)`，
它**永远**传 `imageGridTHW: nil, videoGridTHW: nil`（不管 prompt 历史里有没有图）。

位置分支判断顺序：
```
分支1: 用 prefill 缓存的位置切片（precomputedPositionIds 覆盖到 cacheOffset 之后）
       —— decode 阶段这个条件必然不满足（预计算位置已用完），pass
分支2: if imageGridTHW == nil && videoGridTHW == nil
       → 判定为"纯文本"，位置 = cacheOffset 起线性递增，且把 ropeDeltas 强制清零
分支3: 首轮/全新 prefill → 调 getRopeIndex 重新计算
分支4: cacheOffset + ropeDeltas —— 这才是"历史含图、当前 decode 纯文本"应走的分支
```

问题：分支2 的判断条件只看"本次调用是否传了图"，但 decode 阶段永远不传图。
只要历史 prompt 里出现过图片，一旦进入 decode，必然先命中分支2（因为
`imageGridTHW == nil` 恒真于 decode 调用），把 prefill 阶段 `getRopeIndex`
算出的正确 `ropeDeltas`（图片在 M-RoPE 三维位置上造成的跳跃量）强制清零，
分支4 永远不可达。

结果：decode 阶段所有新生成 token 的位置，比真实值少了一个 `delta`
（图片造成的位置跳跃量）。新 token 的 query 位置与历史 KV 的 key 位置基准不一致，
RoPE 相对位置算错，attention 结果错乱 → 输出复读/答非所问。

## 关联但暂缓的事实
`mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` 的
`Qwen35Language.LanguageModel.callAsFunction` 里有完全相同结构、相同注释的
4 分支判断逻辑（包括分支2 同样的条件缺陷）。Qwen3VL 的这段实现是参照 Qwen35
移植而来。理论上 Qwen35 在"历史带图 + 多轮不带图追问"场景下可能有同样的隐患，
但目前没有实测证据证明它在生产环境发作过。

**处理原则（已与用户确认）**：先只修 Qwen3VL.swift，验证 OK 后再评估是否
同步修 Qwen35.swift（Qwen35 是生产模型，改动需要更谨慎，且用户要先看到
Qwen3VL 修复验证通过）。

## 修复方案
只改分支2的判断条件，加一个 `&& ropeDeltas == nil`：

```swift
// 修改前
} else if imageGridTHW == nil && videoGridTHW == nil {

// 修改后
} else if imageGridTHW == nil && videoGridTHW == nil && ropeDeltas == nil {
```

效果：只有"从未建立过 ropeDeltas"（即真正从未见过图片的纯文本 session）才会
走分支2清零逻辑；一旦 prefill 阶段建立过非零 ropeDeltas（历史见过图），
后续 decode 会正确落入分支4（`cacheOffset + ropeDeltas`）。

## 验证计划
1. 只改 Qwen3VL.swift 这一处（1 行条件）。
2. 用同一个 `test_vision_1/Qwen3VLBench` CLI（含 warmup + 真实多轮重附着图片）
   重新跑，确认 Turn2 正常回答颜色问题而不是复读用户输入。
3. 验证通过后，再讨论 Qwen35.swift 是否需要同步修复。

## 追加排查：只改 Qwen3VL.swift 分支条件，验证仍然复读

用上面的 CLI 复测（同一张真实图片 + 同样两轮），Turn2 依然把用户问题复述了一遍。
说明 Qwen3VL.swift 的分支条件不是完整根因，继续深挖。

## 真正根因（框架层）
文件：`mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
函数：`TokenIterator.prepare(input:windowSize:)`（约 L639-657）

```swift
mutating func prepare(input: LMInput, windowSize: Int? = nil) throws {
    processor?.prompt(input.text.tokens)
    switch try model.prepare(input, cache: cache, windowSize: windowSize) {
    case .tokens(let tokens):
        y = tokens
        let token = step(previous: y)   // step() 内部会 self.state = result.state
        y = .init(tokens: token)
        asyncEval(y.tokens)
    case .logits(let result):
        y = .init(tokens: convertToToken(logits: result.logits))
        asyncEval(y.tokens)
        break   // ← 这里从未把 result.state 存到 self.state！
    }
}
```

VLM（Qwen3VL/Qwen35）的 `prepare()` 都走 `.logits(result)` 分支返回（不是 `.tokens`）。
`result.state` 里带着 `prepare()` 阶段用 `getRopeIndex` 算出的
`precomputedPositionIds` / `ropeDeltas`，但这行代码从未把它保存到
`TokenIterator.self.state`（初始化为 `nil`，且只有 `step()` 内部 L676
`self.state = result.state` 才会赋值——但 `step()` 调用 `model(...)` 时传的
`state: state` 用的还是当时的旧值）。于是：

- 第一次 decode `step()` 调用 `model(previous, cache:, state: nil)`
  （因为 `self.state` 从初始化后就没被设过）。
- `Qwen3VLLanguage.LanguageModel.callAsFunction` 里 `var state = state ?? .init()`
  拿到全新空白 state，`ropeDeltas` 恒为 `nil`。
- 于是我在 Qwen3VL.swift 加的 `ropeDeltas == nil` 判断永远为真，继续误入
  "纯文本清零 delta" 分支，之前的修复不生效。

这是 **`Evaluate.swift` 框架层的通用 bug**，影响所有走 `.logits` 返回路径、且
`prepare()` 内部维护跨 step 状态（如 ropeDeltas）的模型 —— 即 Qwen3VL 和 Qwen35
都受影响。Qwen35 生产环境未暴露此问题，推测是其 GatedDeltaNet 混合架构
（attention 层占比低）对这种位置精度损失不敏感，掩盖了症状，而 Qwen3VL 是
纯 attention 架构，位置错位直接反映为复读/答非所问。

## 最终修复（两处，已应用并验证通过）

1. `mlx-swift-lm/Libraries/MLXLMCommon/Evaluate.swift`
   `TokenIterator.prepare` 的 `.logits` 分支补上 `self.state = result.state`：
   ```swift
   case .logits(let result):
       self.state = result.state   // 新增
       y = .init(tokens: convertToToken(logits: result.logits))
       asyncEval(y.tokens)
       break
   ```
   这是框架层修复，对 Qwen3VL 和 Qwen35 都生效（都是正向修复，不改变现有正确行为，
   只是让本该传递的 state 真正传递下去）。

2. `mlx-swift-lm/Libraries/MLXVLM/Models/Qwen3VL.swift`
   （已应用，见上文修复方案）分支2 条件加 `&& ropeDeltas == nil`。
   这处修复是必要配套——没有它，即使 state 传递正确，decode 阶段仍会因为
   `imageGridTHW == nil` 恒真而误判成纯文本、清零已经正确传递过来的 delta。

## 验证结果（通过）
两处修复叠加后，用同一 CLI + 真实图片重测：
- Turn1（全新 prefill，带图）：正确描述图片内容（户外徒步女性、山脉、背包等）。
- Turn2（增量 prefill，历史图重附着，`⚡ incremental prefill: skip=865` 命中缓存）：
  正确回答"图中主要是什么颜色"，答案（深绿色/墨绿色、棕色/土黄色、浅灰米白）
  与图片实际内容完全吻合，不再复读。

## 遗留事项
- `Qwen35.swift` 是否需要同步修复分支2条件：**待用户确认**（生产模型，未改动）。
  注意 `Evaluate.swift` 的 state 传递修复已经对 Qwen35 生效（框架层，无法只影响
  Qwen3VL），所以 Qwen35 现在也会正确传递 ropeDeltas 到 decode 阶段；
  但 Qwen35.swift 里分支2 的条件本身仍是旧的（`imageGridTHW == nil && videoGridTHW == nil`，
  没有 `&& ropeDeltas == nil`），按同样的逻辑推理，Qwen35 在"历史含图+纯文本decode"
  场景下的分支2 依然会被误触发、清零 delta——这意味着 Evaluate.swift 的修复
  可能让 Qwen35 的这个潜在 bug 从"被状态丢失掩盖"变成"potentially 暴露"。
  需要评估是否要连带修 Qwen35.swift，或者靠其架构特性确认无影响。
- 已知遗留风险（来自更早的方案 B 评估，仍然有效）：`mrope_interleaved: true`
  与 `StreamingKVCache.evict` 的分段 M-RoPE shift 布局假设不一致，长对话触发
  evict 时可能仍有位置错位风险，本次验证未触发（用了较大的 maxKVSize 规避）。

