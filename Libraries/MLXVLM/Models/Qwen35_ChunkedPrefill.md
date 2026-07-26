# VLM Chunked Prefill 实现详解

## 1. 背景

### 1.1 为什么需要 Chunked Prefill

VLM 模型的 prompt 包含图片 visual tokens（通常 600-2000 个），加上系统提示词、用户文本，一个带图对话的 prefill 序列很容易达到 **800-2000 tokens**。

一次性 prefill 全量序列会带来两个问题：

1. **显存峰值过高**：attention 计算中的 Q·K^T 矩阵是 `[seq_len × seq_len]`，800 tokens 需要 800²=640K 个元素，每个 FP16 需要临时 ~1.2MB — 看起来不多，但加上 multi-head、多层叠加的中间结果，瞬时分片可达数百 MB。
2. **无进度感知**：prefill 是同步阻塞的，长序列时用户看到 app "卡住" 数秒没有反馈。

分块 prefill（chunked prefill）将长序列切分为小块，逐块前向，每块间 `asyncEval` 释放显存，同时回调进度给 UI。

### 1.2 纯文本 LLM 已有分块支持

smlx 在 `InferenceEngine.executeInference()` 中对纯文本路径已实现分块 prefill：

```swift
// 仅当后缀 > 256 tokens 时才分块
if suffixSize > EngineConfig.prefillProgressThreshold {
    let chunkSize = EngineConfig.prefillChunkSize  // 128
    while y.tokens.size > chunkSize {
        let chunk = y[.newAxis, ..<chunkSize]
        let output = context.model(chunk, cache: cache, state: lmState)
        lmState = output.state
        asyncEval(cache)
        onPrefillProgress?(processed, total)
        y = y[chunkSize...]
    }
}
```

纯文本分块很直接：切 token 序列 → 逐块调 `model(chunk)` → 通过 `lmState` 传递位置信息。

### 1.3 VLM 的特殊性

VLM 比纯文本复杂得多，原因有三：

1. **输入是 embedding 而非 token**：图片经 vision encoder 编码为 visual features，通过 `mergeInputIdsWithImageFeatures` 替换文本序列中的 `<image>` 占位符。分块时需要同时切 token IDs（用于位置计算）和 embeddings（实际输入）。
2. **M-RoPE 3D 位置编码**：文本 token 用 1D 位置 `[pos, pos, pos]`，图片 visual token 用 3D 位置 `[t_pos, h_pos, w_pos]`（时间/高度/宽度）。`getRopeIndex` 需要**完整序列**的 `inputIds`、`imageGridTHW` 和 `attentionMask` 才能正确计算。
3. **languageModel 内部调用 getRopeIndex**：`LanguageModel.callAsFunction()` 内部会根据 `pixelValues`/`imageGridTHW` 调用 `getRopeIndex`。传入 chunk 时它拿不到完整图像上下文，会导致计算错误或 crash。

## 2. 总体方案

**分两层处理**：

```
┌─ Qwen35.prepare() ──────────────────────────────────────────────────────┐
│                                                                         │
│  1. Vision encoder:       图片 → visual features   (不分块)               │
│  2. Merge:                visual features + text embeddings → 完整序列   │
│  3. Pre-compute position: getRopeIndex(完整 inputIds, 完整 imageGridTHW) │
│                           → fullPositionIds[3, 1, totalLength]           │
│  4. Chunk loop:                                                         │
│     for offset in stride(totalLength, prefillStepSize):                 │
│       chunkEmbeds  = mergedEmbeds[0..., offset:chunkEnd, :]             │
│       chunkPosIds  = fullPositionIds[0..., 0..., offset:chunkEnd]       │
│       languageModel(chunkIds, inputsEmbeds: chunkEmbeds,                │
│                     positionIds: chunkPosIds, ...)                       │
│       prefillProgressCallback?(chunkEnd, totalLength)                   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心思想**：把「需要完整上下文」的 `getRopeIndex` 提到 `prepare()` 最外层一次性算完，chunk 循环内只做纯前向（position IDs 已就绪），不再依赖 languageModel 内部的位置计算。

## 3. 实现细节

### 3.1 position IDs 预计算（qwen35.prepare 第 1342-1360 行）

```swift
// 一次性算好完整序列的 3D position IDs
if pixelValues != nil {
    let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
        inputIds: inputIds,          // 完整 [1, totalLength]
        imageGridTHW: imageFrames,   // 完整图像网格
        videoGridTHW: videoFrames,
        spatialMergeSize: config.visionConfiguration.spatialMergeSize,
        imageTokenId: config.imageTokenId,
        videoTokenId: config.videoTokenId,
        visionStartTokenId: config.visionStartTokenId,
        attentionMask: input.text.mask
    )
    precomputedState = .init()
    precomputedState?[precomputedPositionIdsKey] = computed  // [3, 1, totalLength]
    precomputedState?[ropeDeltasKey] = deltas
    fullPositionIds = computed
}
```

这一步只在有图片（`pixelValues != nil`）时执行。纯文本时 `fullPositionIds = nil`，由 languageModel 内部走 text-only 快速路径。

### 3.2 chunk 循环（第 1362-1398 行）

```swift
var offset = 0
while offset < totalLength {
    let chunkEnd = min(offset + prefillStepSize, totalLength)
    let chunkIds = inputIds[0..., offset ..< chunkEnd]
    let chunkEmbeds = inputEmbeddings?[0..., offset ..< chunkEnd, 0...]

    // 本 chunk 的 position IDs 直接切片，无需重新计算
    let chunkPosIds: MLXArray?
    if let fullPositionIds {
        chunkPosIds = fullPositionIds[0..., 0..., offset ..< chunkEnd]
    } else {
        chunkPosIds = nil
    }

    let output = languageModel(
        chunkIds,
        inputsEmbeds: chunkEmbeds,
        cache: typedCache,
        state: precomputedState,   // 含预计算的位置缓存
        mask: nil,                 // 不传 mask，getRopeIndex 已在外部完成
        positionIds: chunkPosIds,  // 直接喂切片位置
        pixelValues: nil,          // 不触内部 getRopeIndex
        imageGridTHW: nil,
        videoGridTHW: nil
    )
    state = output.state
    lastOutput = output
    asyncEval(cache)
    offset = chunkEnd
    prefillProgressCallback?(chunkEnd, totalLength)
}
eval(cache)
```

关键点：
- **`state: precomputedState`** 而非 `nil`；传入的 state 已含 `precomputedPositionIds` 和 `ropeDeltas`，languageModel 内部发现 `positionIds != nil` 后直接使用，不走 `getRopeIndex`
- **`mask: nil`** 避免 mask 长度与 chunk 长度不一致导致的 `mask.dim(-1) != inputs.dim(-1)` 检查失败
- **`pixelValues: nil`** 防止 languageModel 内部 `state[precomputedPositionIdsKey] = nil` 重置缓存
- **`asyncEval(cache)`** 每块后释放中间计算图，控制显存峰值

### 3.3 进度回调机制

| 层 | 位置 | 作用 |
|---|---|---|
| `Qwen35.prefillProgressCallback` | mlx-swift-lm | 属性，prepare 在每块完成时调用 |
| `InferenceEngine.executeVisionInference` | smlx | 调用前 `vlmModel?.prefillProgressCallback = onPrefillProgress` |
| `InferenceEngine.submit` | smlx | 接 UI 回调 `onPrefillProgress` |
| `ChatViewController` | smlx UI | 设 `statusLabel = "Processing prompt… X%"` |

数据流：

```
ChatViewController.onPrefillProgress
  └→ submit(..., onPrefillProgress:)
       └→ executeVisionInference(..., onPrefillProgress:)
            └→ vlmModel.prefillProgressCallback = onPrefillProgress
                 └→ Qwen35.prepare()
                      └→ while chunk:
                           prefillProgressCallback?(chunkEnd, totalLength)
```

chunk 大小由 `GenerateParameters.prefillStepSize` 控制，默认 512。smlx 覆盖为 `EngineConfig.prefillChunkSize = 128`（与纯文本路径一致）。

仅当 `totalLength > prefillStepSize` 时进入分块路径；短 prompt 走一次性前向，不触发进度上报。

## 4. 踩过的坑

### 4.1 `inputIds.dim(0)` 不是序列长度（严重）

`input.text.tokens` 形状是 `[1, totalLength]`（batch=1，seq=totalLength）。

- `dim(0)` = **batch size** = 1
- `dim(1)` = **序列长度** = 实际 token 数

错用 `dim(0)` 导致 `totalLength = 1`，永远不满足 `> prefillStepSize`，分块不触发，进度不显示。

### 4.2 getRopeIndex 不能对 chunk 调用（会导致 crash）

`getRopeIndex` 依赖完整序列的 `inputIds` 和 `attentionMask` 来：
- 定位 `visionStartTokenId` 标记
- 统计图像 token 数量
- 计算 mrope position deltas

传入 chunk（512 tokens）时，chunk 内可能不包含 `visionStartTokenId` 或图像占位 token，导致 `getRopeIndex` 内部逻辑异常或 crash。

**修复**：在 `prepare()` 外层用完整 inputIds 一次性调用 `getRopeIndex`，结果切成 chunk 再传给 languageModel。languageModel 检测到 `positionIds != nil` 后跳过内部的位置计算。

### 4.3 mask 长度与 chunk 不匹配

`input.text.mask` 的长度是完整序列长度（如 813），但 chunk 只有 512。`LanguageModel.callAsFunction` 中有：

```swift
if let mask, mask.dim(-1) != inputs.dim(-1) {
    ropeMask = nil
}
```

传入不匹配的 mask 会导致 `ropeMask = nil`，进而可能错误地进入 `getRopeIndex` 路径。**修复**：chunk 时不传 mask（`mask: nil`）。

### 4.4 pixelValues 触发 state 重置

`LanguageModel.callAsFunction` 中：

```swift
if pixelValues != nil {
    state[precomputedPositionIdsKey] = nil
    state[ropeDeltasKey] = nil
}
```

chunk 1 算好 `precomputedPositionIds` 存到 state，但如果在后续 chunk 中也传了 `pixelValues`，state 中的缓存会被清空。**修复**：chunk 时不传 `pixelValues`。

## 5. 图片与文本混合场景

chunk 不区分图片 token 和文本 token。两种 token 在语言模型中计算代价相近（都经过 64 层 transformer），按 token 数报进度是合理的。

```
完整序列 (813 tokens):
[text_0, ..., text_50, visionStart, img_0, img_1, ..., img_749, text_51, ..., text_60]
 └─ 提示词 ─┘           └──────── 图片 visual tokens (750) ────────┘  └─ 用户文本 ─┘

chunk 1 (0..<512):  前半 → 63%
chunk 2 (512..<640): 后半+文本 → 79%
...
chunk 7: → 100%
```

## 6. chunk prefill 后的生成阶段

chunk prefill 完成后，state 中含 `precomputedPositionIds` 和 `ropeDeltas`。

后续 MTP 投机解码的 `forwardWithHiddenStates` 中，languageModel 检测到 `precomputedPositionIds.dim(-1) > cacheOffset` 时，从缓存切片位置：

```swift
// LanguageModel.callAsFunction
if let precomputedPositionIds, precomputedPositionIds.dim(-1) > cacheOffset {
    let seqLength = inputs.dim(1)
    positionIds = precomputedPositionIds[0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
}
```

`> cacheOffset` 守卫确保只有 prefill 阶段使用缓存位置；生成阶段 prefill 走完（cacheOffset ≥ 缓存长度），走 else 分支用 `cacheOffset + ropeDeltas` 算纯文本位置。

## 7. 文件变更清单

| 文件 | 修改内容 |
|---|---|
| `mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` | `prepare()` 加 while 循环分块逻辑（~60 行）；加 `prefillProgressCallback` 属性；position IDs 预计算 |
| `mlx-swift-lm/Libraries/MLXVLM/Models/Qwen35.swift` (line 1013-1016) | position IDs 优先读缓存，加 `precomputedPositionIds.dim(-1) > cacheOffset` 守卫 |
| `smlx/smlx/InferenceEngine/InferenceEngine.swift` | `executeVisionInference` 加 `onPrefillProgress` 参数；设置 `prefillProgressCallback`；使用 `EngineConfig.prefillChunkSize` |
| `smlx/smlx/InferenceEngine/EngineConfig.swift` | `prefillChunkSize = 128`、`prefillProgressThreshold = 256`（已有，仅引用） |
| `smlx/smlx/Views/ChatViewController.swift` | `onPrefillProgress` → `statusLabel = "Processing prompt… X%"`（已有，仅确认兼容） |
