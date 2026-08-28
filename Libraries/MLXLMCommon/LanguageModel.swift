// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Abstract form of a model that processes language.
public protocol BaseLanguageModel: Module {
    /// Optionally preprocess the weights and modify / remove values as needed.
    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray]

    /// Optionally preprocess the weights with access to safetensor metadata.
    ///
    /// The default implementation forwards to ``sanitize(weights:)``.
    /// Models can override this to inspect metadata (e.g. check `metadata["format"] == "mlx"`)
    /// and skip or customize sanitization accordingly.
    func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String: MLXArray]
}

extension BaseLanguageModel {
    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        sanitize(weights: weights)
    }
}

/// Time/Height/Width struct to represent information about input images.
public struct THW: Sendable {

    public let t: Int
    public let h: Int
    public let w: Int

    public init(_ t: Int, _ h: Int, _ w: Int) {
        self.t = t
        self.h = h
        self.w = w
    }

    public var values: (Int, Int, Int) {
        (t, h, w)
    }

    public var product: Int { t * h * w }
}

/// Representation of ``LanguageModel`` input.
///
/// This can contain text (tokens), prepared images (`MLXArray`), or other media as
/// needed. ``LMInput`` is produced by ``UserInputProcessor`` in response
/// to ``UserInput``.
///
/// The ``ModelContext`` holds the ``UserInputProcessor`` associated with a
/// ``LanguageModel``.
public struct LMInput {
    public let text: Text
    public let image: ProcessedImage?
    public let video: ProcessedVideo?

    /// Representation of tokenized input text.
    public struct Text {

        /// input token array
        public let tokens: MLXArray

        /// optional mask array
        public let mask: MLXArray?

        public init(tokens: MLXArray, mask: MLXArray? = nil) {
            self.tokens = tokens
            self.mask = mask
        }

        public subscript(
            indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask?[indices, stream: stream])
        }

        public subscript(
            text indices: MLXArrayIndex..., stream stream: StreamOrDevice = .default
        ) -> Text {
            Text(tokens: tokens[indices, stream: stream], mask: mask)
        }
    }

    /// Representation of prepared input image(s).
    public struct ProcessedImage {

        /// Concatenated pixels from one or more images
        public let pixels: MLXArray
        /// Time, height, and width of the images
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    /// Representation of prepared input video(s).
    /// For now, this is virtually identical to ProcessedImage.
    public struct ProcessedVideo {

        public let pixels: MLXArray
        public let frames: [THW]?

        public init(
            pixels: MLXArray, frames: [THW]? = nil
        ) {
            self.pixels = pixels
            self.frames = frames
        }
    }

    public init(tokens: MLXArray, mask: MLXArray? = nil) {
        self.init(text: .init(tokens: tokens, mask: mask))
    }

    public init(
        text: LMInput.Text, image: LMInput.ProcessedImage? = nil,
        video: LMInput.ProcessedVideo? = nil
    ) {
        self.text = text
        self.image = image
        self.video = video
    }
}

/// ``LanguageModel`` step output. This is consumed internally
/// by the ``TokenIterator``.
public struct LMOutput {

    /// logits (one hot vector of probabilities for tokens)
    public let logits: MLXArray

    /// optional ``State`` to carry forward into the next step
    public let state: State?

    /// typed key for use in ``State``
    public struct Key<T>: Identifiable, Sendable {
        public let id: String

        public init(_ id: String) {
            self.id = id
        }
    }

    /// Dictionary of typed ``Key`` to carry state between steps.
    public struct State {
        private var contents: [String: Any]

        public init() {
            self.contents = [:]
        }

        public subscript<T>(_ key: Key<T>) -> T? {
            get {
                contents[key.id] as? T
            }
            set {
                contents[key.id] = newValue
            }
        }
    }

    public init(logits: MLXArray, state: LMOutput.State? = nil) {
        self.logits = logits
        self.state = state
    }
}

/// The result of the call to ``LanguageModel/prepare(_:cache:windowSize:)``
public enum PrepareResult {
    /// tokens to process by the ``TokenIterator``
    case tokens(LMInput.Text)

    /// logits representing the next token
    case logits(LMOutput)
}

/// Interface for all Language Models (e.g. LLM, VLM).
///
/// The language model is typically called by the ``TokenIterator`` and it:
///
/// - consumes the ``LMInput``
/// - calls ``prepare(_:cache:windowSize:)`` to initialize the KVCache and consume the prompt
/// - calls ``callAsFunction(_:cache:state:)-9kuvf`` for each token, producing an ``LMOutput``
/// - the ``TokenIterator`` accumulates this information into a ``GenerateResult``
public protocol LanguageModel: BaseLanguageModel {

    /// Prepare the cache state and consume the ``LMInput``.
    ///
    /// This can return:
    /// - ``PrepareResult/tokens(_:)`` if the caller should evaluate the (remaining) tokens normally
    /// - ``PrepareResult/logits(_:)`` to produce the next token from the prompt
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws -> PrepareResult

    /// Primary entry point to produce a step (single token) from the model
    func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput

    /// Models may implement this simplified interface if they do not produce any ``LMOutput/State``
    func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray

    /// create a new array of ``KVCache``: automatic implementation if self
    /// implements ``KVCacheDimensionProvider``
    func newCache(parameters: GenerateParameters?) -> [KVCache]
}

extension LanguageModel {
    public func callAsFunction(_ input: LMInput.Text, cache: [KVCache]?, state: LMOutput.State?)
        -> LMOutput
    {
        let logits = callAsFunction(input.tokens, cache: cache)
        return .init(logits: logits)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        fatalError("callAsFunction(inputs:cache:) not implemented for \(Self.self)")
    }
}

/// Protocol for language models that support Multi-Token Prediction (MTP) speculative decoding.
///
/// Models conforming to this protocol can use the built-in ``MTPSpeculativeTokenIterator``
/// for speculative generation without requiring a separate draft model. The MTP head
/// reuses the backbone's hidden states to predict the next draft token efficiently.
///
/// Currently supported by: Qwen3.5, Qwen3.6 (models with `mtpNumHiddenLayers > 0`).
public protocol MTPCapableModel: LanguageModel {
    /// Whether this model instance has MTP weights loaded and available.
    var hasMTP: Bool { get }

    /// Forward pass that returns both logits and hidden states.
    ///
    /// The hidden states are needed by the MTP head to produce draft tokens
    /// without re-running the full backbone.
    ///
    /// - Parameters:
    ///   - inputs: token IDs tensor of shape `(1, N)`
    ///   - cache: backbone KV cache
    ///   - nConfirmed: number of confirmed tokens at the start of the sequence.
    ///     When > 0 and < input length, linear attention layers should snapshot their state
    ///     after processing the confirmed tokens for zero-cost rollback on draft rejection.
    /// - Returns: A tuple of `(logits, hiddenStates)` where hiddenStates has shape `(1, N, H)`
    func forwardWithHiddenStates(
        _ inputs: MLXArray, cache: [KVCache]?, nConfirmed: Int
    ) -> (logits: MLXArray, hiddenStates: MLXArray)

    /// MTP head forward pass: given backbone hidden states and token IDs,
    /// produce logits for the next draft token.
    ///
    /// - Parameters:
    ///   - tokenIds: token IDs tensor of shape `(1, N)` (the tokens whose embeddings are combined with hidden states)
    ///   - hiddenStates: hidden states from the backbone of shape `(1, N, H)`
    ///   - cache: MTP head KV cache (separate from backbone cache, typically 1 layer)
    ///   - positionOffset: additional offset added to the computed position id. Used when
    ///     iteratively generating multiple draft tokens (D2+) from the same backbone hidden
    ///     without advancing the backbone offset. Defaults to 0 (D1 behavior, unchanged).
    /// - Returns: logits tensor, or `nil` if MTP is not available
    func mtpForward(
        _ tokenIds: MLXArray, hiddenStates: MLXArray, cache: [KVCache]?, positionOffset: Int
    ) -> MLXArray?

    /// Optimized MTP forward with cache commit: commits the accepted draft position to the
    /// MTP cache (KV update only, skipping norm + lm_head) then generates logits for the
    /// current position in a single call.
    ///
    /// This avoids wasted lm_head computation on the commit position whose logits are discarded.
    ///
    /// - Parameters:
    ///   - tokenIds: token IDs for current position, shape `(1, 1)`
    ///   - hiddenStates: hidden states for current position, shape `(1, 1, H)`
    ///   - commitToken: token ID for the accepted draft position to commit
    ///   - commitHidden: hidden states for the commit position, shape `(1, 1, H)`
    ///   - cache: MTP head KV cache
    /// - Returns: logits tensor for current position, or `nil` if MTP is not available
    func mtpForwardWithCommit(
        _ tokenIds: MLXArray, hiddenStates: MLXArray,
        commitToken: MLXArray, commitHidden: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray?

    /// Create a new MTP cache (typically just `[KVCacheSimple()]` since MTP has 1 transformer layer).
    func newMTPCache() -> [KVCache]

    /// Hidden states from the last forward pass (e.g., from prepare).
    ///
    /// Used by MTP speculative decoding to get the backbone hidden state
    /// without re-running the backbone. The MTP head needs the hidden state
    /// **before** the next token is processed (position t-1), not after.
    var lastForwardHiddenStates: MLXArray? { get }

    /// 方案C（变换链）部分接受恢复：对所有 GatedDeltaNet 层，从 confirmed 后的
    /// base state 出发，用 `draftOperators[draftIndex]` 折叠出“走完该 draft 后”的
    /// SSM 状态并写回 cache，免去搬运完整状态快照。返回是否折叠成功。
    ///
    /// ⚠️ 必须声明为协议要求（而非仅扩展默认实现），否则通过 `MTPCapableModel`
    /// 协议类型调用时会静态派发到下方默认实现 `{ false }`，具体模型的真实折叠被绕过。
    @discardableResult
    func foldPartialAccept(cache: [KVCache], draftIndex: Int) -> Bool
}

/// Default implementations for MTPCapableModel.
extension MTPCapableModel {
    public func newMTPCache() -> [KVCache] {
        [KVCacheSimple()]
    }

    /// Convenience overload preserving the original signature (D1 callers).
    /// Forwards with `positionOffset: 0`.
    public func mtpForward(
        _ tokenIds: MLXArray, hiddenStates: MLXArray, cache: [KVCache]?
    ) -> MLXArray? {
        mtpForward(tokenIds, hiddenStates: hiddenStates, cache: cache, positionOffset: 0)
    }

    /// Default: no hidden states available.
    public var lastForwardHiddenStates: MLXArray? { nil }

    /// Default: model has no GatedDeltaNet layers to fold; no-op.
    @discardableResult
    public func foldPartialAccept(cache: [KVCache], draftIndex: Int) -> Bool { false }

    /// Default fallback: concatenate commit + current and call standard mtpForward.
    /// Models can override this to provide an optimized implementation that skips
    /// norm + lm_head computation on the commit position.
    public func mtpForwardWithCommit(
        _ tokenIds: MLXArray, hiddenStates: MLXArray,
        commitToken: MLXArray, commitHidden: MLXArray,
        cache: [KVCache]?
    ) -> MLXArray? {
        // Fallback: concatenate and call standard mtpForward, then take last position
        let combinedHidden = MLX.concatenated([commitHidden, hiddenStates], axis: 1)
        let combinedTokens = MLX.concatenated(
            [commitToken.reshaped(1, 1), tokenIds.reshaped(1, 1)], axis: 1)
        guard let logits = mtpForward(combinedTokens, hiddenStates: combinedHidden, cache: cache)
        else { return nil }
        // Return only the last position's logits, keeping 3D shape [1, 1, vocab]
        // (-1 removes the dimension which causes downstream crash; use range slice to keep it)
        let lastIdx = logits.dim(1) - 1
        return logits[0..., lastIdx ..< (lastIdx + 1), 0...]
    }
}

/// Optional protocol that can be implemented by ``LanguageModel`` and will
/// provide an automatic implementation of ``LanguageModel/newCache(parameters:)``
public protocol KVCacheDimensionProvider {
    var kvHeads: [Int] { get }
}

extension LanguageModel where Self: KVCacheDimensionProvider {
    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        // Create one cache per layer (kvHeads.count = number of layers)
        // The number of heads per layer (kvHeads[i]) is not used for cache creation
        let numLayers = kvHeads.count

        // Follow Python logic: use RotatingKVCache if maxKVSize is provided
        if let maxKVSize = parameters?.maxKVSize {
            return (0 ..< numLayers).map { _ in
                RotatingKVCache(maxSize: maxKVSize, keep: 4)
            }
        } else {
            return (0 ..< numLayers).map { _ in KVCacheSimple() }
        }
    }
}
