//
//  MTPHead.swift
//  mlx-swift-lm
//
//  MTP (Multi-Token Prediction) head implementation for Qwen3.5/3.6.
//  The MTP head consists of:
//  1. A projection layer that combines previous hidden states with token embeddings
//  2. A single transformer decoder layer (shared architecture with backbone)
//  3. A shared lm_head (from the main model) to predict the next-next token
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

// MARK: - MTP Block

/// A single MTP decoder block for Qwen3.5/3.6.
/// This is essentially one transformer layer that takes:
/// - The hidden states from the backbone (after the last layer)
/// - The embedding of the token predicted by the backbone
/// And produces hidden states that are projected to vocabulary logits via the shared lm_head.
public final class Qwen35MTPBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen35Attention
    @ModuleInfo(key: "mlp") var mlp: Qwen3NextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(_ args: Qwen35TextConfiguration) {
        _selfAttn.wrappedValue = Qwen35Attention(args)
        _mlp.wrappedValue = Qwen3NextMLP(
            dimensions: args.hiddenSize,
            hiddenDimensions: args.intermediateSize
        )
        _inputLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(
            dimensions: args.hiddenSize, eps: args.rmsNormEps)
        super.init()
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let r = selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        let h = x + r
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// Attention module for MTP block (standard multi-head attention without gating)
public final class Qwen35MTPAttention: Module {
    let attentionHeads: Int
    let kvHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPELayer

    init(_ args: Qwen35TextConfiguration) {
        let hd = args.headDim ?? (args.hiddenSize / args.attentionHeads)
        self.headDim = hd
        self.attentionHeads = args.attentionHeads
        self.kvHeads = args.kvHeads
        self.scale = pow(Float(hd), -0.5)

        _qProj.wrappedValue = Linear(
            args.hiddenSize, args.attentionHeads * hd, bias: args.attentionBias)
        _kProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * hd, bias: args.attentionBias)
        _vProj.wrappedValue = Linear(
            args.hiddenSize, args.kvHeads * hd, bias: args.attentionBias)
        _oProj.wrappedValue = Linear(
            args.attentionHeads * hd, args.hiddenSize, bias: args.attentionBias)

        _qNorm.wrappedValue = RMSNorm(dimensions: hd, eps: args.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: hd, eps: args.rmsNormEps)

        let ropeDims = Int(Float(hd) * args.partialRotaryFactor)
        self.rope = initializeRope(
            dims: max(1, ropeDims),
            base: args.ropeTheta,
            traditional: false,
            scalingConfig: args.ropeScaling,
            maxPositionEmbeddings: args.maxPositionEmbeddings
        )

        super.init()
    }

    func callAsFunction(
        _ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, attentionHeads, headDim)
        var keys = kProj(x).reshaped(B, L, kvHeads, headDim)
        var values = vProj(x).reshaped(B, L, kvHeads, headDim)

        queries = qNorm(queries).transposed(0, 2, 1, 3)
        keys = kNorm(keys).transposed(0, 2, 1, 3)
        values = values.transposed(0, 2, 1, 3)

        let offset = cache?.ropeOffset
        queries = applyRotaryPosition(rope, to: queries, offset: offset)
        keys = applyRotaryPosition(rope, to: keys, offset: offset)

        let output = attentionWithCacheUpdate(
            queries: queries,
            keys: keys,
            values: values,
            cache: cache,
            scale: scale,
            mask: mask
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, -1)

        return oProj(output)
    }
}

// MARK: - MTP Model (used as a drafter in speculative decoding)

/// The complete MTP model that wraps the backbone and MTP head.
/// It acts as a drafter for speculative decoding:
/// 1. Forward pass through backbone → get hidden states
/// 2. Combine hidden states with token embedding
/// 3. Forward through MTP block → get logits for next-next token
public class Qwen35MTPModel: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    /// LoRAModel conformance - MTP model doesn't use LoRA adapters
    public var loraLayers: [Module] { [] }

    /// The backbone model (shared, not owned - we use it read-only)
    let backbone: Qwen35TextModelInner
    let configuration: Qwen35TextConfiguration

    /// MTP-specific layers
    @ModuleInfo(key: "enorm") var eNorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hNorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "block") var block: Qwen35MTPBlock

    /// Shared lm_head from the main model
    var lmHead: Linear?
    var tieWordEmbeddings: Bool

    public init(_ args: Qwen35TextConfiguration, backbone: Qwen35TextModelInner) {
        self.backbone = backbone
        self.configuration = args
        self.vocabularySize = args.vocabularySize
        self.kvHeads = [args.kvHeads]
        self.tieWordEmbeddings = args.tieWordEmbeddings

        _eNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _hNorm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
        _ehProj.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
        _block.wrappedValue = Qwen35MTPBlock(args)
    }

    /// Forward pass: takes token IDs, returns logits for the NEXT token after what backbone predicts
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Get embedding of input tokens
        let tokenEmb = backbone.embedTokens(inputs)

        // The MTP head combines:
        // 1. eNorm(token_embedding) - embedding of the backbone's predicted token
        // 2. hNorm(hidden_states) - hidden states from backbone's last layer
        //
        // In the context of speculative decoding, the backbone has already run,
        // and we have cached hidden states. But since we're using this as a
        // standalone drafter model, we pass through the backbone first.

        // Run through backbone to get hidden states
        let hiddenStates = backbone(inputs, cache: nil)

        // Combine embedding and hidden states
        let eNormed = eNorm(tokenEmb)
        let hNormed = hNorm(hiddenStates)
        let combined = MLX.concatenated([eNormed, hNormed], axis: -1)
        var x = ehProj(combined)

        // Run through MTP block
        let mask = createAttentionMask(h: x, cache: cache?.first)
        x = block(x, mask: mask, cache: cache?.first)

        // Project to vocabulary
        if let lmHead {
            return lmHead(x)
        } else {
            return backbone.embedTokens.asLinear(x)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}

// MARK: - Lightweight MTP Drafter

/// A lightweight MTP drafter that doesn't re-run the backbone.
/// Instead, it receives cached hidden states from the backbone
/// and uses them directly with the MTP head.
///
/// This is the efficient version used during speculative decoding:
/// - The backbone runs once and produces hidden_states + lm_head logits
/// - The MTP drafter takes those hidden_states + the predicted token's embedding
///   to predict the NEXT token without re-running the full backbone
public class Qwen35MTPDrafter: Module, LLMModel, KVCacheDimensionProvider {
    public let vocabularySize: Int
    public let kvHeads: [Int]

    /// LoRAModel conformance - MTP drafter doesn't use LoRA adapters
    public var loraLayers: [Module] { [] }

    /// Reference to backbone's embedding layer
    let embedTokens: Embedding

    /// MTP-specific layers
    let eNorm: RMSNorm
    let hNorm: RMSNorm
    let ehProj: Linear
    let block: Qwen35MTPBlock

    /// Shared lm_head
    var lmHead: Linear?
    let tieWordEmbeddings: Bool

    /// Cached hidden states from the backbone (set externally before each draft round)
    public var cachedHiddenStates: MLXArray?

    public init(
        embedTokens: Embedding,
        eNorm: RMSNorm,
        hNorm: RMSNorm,
        ehProj: Linear,
        block: Qwen35MTPBlock,
        lmHead: Linear?,
        tieWordEmbeddings: Bool,
        configuration: Qwen35TextConfiguration
    ) {
        self.embedTokens = embedTokens
        self.eNorm = eNorm
        self.hNorm = hNorm
        self.ehProj = ehProj
        self.block = block
        self.lmHead = lmHead
        self.tieWordEmbeddings = tieWordEmbeddings
        self.vocabularySize = configuration.vocabularySize
        self.kvHeads = [configuration.kvHeads]
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        // Get token embedding
        let tokenEmb = embedTokens(inputs)
        let eNormed = eNorm(tokenEmb)

        // Use cached hidden states if available, otherwise just use embedding
        let x: MLXArray
        if let hiddenStates = cachedHiddenStates {
            let hNormed = hNorm(hiddenStates)
            let combined = MLX.concatenated([eNormed, hNormed], axis: -1)
            x = ehProj(combined)
        } else {
            // Fallback: use doubled embedding (shouldn't happen in normal flow)
            let combined = MLX.concatenated([eNormed, eNormed], axis: -1)
            x = ehProj(combined)
        }

        // Run through MTP block
        let mask = createAttentionMask(h: x, cache: cache?.first)
        let out = block(x, mask: mask, cache: cache?.first)

        // Project to vocabulary
        if let lmHead {
            return lmHead(out)
        } else {
            return embedTokens.asLinear(out)
        }
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        [KVCacheSimple()]
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        weights
    }
}
