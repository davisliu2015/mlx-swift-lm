//
//  Qwen35.swift
//  mlx-swift-lm
//
//  Created by John Mai on 2026/2/25.
//
//  Port of https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen3_5
//

import Foundation
import MLX
import MLXLMCommon
import MLXNN

private enum Qwen35VLError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
}

private let precomputedPositionIdsKey = LMOutput.Key<MLXArray>(
    "qwen35.precomputedPositionIds")
private let ropeDeltasKey = LMOutput.Key<MLXArray>(
    "qwen35.ropeDeltas")

// MARK: - Configuration

public struct Qwen35Configuration: Codable, Sendable {

    public struct TextConfiguration: Codable, Sendable {
        public var modelType: String = ""
        public var hiddenSize: Int = 4096
        public var hiddenLayers: Int = 32
        public var intermediateSize: Int = 14_336
        public var attentionHeads: Int = 32
        public var kvHeads: Int = 8
        public var linearNumValueHeads: Int = 64
        public var linearNumKeyHeads: Int = 16
        public var linearKeyHeadDim: Int = 192
        public var linearValueHeadDim: Int = 128
        public var linearConvKernelDim: Int = 4
        public var rmsNormEps: Float = 1e-6
        public var vocabularySize: Int = 248_320
        public var ropeTheta: Float = 100_000.0
        public var partialRotaryFactor: Float = 0.25
        public var maxPositionEmbeddings: Int = 131_072
        public var tieWordEmbeddings: Bool = false
        public var attentionBias: Bool = false
        public var headDim: Int?
        public var ropeParameters: [String: StringOrNumber]?
        public var fullAttentionInterval: Int = 4

        // MoE fields
        public var numExperts: Int = 0
        public var numExpertsPerTok: Int = 0
        public var decoderSparseStep: Int = 1
        public var sharedExpertIntermediateSize: Int = 0
        public var moeIntermediateSize: Int = 0
        public var normTopkProb: Bool = true

        // MTP support
        public var mtpNumHiddenLayers: Int = 0

        enum CodingKeys: String, CodingKey {
            case modelType = "model_type"
            case hiddenSize = "hidden_size"
            case hiddenLayers = "num_hidden_layers"
            case intermediateSize = "intermediate_size"
            case attentionHeads = "num_attention_heads"
            case kvHeads = "num_key_value_heads"
            case linearNumValueHeads = "linear_num_value_heads"
            case linearNumKeyHeads = "linear_num_key_heads"
            case linearKeyHeadDim = "linear_key_head_dim"
            case linearValueHeadDim = "linear_value_head_dim"
            case linearConvKernelDim = "linear_conv_kernel_dim"
            case rmsNormEps = "rms_norm_eps"
            case vocabularySize = "vocab_size"
            case ropeTheta = "rope_theta"
            case partialRotaryFactor = "partial_rotary_factor"
            case maxPositionEmbeddings = "max_position_embeddings"
            case tieWordEmbeddings = "tie_word_embeddings"
            case attentionBias = "attention_bias"
            case headDim = "head_dim"
            case ropeParameters = "rope_parameters"
            case fullAttentionInterval = "full_attention_interval"
            case numExperts = "num_experts"
            case numExpertsPerTok = "num_experts_per_tok"
            case decoderSparseStep = "decoder_sparse_step"
            case sharedExpertIntermediateSize = "shared_expert_intermediate_size"
            case moeIntermediateSize = "moe_intermediate_size"
            case normTopkProb = "norm_topk_prob"
            case mtpNumHiddenLayers = "mtp_num_hidden_layers"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)

            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? ""
            self.hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
            self.hiddenLayers = try container.decodeIfPresent(Int.self, forKey: .hiddenLayers) ?? 32
            self.intermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 14_336
            self.attentionHeads =
                try container.decodeIfPresent(Int.self, forKey: .attentionHeads) ?? 32
            self.kvHeads = try container.decodeIfPresent(Int.self, forKey: .kvHeads) ?? 8
            self.linearNumValueHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumValueHeads) ?? 64
            self.linearNumKeyHeads =
                try container.decodeIfPresent(Int.self, forKey: .linearNumKeyHeads) ?? 16
            self.linearKeyHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearKeyHeadDim) ?? 192
            self.linearValueHeadDim =
                try container.decodeIfPresent(Int.self, forKey: .linearValueHeadDim) ?? 128
            self.linearConvKernelDim =
                try container.decodeIfPresent(Int.self, forKey: .linearConvKernelDim) ?? 4
            self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
            self.vocabularySize =
                try container.decodeIfPresent(Int.self, forKey: .vocabularySize) ?? 248_320
            self.maxPositionEmbeddings =
                try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 131_072
            self.tieWordEmbeddings =
                try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
            self.attentionBias =
                try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
            self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim)
            self.fullAttentionInterval =
                try container.decodeIfPresent(Int.self, forKey: .fullAttentionInterval) ?? 4

            self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 0
            self.numExpertsPerTok =
                try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 0
            self.decoderSparseStep =
                try container.decodeIfPresent(Int.self, forKey: .decoderSparseStep) ?? 1
            self.sharedExpertIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .sharedExpertIntermediateSize) ?? 0
            self.moeIntermediateSize =
                try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize) ?? 0
            self.normTopkProb =
                try container.decodeIfPresent(Bool.self, forKey: .normTopkProb) ?? true
            self.mtpNumHiddenLayers =
                try container.decodeIfPresent(Int.self, forKey: .mtpNumHiddenLayers) ?? 0

            let defaultRopeParameters: [String: StringOrNumber] = [
                "type": .string("default"),
                "mrope_section": .ints([11, 11, 10]),
                "rope_theta": .float(100_000.0),
                "partial_rotary_factor": .float(0.25),
            ]

            var decodedRope = try container.decodeIfPresent(
                [String: StringOrNumber].self, forKey: .ropeParameters)

            if decodedRope == nil {
                let ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta)
                let partial = try container.decodeIfPresent(
                    Float.self, forKey: .partialRotaryFactor)
                if ropeTheta != nil || partial != nil {
                    decodedRope = defaultRopeParameters
                    if let ropeTheta {
                        decodedRope?["rope_theta"] = .float(ropeTheta)
                    }
                    if let partial {
                        decodedRope?["partial_rotary_factor"] = .float(partial)
                    }
                }
            }

            if var decodedRope {
                if decodedRope["type"] == nil, let ropeType = decodedRope["rope_type"] {
                    decodedRope["type"] = ropeType
                }
                self.ropeParameters = decodedRope
                self.ropeTheta = decodedRope["rope_theta"]?.asFloat() ?? 100_000.0
                self.partialRotaryFactor = decodedRope["partial_rotary_factor"]?.asFloat() ?? 0.25
            } else {
                self.ropeParameters = defaultRopeParameters
                self.ropeTheta = 100_000.0
                self.partialRotaryFactor = 0.25
            }

            if self.headDim == nil {
                self.headDim = self.hiddenSize / self.attentionHeads
            }
        }
    }

    public typealias VisionConfiguration = Qwen3VLConfiguration.VisionConfiguration

    public let textConfiguration: TextConfiguration
    public let visionConfiguration: VisionConfiguration
    public let modelType: String
    private let _ignoreIndex: Int?
    public var ignoreIndex: Int { _ignoreIndex ?? -100 }
    private let _imageTokenId: Int?
    public var imageTokenId: Int { _imageTokenId ?? 248_056 }
    private let _videoTokenId: Int?
    public var videoTokenId: Int { _videoTokenId ?? 248_057 }
    private let _imageTokenIndex: Int?
    public var imageTokenIndex: Int { _imageTokenIndex ?? imageTokenId }
    private let _videoTokenIndex: Int?
    public var videoTokenIndex: Int { _videoTokenIndex ?? videoTokenId }
    private let _visionStartTokenId: Int?
    public var visionStartTokenId: Int { _visionStartTokenId ?? 248_045 }
    private let _visionEndTokenId: Int?
    public var visionEndTokenId: Int { _visionEndTokenId ?? 248_046 }
    private let _vocabSize: Int?
    public var vocabSize: Int { _vocabSize ?? textConfiguration.vocabularySize }
    private let _eosTokenId: IntOrIntArray?
    public var eosTokenId: [Int]? { _eosTokenId?.values }

    enum CodingKeys: String, CodingKey {
        case textConfiguration = "text_config"
        case visionConfiguration = "vision_config"
        case modelType = "model_type"
        case _ignoreIndex = "ignore_index"
        case _imageTokenId = "image_token_id"
        case _videoTokenId = "video_token_id"
        case _imageTokenIndex = "image_token_index"
        case _videoTokenIndex = "video_token_index"
        case _visionStartTokenId = "vision_start_token_id"
        case _visionEndTokenId = "vision_end_token_id"
        case _vocabSize = "vocab_size"
        case _eosTokenId = "eos_token_id"
    }
}

// MARK: - Language

enum Qwen35Language {

    final class RotaryEmbedding {
        private let invFreq: MLXArray
        private let mropeSection: [Int]

        init(dim: Int, base: Float, mropeSection: [Int]) {
            let safeDim = max(1, dim)
            var freq = MLXArray(stride(from: 0, to: safeDim, by: 2)).asType(.float32)
            freq = freq / Float(safeDim)
            self.invFreq = 1.0 / pow(MLXArray(base), freq)
            self.mropeSection =
                mropeSection.count >= 3 ? mropeSection : [11, 11, 10]
        }

        private func applyInterleavedMRope(_ freqs: MLXArray) -> MLXArray {
            let freqsT = freqs[0, 0..., 0..., 0...]
            let dims = freqsT.dim(-1)
            var slices: [MLXArray] = []
            slices.reserveCapacity(dims)

            for idx in 0 ..< dims {
                var slice = freqsT[0..., 0..., idx]
                for (dim, offset) in [(1, 1), (2, 2)] {
                    let length = min(mropeSection[dim] * 3, dims)
                    if idx >= offset && idx < length && ((idx - offset) % 3 == 0) {
                        slice = freqs[dim, 0..., 0..., idx]
                        break
                    }
                }
                slices.append(slice)
            }

            return stacked(slices, axis: -1)
        }

        func callAsFunction(x: MLXArray, positionIds: MLXArray) -> (MLXArray, MLXArray) {
            var positionIds = positionIds
            if positionIds.ndim == 2 {
                positionIds = broadcast(
                    positionIds[.newAxis, 0..., 0...],
                    to: [3, positionIds.dim(0), positionIds.dim(1)])
            }

            let pos = positionIds.asType(.float32)
            var inv = invFreq.asType(.float32)
            inv = inv[.newAxis, .newAxis, .newAxis, 0...]
            var freqs = pos[0..., 0..., 0..., .newAxis] * inv
            freqs = applyInterleavedMRope(freqs)

            let emb = concatenated([freqs, freqs], axis: -1)
            return (cos(emb).asType(x.dtype), sin(emb).asType(x.dtype))
        }
    }

    static func applyMultimodalRotaryPosEmb(
        q: MLXArray,
        k: MLXArray,
        cos: MLXArray,
        sin: MLXArray
    ) -> (MLXArray, MLXArray) {
        let cos = expandedDimensions(cos, axis: 1)
        let sin = expandedDimensions(sin, axis: 1)

        let rotaryDim = cos.dim(-1)
        let qDim = q.dim(-1)
        let kDim = k.dim(-1)

        let qRot = q[.ellipsis, ..<rotaryDim]
        let kRot = k[.ellipsis, ..<rotaryDim]

        let qEmbedded = (qRot * cos) + (QwenVL.rotateHalf(qRot) * sin)
        let kEmbedded = (kRot * cos) + (QwenVL.rotateHalf(kRot) * sin)

        let qOut: MLXArray
        if rotaryDim < qDim {
            qOut = concatenated([qEmbedded, q[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            qOut = qEmbedded
        }

        let kOut: MLXArray
        if rotaryDim < kDim {
            kOut = concatenated([kEmbedded, k[.ellipsis, rotaryDim...]], axis: -1)
        } else {
            kOut = kEmbedded
        }

        return (qOut, kOut)
    }

    final class RMSNormGated: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            self.eps = eps
            _weight.wrappedValue = MLXArray.ones([dimensions])
            super.init()
        }

        func callAsFunction(_ hiddenStates: MLXArray, gate: MLXArray? = nil) -> MLXArray {
            var x = MLXFast.rmsNorm(hiddenStates, weight: weight, eps: eps)
            if let gate {
                x = x * silu(gate)
            }
            return x
        }
    }

    final class Attention: Module {
        let numKeyValueHeads: Int
        let numAttentionHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "q_proj") var qProj: Linear
        @ModuleInfo(key: "k_proj") var kProj: Linear
        @ModuleInfo(key: "v_proj") var vProj: Linear
        @ModuleInfo(key: "o_proj") var oProj: Linear

        @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
        @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

        let rotaryEmbedding: RotaryEmbedding

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.numKeyValueHeads = args.kvHeads
            self.numAttentionHeads = args.attentionHeads
            self.headDim = args.headDim ?? (args.hiddenSize / args.attentionHeads)
            self.scale = pow(Float(headDim), -0.5)

            _qProj.wrappedValue = Linear(
                args.hiddenSize, numAttentionHeads * headDim * 2, bias: args.attentionBias)
            _kProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _vProj.wrappedValue = Linear(
                args.hiddenSize, numKeyValueHeads * headDim, bias: args.attentionBias)
            _oProj.wrappedValue = Linear(
                numAttentionHeads * headDim, args.hiddenSize, bias: args.attentionBias)

            _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)
            _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: args.rmsNormEps)

            let mrope = args.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            let rotaryDim = Int(Float(headDim) * args.partialRotaryFactor)
            self.rotaryEmbedding = RotaryEmbedding(
                dim: rotaryDim, base: args.ropeTheta, mropeSection: mrope)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            let B = x.dim(0)
            let L = x.dim(1)

            let qProjOutput = qProj(x)
            let qSplit = qProjOutput.reshaped(B, L, numAttentionHeads, -1).split(parts: 2, axis: -1)
            var queries = qSplit[0]
            let gate = qSplit[1].reshaped(B, L, -1)

            var keys = kProj(x)
            var values = vProj(x)

            queries = qNorm(queries).transposed(0, 2, 1, 3)
            keys = kNorm(keys.reshaped(B, L, numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
            values = values.reshaped(B, L, numKeyValueHeads, -1).transposed(0, 2, 1, 3)

            var kvSeqLen = keys.dim(-2)
            var positionIds = positionIds

            if positionIds == nil {
                let offset = cache?.offset ?? 0
                kvSeqLen += offset + 1
                var base = MLXArray(stride(from: offset, to: offset + L, by: 1)).asType(.int32)
                base = tiled(base[.newAxis, 0...], repetitions: [B, 1])
                positionIds = base[.newAxis, 0..., 0...]
                positionIds = tiled(positionIds!, repetitions: [3, 1, 1])
            } else if let cache {
                kvSeqLen += cache.offset + 1
            }

            let (cosValues, sinValues) = rotaryEmbedding(x: values, positionIds: positionIds!)
            (queries, keys) = applyMultimodalRotaryPosEmb(
                q: queries, k: keys, cos: cosValues, sin: sinValues)

            let attentionMask: MLXFast.ScaledDotProductAttentionMaskMode
            if let mask {
                attentionMask = .array(mask[.ellipsis, 0 ..< kvSeqLen])
            } else {
                attentionMask = .none
            }

            let output = attentionWithCacheUpdate(
                queries: queries,
                keys: keys,
                values: values,
                cache: cache,
                scale: scale,
                mask: attentionMask
            )
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

            return oProj(output * sigmoid(gate))
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "gate_proj") var gateProj: Linear
        @ModuleInfo(key: "down_proj") var downProj: Linear
        @ModuleInfo(key: "up_proj") var upProj: Linear

        init(dimensions: Int, hiddenDimensions: Int) {
            _gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            _downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
            _upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            downProj(silu(gateProj(x)) * upProj(x))
        }
    }

    final class GatedDeltaNet: Module {
        let hiddenSize: Int
        let numVHeads: Int
        let numKHeads: Int
        let headKDim: Int
        let headVDim: Int
        let keyDim: Int
        let valueDim: Int
        let convKernelSize: Int
        let convDim: Int

        @ModuleInfo(key: "conv1d") var conv1d: Conv1d
        @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
        @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
        @ModuleInfo(key: "in_proj_b") var inProjB: Linear
        @ModuleInfo(key: "in_proj_a") var inProjA: Linear

        @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
        @ParameterInfo(key: "A_log") var aLog: MLXArray

        @ModuleInfo(key: "norm") var norm: RMSNormGated
        @ModuleInfo(key: "out_proj") var outProj: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.hiddenSize = args.hiddenSize
            self.numVHeads = args.linearNumValueHeads
            self.numKHeads = args.linearNumKeyHeads
            self.headKDim = args.linearKeyHeadDim
            self.headVDim = args.linearValueHeadDim
            self.keyDim = headKDim * numKHeads
            self.valueDim = headVDim * numVHeads
            self.convKernelSize = args.linearConvKernelDim
            self.convDim = keyDim * 2 + valueDim

            precondition(
                numVHeads % numKHeads == 0,
                "num_v_heads (\(numVHeads)) must be divisible by num_k_heads (\(numKHeads))"
            )

            _conv1d.wrappedValue = Conv1d(
                inputChannels: convDim,
                outputChannels: convDim,
                kernelSize: convKernelSize,
                stride: 1,
                padding: 0,
                dilation: 1,
                groups: convDim,
                bias: false
            )

            _inProjQKV.wrappedValue = Linear(hiddenSize, keyDim * 2 + valueDim, bias: false)
            _inProjZ.wrappedValue = Linear(hiddenSize, valueDim, bias: false)
            _inProjB.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)
            _inProjA.wrappedValue = Linear(hiddenSize, numVHeads, bias: false)

            _dtBias.wrappedValue = MLXArray.ones([numVHeads])
            let a = MLXRandom.uniform(low: 0, high: 16, [numVHeads])
            _aLog.wrappedValue = log(a)

            _norm.wrappedValue = RMSNormGated(dimensions: headVDim, eps: args.rmsNormEps)
            _outProj.wrappedValue = Linear(valueDim, hiddenSize, bias: false)
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            mask: MLXArray? = nil,
            cache: MambaCache? = nil,
            nConfirmed: Int = 0
        ) -> MLXArray {
            let B = inputs.dim(0)
            let S = inputs.dim(1)

            var mixedQKV = inProjQKV(inputs)
            let z = inProjZ(inputs).reshaped(B, S, numVHeads, headVDim)
            let b = inProjB(inputs)
            let a = inProjA(inputs)

            let convState: MLXArray
            if let cacheState = cache?[0] {
                convState = cacheState
            } else {
                convState = MLXArray.zeros(
                    [B, max(0, convKernelSize - 1), convDim], dtype: inputs.dtype)
            }

            if let mask {
                mixedQKV = MLX.where(mask[.ellipsis, .newAxis], mixedQKV, 0)
            }

            // nConfirmed 两块处理：confirmed → 存快照，draft → 可回滚
            if nConfirmed > 0 && nConfirmed < S {
                // --- Chunk 1: Confirmed tokens ---
                let qkvC = mixedQKV[0..., ..<nConfirmed, 0...]
                let bC = b[0..., ..<nConfirmed, 0...]
                let aC = a[0..., ..<nConfirmed, 0...]
                let zC = z[0..., ..<nConfirmed, 0..., 0...]
                let maskC = mask?[0..., ..<nConfirmed]

                let convInputC = concatenated([convState, qkvC], axis: 1)
                let convStateC = convInputC[0..., (-(convKernelSize - 1))...]

                let convOutC = silu(conv1d(convInputC))
                let splitC = MLX.split(convOutC, indices: [keyDim, 2 * keyDim], axis: -1)
                let qC = splitC[0].reshaped(B, nConfirmed, numKHeads, headKDim)
                let kC = splitC[1].reshaped(B, nConfirmed, numKHeads, headKDim)
                let vC = splitC[2].reshaped(B, nConfirmed, numVHeads, headVDim)

                let dtypeC = qC.dtype
                let invScaleC = pow(Float(headKDim), -0.5)
                let qNormedC =
                    MLXArray(pow(invScaleC, 2)).asType(dtypeC)
                    * MLXFast.rmsNorm(qC, weight: MLXArray.mlxNone, eps: 1e-6)
                let kNormedC =
                    MLXArray(invScaleC).asType(dtypeC)
                    * MLXFast.rmsNorm(kC, weight: MLXArray.mlxNone, eps: 1e-6)

                var stateC = cache?[1]
                var outC: MLXArray
                (outC, stateC) = gatedDeltaUpdate(
                    q: qNormedC, k: kNormedC, v: vC,
                    a: aC, b: bC, aLog: aLog, dtBias: dtBias,
                    state: stateC, mask: maskC
                )
                outC = norm(outC, gate: zC)

                // ★ 保存快照 ★
                if let cache {
                    var snapshotState = [MLXArray]()
                    snapshotState.append(convStateC[.ellipsis])
                    if let s = stateC {
                        snapshotState.append(s[.ellipsis])
                    }
                    cache.rollbackState = snapshotState
                }

                // --- Chunk 2: Draft tokens ---
                let nDraft = S - nConfirmed
                let qkvD = mixedQKV[0..., nConfirmed..., 0...]
                let bD = b[0..., nConfirmed..., 0...]
                let aD = a[0..., nConfirmed..., 0...]
                let zD = z[0..., nConfirmed..., 0..., 0...]
                let maskD = mask?[0..., nConfirmed...]

                let convInputD = concatenated([convStateC, qkvD], axis: 1)
                let convStateD = convInputD[0..., (-(convKernelSize - 1))...]

                let convOutD = silu(conv1d(convInputD))
                let splitD = MLX.split(convOutD, indices: [keyDim, 2 * keyDim], axis: -1)
                let qD = splitD[0].reshaped(B, nDraft, numKHeads, headKDim)
                let kD = splitD[1].reshaped(B, nDraft, numKHeads, headKDim)
                let vD = splitD[2].reshaped(B, nDraft, numVHeads, headVDim)

                let dtypeD = qD.dtype
                let qNormedD =
                    MLXArray(pow(invScaleC, 2)).asType(dtypeD)
                    * MLXFast.rmsNorm(qD, weight: MLXArray.mlxNone, eps: 1e-6)
                let kNormedD =
                    MLXArray(invScaleC).asType(dtypeD)
                    * MLXFast.rmsNorm(kD, weight: MLXArray.mlxNone, eps: 1e-6)

                var stateD = stateC
                var outD: MLXArray
                (outD, stateD) = gatedDeltaUpdate(
                    q: qNormedD, k: kNormedD, v: vD,
                    a: aD, b: bD, aLog: aLog, dtBias: dtBias,
                    state: stateD, mask: maskD
                )
                outD = norm(outD, gate: zD)

                if let cache {
                    cache[0] = convStateD
                    cache[1] = stateD
                }

                let outFull = concatenated(
                    [outC.reshaped(B, nConfirmed, -1), outD.reshaped(B, nDraft, -1)], axis: 1)
                return outProj(outFull)
            }

            // 普通路径（无 nConfirmed）
            let convInput = concatenated([convState, mixedQKV], axis: 1)
            if let cache, convKernelSize > 1 {
                cache[0] = convInput[0..., (-(convKernelSize - 1))...]
            }

            let convOut = silu(conv1d(convInput))
            let split = MLX.split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
            let q = split[0].reshaped(B, S, numKHeads, headKDim)
            let k = split[1].reshaped(B, S, numKHeads, headKDim)
            let v = split[2].reshaped(B, S, numVHeads, headVDim)

            var state = cache?[1]
            let dtype = q.dtype
            let invScale = pow(Float(headKDim), -0.5)
            let qNormed =
                MLXArray(pow(invScale, 2)).asType(dtype)
                * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
            let kNormed =
                MLXArray(invScale).asType(dtype)
                * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)

            var out: MLXArray
            (out, state) = gatedDeltaUpdate(
                q: qNormed,
                k: kNormed,
                v: v,
                a: a,
                b: b,
                aLog: aLog,
                dtBias: dtBias,
                state: state,
                mask: mask
            )

            if let cache {
                cache[1] = state
            }

            out = norm(out, gate: z)
            return outProj(out.reshaped(B, S, -1))
        }
    }

    final class SparseMoeBlock: Module, UnaryLayer {
        let normTopkProb: Bool
        let numExperts: Int
        let topK: Int

        @ModuleInfo(key: "gate") var gate: Linear
        @ModuleInfo(key: "switch_mlp") var switchMLP: SwitchGLU

        @ModuleInfo(key: "shared_expert") var sharedExpert: MLP
        @ModuleInfo(key: "shared_expert_gate") var sharedExpertGate: Linear

        init(_ args: Qwen35Configuration.TextConfiguration) {
            self.normTopkProb = args.normTopkProb
            self.numExperts = args.numExperts
            self.topK = args.numExpertsPerTok

            _gate.wrappedValue = Linear(args.hiddenSize, args.numExperts, bias: false)
            _switchMLP.wrappedValue = SwitchGLU(
                inputDims: args.hiddenSize,
                hiddenDims: args.moeIntermediateSize,
                numExperts: args.numExperts
            )

            _sharedExpert.wrappedValue = MLP(
                dimensions: args.hiddenSize,
                hiddenDimensions: args.sharedExpertIntermediateSize
            )
            _sharedExpertGate.wrappedValue = Linear(args.hiddenSize, 1, bias: false)
            super.init()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var gates = gate(x)
            gates = MLX.softmax(gates, axis: -1, precise: true)

            let kth = gates.dim(-1) - topK
            let inds = MLX.argPartition(gates, kth: kth, axis: -1)[.ellipsis, kth...]
            var scores = MLX.takeAlong(gates, inds, axis: -1)
            if normTopkProb {
                scores = scores / scores.sum(axis: -1, keepDims: true)
            }

            let y = switchMLP(x, inds)
            let combined = (y * scores[.ellipsis, .newAxis]).sum(axis: -2)

            var sharedY = sharedExpert(x)
            sharedY = sigmoid(sharedExpertGate(x)) * sharedY

            return combined + sharedY
        }
    }

    final class DecoderLayer: Module {
        let isLinear: Bool

        @ModuleInfo(key: "self_attn") var selfAttn: Attention?
        @ModuleInfo(key: "linear_attn") var linearAttn: GatedDeltaNet?

        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

        @ModuleInfo(key: "mlp") var mlp: Module

        init(_ args: Qwen35Configuration.TextConfiguration, layerIdx: Int) {
            self.isLinear = (layerIdx + 1) % args.fullAttentionInterval != 0

            if isLinear {
                _linearAttn.wrappedValue = GatedDeltaNet(args)
            } else {
                _selfAttn.wrappedValue = Attention(args)
            }

            if args.numExperts > 0 {
                _mlp.wrappedValue = SparseMoeBlock(args)
            } else {
                _mlp.wrappedValue = MLP(
                    dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            }

            _inputLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = RMSNorm(
                dimensions: args.hiddenSize, eps: args.rmsNormEps)

            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            attentionMask: MLXArray?,
            ssmMask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?,
            nConfirmed: Int = 0
        ) -> MLXArray {
            let r: MLXArray
            if isLinear {
                r = linearAttn!(inputLayerNorm(x), mask: ssmMask, cache: cache as? MambaCache, nConfirmed: nConfirmed)
            } else {
                r = selfAttn!(
                    inputLayerNorm(x), mask: attentionMask, cache: cache, positionIds: positionIds)
            }

            let h = x + r
            return h + (mlp as! UnaryLayer)(postAttentionLayerNorm(h))
        }
    }

    final class Model: Module {
        @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
        @ModuleInfo(key: "layers") fileprivate var layers: [DecoderLayer]
        @ModuleInfo(key: "norm") var norm: RMSNorm

        let ssmIdx: Int
        let faIdx: Int

        init(_ args: Qwen35Configuration.TextConfiguration) {
            precondition(args.vocabularySize > 0)
            _embedTokens.wrappedValue = Embedding(
                embeddingCount: args.vocabularySize, dimensions: args.hiddenSize)
            _layers.wrappedValue = (0 ..< args.hiddenLayers).map {
                DecoderLayer(args, layerIdx: $0)
            }
            _norm.wrappedValue = RMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)

            self.ssmIdx = 0
            self.faIdx = args.fullAttentionInterval - 1
            super.init()
        }

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            positionIds: MLXArray? = nil,
            nConfirmed: Int = 0
        ) -> MLXArray {
            var hiddenStates: MLXArray
            if let inputsEmbeds {
                hiddenStates = inputsEmbeds
            } else {
                hiddenStates = embedTokens(inputs)
            }

            var cacheArray = cache
            if cacheArray == nil {
                cacheArray = Array(repeating: nil as KVCache?, count: layers.count)
            }

            let faMaskMode = createAttentionMask(
                h: hiddenStates, cache: cacheArray?[faIdx], returnArray: true)
            let faMask: MLXArray?
            if case .array(let arrayMask) = faMaskMode {
                faMask = arrayMask
            } else {
                faMask = nil
            }
            let ssmMask = createSSMMask(h: hiddenStates, cache: cacheArray?[ssmIdx] as? MambaCache)

            for (index, layer) in layers.enumerated() {
                let layerSSMMask = layer.isLinear ? ssmMask : nil
                hiddenStates = layer(
                    hiddenStates,
                    attentionMask: faMask,
                    ssmMask: layerSSMMask,
                    cache: cacheArray?[index],
                    positionIds: positionIds,
                    nConfirmed: layer.isLinear ? nConfirmed : 0
                )
            }

            return norm(hiddenStates)
        }
    }

    final class LanguageModel: Module {
        @ModuleInfo var model: Model
        @ModuleInfo(key: "lm_head") var lmHead: Linear?
        @ModuleInfo(key: "mtp") var mtp: MTPModule?

        let config: Qwen35Configuration
        let textConfig: Qwen35Configuration.TextConfiguration
        let modelType: String
        let kvHeads: [Int]

        /// Whether MTP speculative decoding is available
        var hasMTP: Bool { mtp != nil }

        /// 缓存最后一次 forward 的 hidden states（供 MTP 使用）
        fileprivate var _lastHiddenStates: MLXArray?

        init(_ config: Qwen35Configuration) {
            self.config = config
            self.textConfig = config.textConfiguration
            self.modelType = config.textConfiguration.modelType
            self.model = Model(config.textConfiguration)
            self.kvHeads = Array(
                repeating: config.textConfiguration.kvHeads,
                count: config.textConfiguration.hiddenLayers
            )

            if !config.textConfiguration.tieWordEmbeddings {
                _lmHead.wrappedValue = Linear(
                    config.textConfiguration.hiddenSize,
                    config.textConfiguration.vocabularySize,
                    bias: false)
            }

            // MTP 模块不在 init 中创建，避免 Module.update 报 keyNotFound
            // 改在 loadMTPWeights() 中按需创建

            super.init()
        }

        // MARK: - MTP Methods

        /// Forward pass returning (logits, hiddenStates) for MTP speculative decoding.
        /// 走完整 VLM callAsFunction 路径（position IDs、mask、state 全部正确）。
        func forwardWithHiddenStates(
            _ inputs: MLXArray, cache: [KVCache]?, state: LMOutput.State?, nConfirmed: Int = 0
        ) -> (logits: MLXArray, hiddenStates: MLXArray, nextState: LMOutput.State) {
            let typedCache: [KVCache?]? = cache?.map { $0 as KVCache? }
            let output = self(
                inputs,
                inputsEmbeds: nil,
                cache: typedCache,
                state: state,
                mask: nil,
                positionIds: nil,
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil,
                nConfirmed: nConfirmed
            )
            return (output.logits, _lastHiddenStates!, output.state!)
        }

        /// MTP head forward: given token IDs and backbone hidden states, produce draft logits.
        func mtpForward(
            _ tokenIds: MLXArray, hiddenStates: MLXArray, cache: [KVCache]?,
            positionIds: MLXArray?
        ) -> MLXArray? {
            guard let mtp else { return nil }
            let tokenEmb = model.embedTokens(tokenIds)
            let eNormed = mtp.eNorm(tokenEmb)
            let hNormed = mtp.hNorm(hiddenStates)
            // vLLM qwen3_next_mtp 参考实现确认：[embedding, hidden] 顺序
            // torch.cat([inputs_embeds, hidden_states], dim=-1)
            let combined = MLX.concatenated([eNormed, hNormed], axis: -1)
            var x = mtp.fc(combined)
            let faMaskMode = createAttentionMask(h: x, cache: cache?.first)
            let mask: MLXArray?
            if case .array(let a) = faMaskMode { mask = a } else { mask = nil }
            x = mtp.layers[0](x, mask: mask, cache: cache?.first, positionIds: positionIds)
            x = mtp.norm(x)
            let logits: MLXArray
            if let lmHead { logits = lmHead(x) }
            else { logits = model.embedTokens.asLinear(x) }
            return logits
        }

        /// Create new MTP KV cache (single layer for MTP head).
        func newMTPCache() -> [KVCache] {
            [KVCacheSimple()]
        }

        /// 手动加载 MTP 权重（绕过 Module.update 的 MXFP4 包装问题）
        func loadMTPWeights(_ weights: [String: MLXArray]) {
            // 按需创建 MTP 模块（不在 init 中创建以避免 update 报错）
            if mtp == nil && config.textConfiguration.mtpNumHiddenLayers > 0 {
                _mtp.wrappedValue = MTPModule(config.textConfiguration)
            }
            guard let mtp else { return }
            var mtpParams: [String: MLXArray] = [:]
            for (key, value) in weights {
                let stripped = key.replacingOccurrences(of: "language_model.mtp.", with: "")
                mtpParams[stripped] = value
            }
            // 逐个子模块 update，确保数组子模块（layers）也能加载
            let flatPairs = Array(mtpParams)
            let nested = NestedItem<String, MLXArray>.unflattened(flatPairs)

            // 先用整体 update（加载 fc, norm, eNorm, hNorm）
            try? mtp.update(parameters: ModuleParameters(item: nested), verify: .none)

            // 单独 update layers[0]（绕过可能的数组处理问题）
            if case .dictionary(let tree) = nested,
               case .array(let layerItems) = tree["layers"] ?? .none,
               layerItems.count > 0
            {
                try? mtp.layers[0].update(
                    parameters: ModuleParameters(item: layerItems[0]), verify: .none)
            }
        }

        // MARK: -

        func callAsFunction(
            _ inputs: MLXArray,
            inputsEmbeds: MLXArray? = nil,
            cache: [KVCache?]? = nil,
            state: LMOutput.State?,
            mask: MLXArray? = nil,
            positionIds providedPositionIds: MLXArray? = nil,
            pixelValues: MLXArray? = nil,
            imageGridTHW: [THW]? = nil,
            videoGridTHW: [THW]? = nil,
            nConfirmed: Int = 0
        ) -> LMOutput {
            var state = state ?? .init()

            // Ensure inputs is 2D [batch, seq]. Text-only callers (e.g.
            // WiredMemoryUtils, TokenIterator) may pass 1D token arrays.
            let inputs = inputs.ndim == 1 ? inputs.expandedDimensions(axis: 0) : inputs

            if pixelValues != nil {
                state[precomputedPositionIdsKey] = nil
                state[ropeDeltasKey] = nil
            }
            let precomputedPositionIds = state[precomputedPositionIdsKey]
            let ropeDeltas = state[ropeDeltasKey]

            var cacheOffset = 0
            if let cache, let faCache = cache[model.faIdx] {
                cacheOffset = faCache.offset
            }

            var ropeMask = mask
            if let mask, mask.dim(-1) != inputs.dim(-1) {
                ropeMask = nil
            }

            var positionIds = providedPositionIds
            if positionIds == nil && (ropeMask == nil || ropeMask?.ndim == 2) {
                // ★ 使用缓存的 position IDs：仅在 chunk prefill 内部（prefill 的总长度 > 当前 cacheOffset）
                //    排除 text generation 场景（prefill 长度 ≤ cacheOffset，缓存位置已"用完"）
                if let precomputedPositionIds, precomputedPositionIds.dim(-1) > cacheOffset {
                    let seqLength = inputs.dim(1)
                    positionIds =
                        precomputedPositionIds[
                            0..., 0..., cacheOffset ..< (cacheOffset + seqLength)]
                } else if imageGridTHW == nil && videoGridTHW == nil {
                    // ★ 纯文本快速路径：位置 = cacheOffset 起的线性绝对位置（M-RoPE 三通道相同）。
                    // 原先走 getRopeIndex 会对"增量后缀片段"返回 0-based 位置（无前缀上下文），
                    // 与前缀位置冲突；这里直接按 cacheOffset 连续排布，deltas 记 0
                    // （供后续 decode 步走 cacheOffset + delta 分支，行为不变）。
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)
                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])
                    base = base + MLXArray(Int32(cacheOffset))
                    positionIds = broadcast(base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                    state[ropeDeltasKey] = MLXArray.zeros([batchSize], dtype: .int32)
                } else if (cache != nil && cache?[model.faIdx] != nil && cacheOffset == 0)
                    || ropeDeltas == nil
                    || cache == nil
                {
                    let (computed, deltas) = Qwen3VLLanguage.getRopeIndex(
                            inputIds: inputs,
                            imageGridTHW: imageGridTHW,
                            videoGridTHW: videoGridTHW,
                            spatialMergeSize: config.visionConfiguration.spatialMergeSize,
                            imageTokenId: config.imageTokenId,
                            videoTokenId: config.videoTokenId,
                            visionStartTokenId: config.visionStartTokenId,
                            attentionMask: ropeMask)
                        positionIds = computed
                        state[precomputedPositionIdsKey] = computed
                        state[ropeDeltasKey] = deltas
                } else {
                    let batchSize = inputs.dim(0)
                    let seqLength = inputs.dim(1)

                    var delta = MLXArray(cacheOffset).asType(.int32)
                    if let ropeDeltas {
                        delta = delta + ropeDeltas.asType(.int32)
                    }

                    var base = MLXArray(0 ..< seqLength).asType(.int32)
                    base = broadcast(base[.newAxis, 0...], to: [batchSize, seqLength])

                    if delta.ndim == 0 {
                        delta = broadcast(delta, to: [batchSize])
                    } else if delta.dim(0) < batchSize {
                        delta = repeated(delta, count: batchSize, axis: 0)
                    } else if delta.dim(0) > batchSize {
                        delta = delta[0 ..< batchSize]
                    }

                    base = base + delta[0..., .newAxis]
                    positionIds = broadcast(
                        base[.newAxis, 0..., 0...], to: [3, batchSize, seqLength])
                }
            }

            var hiddenStates = model(
                inputs,
                inputsEmbeds: inputsEmbeds,
                cache: cache,
                positionIds: positionIds,
                nConfirmed: nConfirmed
            )

            // 缓存 hidden states 供 MTP forwardWithHiddenStates 使用
            // vLLM 参考确认：主模型返回 POST-norm hidden states 给 MTP
            // MTP 的 pre_fc_norm_hidden 会再次 normalize（double-norm 是正确行为）
            _lastHiddenStates = hiddenStates

            var out: MLXArray
            if let lmHead {
                out = lmHead(hiddenStates)
            } else {
                out = model.embedTokens.asLinear(hiddenStates)
            }

            return LMOutput(logits: out, state: state)
        }

        func makeCache(maxKVSize: Int?) -> [KVCache] {
            let headDim = textConfig.headDim ?? (textConfig.hiddenSize / textConfig.attentionHeads)
            let rotaryDims = max(1, Int(Float(headDim) * textConfig.partialRotaryFactor))
            let mrope = textConfig.ropeParameters?["mrope_section"]?.asInts() ?? [11, 11, 10]
            return model.layers.map { layer in
                if layer.isLinear {
                    return MambaCache()
                }
                if let maxKVSize {
                    return StreamingKVCache(
                        keep: 4,
                        windowSize: maxKVSize,
                        ropeDimensions: rotaryDims,
                        ropeBase: textConfig.ropeTheta,
                        ropeTraditional: false,
                        ropeScale: 1.0,
                        mropeSection: mrope
                    )
                }
                return KVCacheSimple()
            }
        }
    }
    // MARK: - MTP (Speculative Decoding)

    /// GemmaRMSNorm: `x * (1 + w) / rms(x)` — 与标准 RMSNorm (`x * w / rms(x)`) 不同。
    ///
    /// Qwen3.5/3.6 的所有 norm 层（包括 MTP）都使用 GemmaRMSNorm。
    /// 权重训练时以 `1 + w` 为有效缩放因子（初始化为 0，有效初始值为 1）。
    /// 用标准 RMSNorm 会导致有效权重错误（如 w=-0.44 时有效权重为 -0.44 而非 0.56），
    /// 造成符号翻转和反相关输出。
    /// 参考: vLLM `vllm.model_executor.layers.layernorm.GemmaRMSNorm`
    final class GemmaRMSNorm: Module {
        @ParameterInfo(key: "weight") var weight: MLXArray
        let eps: Float

        init(dimensions: Int, eps: Float = 1e-6) {
            _weight.wrappedValue = MLXArray.zeros([dimensions])
            self.eps = eps
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let rms = sqrt(mean(x * x, axis: -1, keepDims: true) + eps)
            return x / rms * (1.0 + weight)
        }
    }

    /// MTP block: single-layer transformer (attn + MLP) for draft prediction.
    /// Attention 直接复用 backbone 的 Attention 类（自带 gate + M-RoPE）。
    /// 维度从权重反推：24 heads, 4 kv_heads, head_dim=256
    ///   q_proj: 12288 = 24 * 256 * 2 (gate)  ← 2x 是 gate 机制
    ///   k_proj: 1024  = 4 * 256
    ///   o_proj: 6144  = 24 * 256 (无 gate)
    final class MTPBlock: Module {
        @ModuleInfo(key: "self_attn") var selfAttn: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP
        @ModuleInfo(key: "input_layernorm") var inputLayerNorm: GemmaRMSNorm
        @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: GemmaRMSNorm

        init(_ args: Qwen35Configuration.TextConfiguration) {
            var mtpArgs = args
            mtpArgs.attentionHeads = 24
            mtpArgs.kvHeads = 4
            mtpArgs.headDim = 256
            _selfAttn.wrappedValue = Attention(mtpArgs)

            _mlp.wrappedValue = MLP(dimensions: args.hiddenSize, hiddenDimensions: args.intermediateSize)
            _inputLayerNorm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _postAttentionLayerNorm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }

        func callAsFunction(
            _ x: MLXArray,
            mask: MLXArray?,
            cache: KVCache?,
            positionIds: MLXArray?
        ) -> MLXArray {
            var r = inputLayerNorm(x)
            r = selfAttn(r, mask: mask, cache: cache, positionIds: positionIds)
            r = x + r
            let n = postAttentionLayerNorm(r)
            let m = mlp(n)
            return r + m
        }
    }

    /// MTP module: shared across all layers for draft token prediction
    final class MTPModule: Module {
        @ModuleInfo(key: "pre_fc_norm_embedding") var eNorm: GemmaRMSNorm
        @ModuleInfo(key: "pre_fc_norm_hidden") var hNorm: GemmaRMSNorm
        @ModuleInfo(key: "fc") var fc: Linear
        @ModuleInfo(key: "layers") var layers: [MTPBlock]
        @ModuleInfo(key: "norm") var norm: GemmaRMSNorm

        init(_ args: Qwen35Configuration.TextConfiguration) {
            _eNorm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _hNorm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            _fc.wrappedValue = Linear(args.hiddenSize * 2, args.hiddenSize, bias: false)
            _layers.wrappedValue = [MTPBlock(args)]
            _norm.wrappedValue = GemmaRMSNorm(dimensions: args.hiddenSize, eps: args.rmsNormEps)
            super.init()
        }
    }
}

// MARK: - Model

public class Qwen35: Module, VLMModel, MTPCapableModel {
    @ModuleInfo(key: "vision_tower") private var visionModel: Qwen3VLVision.VisionModel
    @ModuleInfo(key: "language_model") fileprivate var languageModel: Qwen35Language.LanguageModel

    /// Whether MTP speculative decoding is available for this VLM
    public var hasMTP: Bool { languageModel.hasMTP }

    /// MTP 权重在 sanitize 中被捕获，模型加载后需手动注入
    nonisolated(unsafe) private static var pendingMTPWeights: [String: MLXArray] = [:]

    /// 缓存 prepare 的 state（position IDs 等），供 forwardWithHiddenStates 复用
    private var _mtpState: LMOutput.State?

    /// 主模型 KV cache 的引用，供 MTP 动态读取当前 position（避免 rejection trim 后位置过期）。
    /// 之前 MTP 的 Attention 用自身 KV cache offset (0,1,2...) 做 M-RoPE，
    /// 导致位置完全错误（应为 prompt_length, prompt_length+1, ...），输出近乎均匀分布。
    private var _mainCache: [KVCache]?

    /// Chunked prefill 进度回调 (已处理 token 数, 总数)。
    /// smlx 侧 set 此回调后，prepare() 会在每个 chunk 完成后调用。
    public var prefillProgressCallback: (@Sendable (_ processed: Int, _ total: Int) -> Void)?

    public let config: Qwen35Configuration

    public init(_ config: Qwen35Configuration) {
        self.config = config
        _visionModel.wrappedValue = Qwen3VLVision.VisionModel(config.visionConfiguration)
        _languageModel.wrappedValue = Qwen35Language.LanguageModel(config)
        super.init()
    }

    public var vocabularySize: Int { config.vocabSize }

    public var loraLayers: [Module] {
        languageModel.model.layers
    }

    public func newCache(parameters: GenerateParameters?) -> [KVCache] {
        languageModel.makeCache(maxKVSize: parameters?.maxKVSize)
    }

    private func mergeInputIdsWithImageFeatures(
        imageFeatures: MLXArray,
        inputEmbeds: MLXArray,
        inputIds: MLXArray,
        imageTokenIndex: Int,
        videoTokenIndex: Int
    ) throws -> (MLXArray, MLXArray) {
        let imageMask = (inputIds .== MLXArray(imageTokenIndex))
        let videoMask = (inputIds .== MLXArray(videoTokenIndex))
        var specialMask = imageMask .|| videoMask

        let nImageTokens = specialMask.sum().item(Int.self)

        specialMask = expandedDimensions(specialMask, axis: -1)
        let maskExpanded = broadcast(specialMask, to: inputEmbeds.shape)

        let nImageFeatures = imageFeatures.dim(0)
        let nImageMaskElements = maskExpanded.sum().item(Int.self)
        let imageFeatureSize = imageFeatures.size

        guard nImageMaskElements == imageFeatureSize else {
            throw Qwen35VLError.featureTokenMismatch(expected: nImageTokens, actual: nImageFeatures)
        }

        let originalShape = inputEmbeds.shape
        let flattenedEmbeds = inputEmbeds.flattened()
        let flattenedFeatures = imageFeatures.flattened()
        let flattenedMask = maskExpanded.flattened()

        let indices = nonZero(flattenedMask.asType(.bool))

        var result = flattenedEmbeds
        if !indices.isEmpty && indices.count == flattenedFeatures.size {
            let indexArray = MLXArray(indices.map { UInt32($0) })
            result[indexArray] = flattenedFeatures
        }

        result = result.reshaped(originalShape)
        let visualMask = specialMask.squeezed(axis: -1).asType(.bool)
        return (result, visualMask)
    }

    private func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }

    private func combinedFrames(imageFrames: [THW]?, videoFrames: [THW]?) -> [THW] {
        var frames: [THW] = []
        if let imageFrames { frames.append(contentsOf: imageFrames) }
        if let videoFrames { frames.append(contentsOf: videoFrames) }
        return frames
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize: Int?
    ) throws -> PrepareResult {
        let prefillStepSize = windowSize ?? 512
        let inputIds = input.text.tokens
        // inputIds is [batch, seq], use dim(1) for sequence length
        let totalLength = inputIds.dim(1)
        let typedCache = castCache(cache)
        let cacheOffset = typedCache?.first(where: { !($0 is MambaCache) })?.offset ?? 0

        // ── 1. 收集图片/视频的像素与帧网格（此处不跑 vision tower）────────────
        let visionDType = visionModel.patchEmbed.proj.weight.dtype
        var pixelParts: [MLXArray] = []
        var imageFrames: [THW]?
        var videoFrames: [THW]?
        if let image = input.image {
            pixelParts.append(image.pixels.asType(visionDType))
            imageFrames = image.frames
        }
        if let video = input.video {
            pixelParts.append(video.pixels.asType(visionDType))
            videoFrames = video.frames
        }
        let allPixelValues: MLXArray? = pixelParts.isEmpty ? nil : concatenated(pixelParts)

        // ── 2. 扫描 image/video pad token 段并与 frames 做结构校验 ────────────
        // replacePaddingTokens 展开规则：第 i 段 pad 数 = frame_i.product / mergeSize²，
        // 且段顺序与 frames 顺序一致。校验通过是增量 prefill 的前提。
        let mergeSize = config.visionConfiguration.spatialMergeSize
        let mergeSquare = mergeSize * mergeSize
        let ids = inputIds.asArray(Int.self)
        var runs: [(isVideo: Bool, range: Range<Int>)] = []
        var scanIdx = 0
        while scanIdx < ids.count {
            let t = ids[scanIdx]
            if t == config.imageTokenId || t == config.videoTokenId {
                var end = scanIdx + 1
                while end < ids.count && ids[end] == t { end += 1 }
                runs.append((t == config.videoTokenId, scanIdx ..< end))
                scanIdx = end
            } else {
                scanIdx += 1
            }
        }
        let imageRuns = runs.filter { !$0.isVideo }
        let videoRuns = runs.filter { $0.isVideo }

        var structureOK =
            imageRuns.count == (imageFrames?.count ?? 0)
            && videoRuns.count == (videoFrames?.count ?? 0)
        if structureOK, let imageFrames {
            for (run, frame) in zip(imageRuns, imageFrames)
            where run.range.count != frame.product / mergeSquare {
                structureOK = false
                break
            }
        }
        if structureOK, let videoFrames {
            for (run, frame) in zip(videoRuns, videoFrames)
            where run.range.count != frame.product / mergeSquare {
                structureOK = false
                break
            }
        }

        // ── 3. 判定增量可行性 ────────────────────────────────────────────────
        // cacheOffset == 0 → 全新 prefill；
        // cacheOffset > 0 且 prompt 是 cache 的严格前缀扩展（totalLength > cacheOffset、
        // 结构校验通过、无跨 cacheOffset 的图片段）→ 增量 prefill；
        // 否则视为 cache 与 prompt 发散（异常），抛出由上层重置 cache 后全量重试。
        var startPos = 0
        var cachedImageCount = 0
        var cachedVideoCount = 0
        if cacheOffset > 0 {
            let canIncrement =
                structureOK && totalLength > cacheOffset
                && runs.allSatisfy {
                    $0.range.upperBound <= cacheOffset || $0.range.lowerBound >= cacheOffset
                }
            guard canIncrement else {
                print(
                    "⚠️ [Qwen35] cache diverged: offset=\(cacheOffset) total=\(totalLength) "
                        + "runs=\(runs.map { "\($0.isVideo ? "v" : "i")\($0.range)" }) structureOK=\(structureOK)"
                )
                throw VLMError.cacheDiverged
            }
            startPos = cacheOffset
            cachedImageCount = imageRuns.filter { $0.range.upperBound <= cacheOffset }.count
            cachedVideoCount = videoRuns.filter { $0.range.upperBound <= cacheOffset }.count
        }
        let freshLength = totalLength - startPos
        if startPos > 0 {
            print(
                "⚡ [Qwen35] incremental prefill: skip=\(cacheOffset) fresh=\(freshLength)/\(totalLength) newImages=\(imageRuns.count - cachedImageCount) newVideos=\(videoRuns.count - cachedVideoCount)"
            )
        }

        // ── 4. vision tower：只跑新增图片/视频（旧图特征已在 KV cache 中）─────
        // 像素布局：[全部图片行..., 全部视频行...]，每张图行数 = frame.product（patchify）。
        let imageRowCounts = (imageFrames ?? []).map { $0.product }
        let videoRowCounts = (videoFrames ?? []).map { $0.product }
        var pixelValuesToEncode: MLXArray?
        var framesToEncode: [THW]?
        if let allPixelValues {
            if structureOK {
                let newImageStartRow = imageRowCounts[..<cachedImageCount].reduce(0, +)
                let newImageRowCount = imageRowCounts[cachedImageCount...].reduce(0, +)
                let newVideoStartRow =
                    imageRowCounts.reduce(0, +)
                    + videoRowCounts[..<cachedVideoCount].reduce(0, +)
                let newVideoRowCount = videoRowCounts[cachedVideoCount...].reduce(0, +)
                var newParts: [MLXArray] = []
                var newFrames: [THW] = []
                if newImageRowCount > 0 {
                    newParts.append(
                        allPixelValues[
                            newImageStartRow ..< (newImageStartRow + newImageRowCount), 0...])
                    newFrames.append(contentsOf: (imageFrames ?? [])[cachedImageCount...])
                }
                if newVideoRowCount > 0 {
                    newParts.append(
                        allPixelValues[
                            newVideoStartRow ..< (newVideoStartRow + newVideoRowCount), 0...])
                    newFrames.append(contentsOf: (videoFrames ?? [])[cachedVideoCount...])
                }
                if !newParts.isEmpty {
                    pixelValuesToEncode =
                        newParts.count == 1 ? newParts[0] : concatenated(newParts)
                    framesToEncode = newFrames
                }
            } else {
                // 结构异常（仅全新 prefill 可达）：退回旧行为，全量编码 + mask 合并
                pixelValuesToEncode = allPixelValues
                framesToEncode = combinedFrames(
                    imageFrames: imageFrames, videoFrames: videoFrames)
            }
        }

        // ── 5. 文本 embedding + 视觉特征合并 ─────────────────────────────────
        var inputEmbeddings: MLXArray?
        if let pixelValuesToEncode,
            let frames = framesToEncode?.nilIfEmpty
        {
            let textEmbeds = languageModel.model.embedTokens(inputIds)
            let (visionHidden, _) = visionModel(pixelValuesToEncode, gridTHW: frames)
            let visionFeatures = visionHidden.asType(textEmbeds.dtype)

            if structureOK {
                // 把新图特征写入其 pad token 的绝对位置（旧图位置不在 suffix 中，跳过）。
                // 注意迭代顺序必须与特征布局一致：[新图片..., 新视频...]（同 pixelParts），
                // 不能按 token 顺序（两者在图文交错时不同）。
                let newRunsInFeatureOrder =
                    imageRuns.filter { $0.range.lowerBound >= startPos }
                    + videoRuns.filter { $0.range.lowerBound >= startPos }
                var embeds = textEmbeds
                var featureOffset = 0
                for run in newRunsInFeatureOrder {
                    let n = run.range.count
                    embeds[0..., run.range, 0...] =
                        visionFeatures[featureOffset ..< (featureOffset + n), 0...]
                        .expandedDimensions(axis: 0)
                    featureOffset += n
                }
                guard featureOffset == visionFeatures.dim(0) else {
                    throw Qwen35VLError.featureTokenMismatch(
                        expected: featureOffset, actual: visionFeatures.dim(0))
                }
                inputEmbeddings = embeds
            } else {
                let (mergedEmbeds, _) = try mergeInputIdsWithImageFeatures(
                    imageFeatures: visionFeatures,
                    inputEmbeds: textEmbeds,
                    inputIds: inputIds,
                    imageTokenIndex: config.imageTokenIndex,
                    videoTokenIndex: config.videoTokenIndex
                )
                inputEmbeddings = mergedEmbeds
            }
        }

        // ── 6. 记录新增图片 token 掩码（StreamingKVCache evict 的 M-RoPE 分段 shift）
        // 旧图的掩码在以往轮次已记录；evict 时会同步裁剪掩码数组，保持对齐。
        if !runs.isEmpty {
            var suffixMask = [Bool](repeating: false, count: freshLength)
            for run in runs where run.range.lowerBound >= startPos {
                for i in run.range {
                    suffixMask[i - startPos] = true
                }
            }
            for c in cache {
                (c as? StreamingKVCache)?.updateImageMask(suffixMask, from: startPos)
            }
        }

        // ── 7. 全量 M-RoPE 位置预计算（增量与全量统一）────────────────────────
        // 后缀位置依赖完整上下文（含旧图 grid），必须一次性算全量再切片；
        // 纯文本时 getRopeIndex 返回线性绝对位置，切片后同样严格正确。
        let (fullPositionIds, ropeDeltas) = Qwen3VLLanguage.getRopeIndex(
            inputIds: inputIds,
            imageGridTHW: imageFrames,
            videoGridTHW: videoFrames,
            spatialMergeSize: mergeSize,
            imageTokenId: config.imageTokenId,
            videoTokenId: config.videoTokenId,
            visionStartTokenId: config.visionStartTokenId,
            attentionMask: input.text.mask
        )
        var precomputedState = LMOutput.State()
        precomputedState[precomputedPositionIdsKey] = fullPositionIds
        precomputedState[ropeDeltasKey] = ropeDeltas

        // ── 8. 分块或单次前向（只处理 cache 之外的新增后缀）───────────────────
        // 增量轮次的后缀通常较小，单次前向比分块更快（分块边界有 asyncEval 等待）；
        // 后缀超过 512 才分块（如本轮携带多张大图）。
        let chunkThreshold = startPos > 0 ? max(prefillStepSize, 512) : prefillStepSize
        var state: LMOutput.State?
        var lastOutput: LMOutput?

        // 先上报 0%：单次前向没有中间进度事件，否则 UI 在整个 prefill 期间无反馈
        prefillProgressCallback?(0, freshLength)

        if freshLength > chunkThreshold {
            // ★ Chunked prefill：分块处理长序列，降低 prefill 显存峰值。
            // position IDs 已在外层一次性算好，chunk 内直接切片，
            // languageModel 检测到 positionIds != nil 后跳过内部位置计算。
            var offset = 0
            while offset < freshLength {
                let chunkEnd = min(offset + prefillStepSize, freshLength)
                let absFrom = startPos + offset
                let absTo = startPos + chunkEnd
                let chunkIds = inputIds[0..., absFrom ..< absTo]
                let chunkEmbeds = inputEmbeddings?[0..., absFrom ..< absTo, 0...]
                let chunkPosIds = fullPositionIds[0..., 0..., absFrom ..< absTo]

                let output = languageModel(
                    chunkIds,
                    inputsEmbeds: chunkEmbeds,
                    cache: typedCache,
                    state: precomputedState,
                    mask: nil,
                    positionIds: chunkPosIds,
                    pixelValues: nil,
                    imageGridTHW: nil,
                    videoGridTHW: nil
                )
                state = output.state
                lastOutput = output
                asyncEval(cache)
                offset = chunkEnd
                prefillProgressCallback?(chunkEnd, freshLength)
            }
            eval(cache)
        } else {
            // 单次前向（增量小 prompt 或全新小 prompt）
            let output = languageModel(
                inputIds[0..., startPos...],
                inputsEmbeds: inputEmbeddings?[0..., startPos..., 0...],
                cache: typedCache,
                state: precomputedState,
                mask: nil,
                positionIds: fullPositionIds[0..., 0..., startPos ..< totalLength],
                pixelValues: nil,
                imageGridTHW: nil,
                videoGridTHW: nil
            )
            state = output.state
            lastOutput = output
            prefillProgressCallback?(freshLength, freshLength)
        }

        _mtpState = state  // 缓存 state 供 forwardWithHiddenStates 使用
        _mainCache = typedCache  // 保存主模型 cache 引用（供 MTP 动态读取 position）
        return .logits(lastOutput!)
    }

    public func callAsFunction(
        _ input: LMInput.Text, cache: [any KVCache]?, state: LMOutput.State?
    ) -> LMOutput {
        let typedCache = castCacheOptional(cache)
        let result = languageModel(
            input.tokens,
            inputsEmbeds: nil,
            cache: typedCache,
            state: state,
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil
        )
        return result
    }

    public func sanitize(weights: [String: MLXArray], metadata: [String: String]) -> [String:
        MLXArray]
    {
        // 捕获 MTP 权重供后续手动加载（Module.update 对 MXFP4 中的 BF16 MTP 权重不兼容）
        Self.pendingMTPWeights = [:]
        for (key, value) in weights where key.hasPrefix("mtp.") {
            Self.pendingMTPWeights["language_model." + key] = value
        }

        // MXFP4 等 MLX 格式需要 key 重命名（model.visual → vision_tower 等）
        if metadata["format"]?.lowercased() == "mlx" {
            return sanitizeWeightsOnly(weights)
        }
        return sanitize(weights: weights)
    }

    /// 模型加载后手动注入 MTP 权重（绕过 Module.update 的 MXFP4 兼容性问题）
    public func loadMTPWeights() {
        guard !Self.pendingMTPWeights.isEmpty else { return }
        languageModel.loadMTPWeights(Self.pendingMTPWeights)
        Self.pendingMTPWeights = [:]
        if languageModel.hasMTP {
            print("✅ [Qwen35] MTP enabled")
        }
    }

    // MARK: - MTP 推理接口（MTPCapableModel 协议）

    /// Hidden states from the last forward pass (prepare or forwardWithHiddenStates).
    /// MTP speculative decoding uses this to get the backbone hidden state
    /// **before** the next token is processed (position t-1).
    public var lastForwardHiddenStates: MLXArray? {
        languageModel._lastHiddenStates
    }

    public func forwardWithHiddenStates(
        _ inputs: MLXArray, cache: [KVCache]?, nConfirmed: Int
    ) -> (logits: MLXArray, hiddenStates: MLXArray) {
        let typedCache: [KVCache?]? = cache?.map { $0 as KVCache? }
        let output = languageModel(
            inputs,
            inputsEmbeds: nil,
            cache: typedCache,
            state: _mtpState,  // ← 复用 prepare 的 state（position IDs）
            mask: nil,
            positionIds: nil,
            pixelValues: nil,
            imageGridTHW: nil,
            videoGridTHW: nil,
            nConfirmed: nConfirmed
        )
        _mtpState = output.state  // 更新 state 供下次使用

        // ★ 保存主模型 cache 引用，供 MTP 动态读取 position（rejection trim 后也能拿到正确值）
        _mainCache = cache

        return (output.logits, languageModel._lastHiddenStates!)
    }

    // LanguageModel 协议：简化前向（无 state，返回 logits）
    public func callAsFunction(_ inputs: MLXArray, cache: [KVCache]?) -> MLXArray {
        let typedCache: [KVCache?]? = cache?.map { $0 as KVCache? }
        let output = languageModel(
            inputs, inputsEmbeds: nil, cache: typedCache, state: nil,
            mask: nil, positionIds: nil, pixelValues: nil,
            imageGridTHW: nil, videoGridTHW: nil
        )
        return output.logits
    }

    public func mtpForward(
        _ tokenIds: MLXArray, hiddenStates: MLXArray, cache: [KVCache]?
    ) -> MLXArray? {
            // 从主模型 cache 动态读取 position（rejection trim 后也能拿到正确值）
            let mtpPosition: Int
            if let mainCache = _mainCache {
                let faIdx = languageModel.model.faIdx
                mtpPosition = faIdx < mainCache.count ? mainCache[faIdx].offset : 0
            } else {
                mtpPosition = 0
            }

            // MTP 处理的 hidden states 来自主模型位置 [mtpPosition-L, ..., mtpPosition-1]
            let L = tokenIds.dim(1)
            let B = tokenIds.dim(0)
            let startPos = mtpPosition - L

            var delta = MLXArray(Int32(startPos)).asType(.int32)
            if let ropeDeltas = _mtpState?[ropeDeltasKey] {
                delta = delta + ropeDeltas.asType(.int32)
            }
            if delta.ndim == 0 {
                delta = broadcast(delta, to: [B])
            } else if delta.dim(0) < B {
                delta = repeated(delta, count: B, axis: 0)
            } else if delta.dim(0) > B {
                delta = delta[0 ..< B]
            }

            var base = MLXArray(0 ..< L).asType(.int32)
            base = broadcast(base[.newAxis, 0...], to: [B, L])
            base = base + delta[0..., .newAxis]
            let positionIds = broadcast(base[.newAxis, 0..., 0...], to: [3, B, L])

            return languageModel.mtpForward(tokenIds, hiddenStates: hiddenStates, cache: cache, positionIds: positionIds)
    }

    public func newMTPCache() -> [KVCache] {
        languageModel.newMTPCache()
    }

    public func newLanguageCache(maxKVSize: Int? = nil) -> [KVCache] {
        languageModel.makeCache(maxKVSize: maxKVSize)
    }

    /// Key 重命名 + conv1d/Conv3d 转轴 + MTP 过滤 + MXFP4 biases 补齐，
    /// 用于已量化的 MLX 格式权重。
    private func sanitizeWeightsOnly(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        var renamed: [String: (MLXArray, String)] = [:]


        for (key, value) in weights where !key.hasPrefix("mtp.") {
            var newKey = key
            if key.contains("model.visual") {
                newKey = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
            } else if key.contains("model.language_model") {
                newKey = key.replacingOccurrences(
                    of: "model.language_model", with: "language_model.model")
            } else if key.hasPrefix("model.") {
                newKey = "language_model." + key
            } else if key.contains("lm_head") && !key.hasPrefix("language_model.") {
                newKey = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            } else if key.hasPrefix("mtp.") {
                newKey = "language_model." + key
            }
            renamed[newKey] = (value, key)
        }

        // mlx-swift 0.31.4+ 原生支持 MXFP4（不需要 biases）
        let hasUnsanitizedConv1d = renamed.contains { k, v in
            k.contains("conv1d.weight") && v.0.dim(-1) != 1
        }
        let normSuffixes = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (k, (value, _)) in renamed {
            var v = value
            if k.contains("conv1d.weight") && v.dim(-1) != 1 && v.ndim == 3 {
                v = v.movedAxis(source: 2, destination: 1)
            }
            if k.contains("vision_tower") && k.hasSuffix(".weight") && v.ndim == 5 && v.shape[1] == 3 {
                v = v.movedAxis(source: 1, destination: v.ndim - 1)
            }
            // Norm bias 偏移（与 LLM Qwen35TextModel.sanitize 对齐）
            if hasUnsanitizedConv1d && normSuffixes.contains(where: { k.hasSuffix($0) })
                && v.ndim == 1
            {
                v = v + MLXArray(1, dtype: v.dtype)
            }
            sanitized[k] = v
        }

        return sanitized
    }

    public func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        // MTP 权重暂不过 Module.update 加载（MXFP4 格式兼容性问题）
        var weights = weights.filter { !$0.key.hasPrefix("mtp.") }

        if config.textConfiguration.tieWordEmbeddings {
            weights["lm_head.weight"] = nil
        }

        var sanitized: [String: MLXArray] = [:]
        sanitized.reserveCapacity(weights.count)

        let normKeys = [
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        ]

        for (key, originalValue) in weights {
            var key = key
            var value = originalValue

            if key.contains("model") {
                if key.contains("model.language_model") {
                    key = key.replacingOccurrences(
                        of: "model.language_model", with: "language_model.model")
                } else if key.contains("model.visual") {
                    key = key.replacingOccurrences(of: "model.visual", with: "vision_tower")
                } else if key.hasPrefix("model.") {
                    // Unified Qwen 3.5 checkpoints (e.g. Qwen3.5-0.8B-MLX-4bit) ship
                    // language model tensors at bare `model.*` paths instead of
                    // `model.language_model.*`. Mirror the LLM-side fallback.
                    key = "language_model." + key
                }
            } else if key.contains("lm_head") && !key.hasPrefix("language_model.") {
                key = key.replacingOccurrences(of: "lm_head", with: "language_model.lm_head")
            }

            if key.contains("conv1d.weight") && value.dim(-1) != 1 {
                value = value.movedAxis(source: 2, destination: 1)
            }
            if normKeys.contains(where: { key.hasSuffix($0) }) && value.ndim == 1 {
                value = value + MLXArray(1, dtype: value.dtype)
            }

            sanitized[key] = value
        }

        return visionModel.sanitize(weights: sanitized)
    }
}

extension Array where Element == THW {
    fileprivate var nilIfEmpty: [THW]? { isEmpty ? nil : self }
}

extension Qwen35 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}
