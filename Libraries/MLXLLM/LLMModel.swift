// Copyright © 2024 Apple Inc.

import MLX
import MLXLMCommon

/// Marker protocol for LLMModels
public protocol LLMModel: LanguageModel, LoRAModel {

    /// Models can implement this is they need a custom `MessageGenerator`.
    ///
    /// The default implementation returns `DefaultMessageGenerator`.
    func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator
}

extension LLMModel {

    /// Default prepare step for ``LLMModel``.
    ///
    /// This will evaluate the prompt in chunks until there is a small number of
    /// tokens left to feed into the `TokenIterator`.
    ///
    /// When a non-empty KV cache is provided (offset > 0), the tokens that have
    /// already been processed are skipped — only the new suffix is prefilled.
    /// This enables efficient multi-turn conversations where the shared prefix
    /// (system prompt + prior turns) is cached across calls.
    public func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) throws
        -> PrepareResult
    {
        let prefillStepSize = windowSize ?? 512
        var y = input.text

        // If the cache already contains processed tokens, skip the prefix that
        // is already represented in the cache. This avoids redundant prefill work
        // in multi-turn scenarios where the caller provides the full token sequence
        // (system + history + new message) and reuses the same KV cache.
        //
        // NOTE: We use max offset across all cache layers because hybrid models
        // (e.g. Qwen3.5) mix KVCacheSimple (offset updated) with MambaCache
        // (offset stays 0). Using cache.first?.offset would incorrectly return 0
        // when the first layer is a MambaCache.
        let cacheOffset = cache.map { $0.offset }.max() ?? 0
        if cacheOffset > 0 && y.tokens.size > cacheOffset {
            y = y[cacheOffset...]
        }

        // Prepare the prompt in chunks if larger than the prefill size.
        // asyncEval lets the CPU build chunk N+1's graph while the GPU evaluates
        // chunk N.
        var state: LMOutput.State?
        while y.tokens.size > prefillStepSize {
            let input = y[.newAxis, ..<prefillStepSize]
            let output = self(input, cache: cache.isEmpty ? nil : cache, state: state)
            state = output.state
            asyncEval(cache)
            y = y[prefillStepSize...]
        }

        // Single sync after the loop to flush any remaining async work.
        eval(cache)

        return .tokens(y)
    }

    public func messageGenerator(tokenizer: Tokenizer) -> MessageGenerator {
        DefaultMessageGenerator()
    }
}
