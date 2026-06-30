import Foundation
import MLX
import MLXNN
import Testing

@testable import MLXLMCommon

/// Tests for ``StreamingKVCache`` — the attention-sink + sliding-window cache that
/// re-indexes survivors via a uniform RoPE rotation instead of a full re-prefill.
///
/// These tests are fully self-contained: they construct keys with the framework's own
/// `MLXFast.RoPE` and assert numerical equivalence. No model download is required.
@Suite(.serialized)
struct StreamingKVCacheTests {

    // Standard (non-scaled) RoPE config, matching e.g. Qwen3.
    private let dims = 64
    private let base: Float = 1_000_000
    private let traditional = false
    private let scale: Float = 1.0

    private func rope(_ x: MLXArray, offset: Int) -> MLXArray {
        MLXFast.RoPE(
            x, dimensions: dims, traditional: traditional,
            base: base, scale: scale, offset: offset, freqs: nil)
    }

    // MARK: - Core math: uniform shift == re-baking at shifted positions

    /// The whole approach hinges on this identity:
    ///   R(delta) · R(p) = R(p + delta)
    /// i.e. applying a uniform shift to keys baked at positions [a, a+L) must equal
    /// baking the *same raw keys* at positions [a+delta, a+delta+L).
    @Test func uniformShiftMatchesRebaking() {
        MLXRandom.seed(0)
        let B = 1, H = 2, L = 8
        let raw = MLXRandom.normal([B, H, L, dims]).asType(.float32)
        eval(raw)

        let a = 100
        let deltas = [-1, -7, -40, -99, 3, 50]

        // Keys as they live in the cache: baked at absolute positions [a, a+L).
        let baked = rope(raw, offset: a)

        for delta in deltas {
            let shifted = applyUniformRoPEShift(
                baked, delta: delta,
                dimensions: dims, traditional: traditional, base: base, scale: scale)
            let expected = rope(raw, offset: a + delta)
            let maxDiff = abs(shifted - expected).max().item(Float.self)
            let close = allClose(shifted, expected, atol: 1e-4).item(Bool.self)
            #expect(close, "uniform shift by \(delta) did not match re-baking (maxDiff=\(maxDiff))")
        }
    }

    @Test func shiftByZeroIsIdentity() {
        MLXRandom.seed(1)
        let baked = rope(MLXRandom.normal([1, 2, 4, dims]).asType(.float32), offset: 10)
        let shifted = applyUniformRoPEShift(
            baked, delta: 0, dimensions: dims, traditional: traditional, base: base, scale: scale)
        #expect(allClose(shifted, baked).item(Bool.self))
    }

    // MARK: - End-to-end eviction correctness (vs. first principles)

    /// Build a cache the way the model would (rope keys at their positions, then `update`),
    /// evict, and compare against an independently-constructed ground truth:
    ///   - sinks: raw[0..keep) baked at positions [0, keep)         (unchanged)
    ///   - window: raw[keep+evict..N) baked at positions [keep, capacity)
    @Test func evictionProducesCorrectlyReindexedCache() {
        MLXRandom.seed(2)
        let B = 1, H = 2
        let keep = 4
        let windowSize = 16
        let capacity = keep + windowSize    // 20
        let N = 30                           // overflow by 10
        let evict = N - capacity             // 10

        // Raw (pre-RoPE) keys/values for the whole sequence, one row per position.
        let raw = MLXRandom.normal([B, H, N, dims]).asType(.float32)
        let values = MLXRandom.normal([B, H, N, dims]).asType(.float32)
        eval(raw, values)

        // The model rotates each token at its position before caching. RoPE(offset: 0)
        // rotates row i by position i, so this bakes each token at its own index.
        let bakedKeys = rope(raw, offset: 0)
        eval(bakedKeys)

        let cache = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base, ropeTraditional: traditional, ropeScale: scale)

        _ = cache.update(keys: bakedKeys, values: values)
        #expect(cache.offset == N)
        #expect(cache.needsEviction)

        cache.evictToWindow()
        #expect(cache.offset == capacity)
        #expect(!cache.needsEviction)

        // Cache state after eviction (sliced to valid length).
        let gotKeys = cache.state[0]
        let gotValues = cache.state[1]
        eval(gotKeys, gotValues)
        #expect(gotKeys.dim(2) == capacity)

        // Ground truth.
        let sinkRaw = raw[.ellipsis, ..<keep, 0...]
        let winRaw = raw[.ellipsis, (keep + evict)..., 0...]
        let expectedSink = rope(sinkRaw, offset: 0)             // positions [0, keep)
        let expectedWin = rope(winRaw, offset: keep)            // positions [keep, capacity)
        let expectedKeys = concatenated([expectedSink, expectedWin], axis: 2)

        let expectedValues = concatenated(
            [
                values[.ellipsis, ..<keep, 0...],
                values[.ellipsis, (keep + evict)..., 0...],
            ], axis: 2)
        eval(expectedKeys, expectedValues)

        #expect(
            allClose(gotKeys, expectedKeys, atol: 1e-4).item(Bool.self),
            "evicted+reindexed keys do not match ground truth")
        #expect(
            allClose(gotValues, expectedValues, atol: 1e-4).item(Bool.self),
            "evicted values do not match ground truth")
    }

    /// Sinks must be preserved byte-for-byte across eviction (never re-rotated).
    @Test func sinkTokensAreUntouched() {
        MLXRandom.seed(3)
        let keep = 4, windowSize = 8
        let N = 20
        let raw = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let values = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let bakedKeys = rope(raw, offset: 0)

        let cache = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base)
        _ = cache.update(keys: bakedKeys, values: values)

        let sinkBefore = bakedKeys[.ellipsis, ..<keep, 0...]
        cache.evictToWindow()
        let sinkAfter = cache.state[0][.ellipsis, ..<keep, 0...]
        eval(sinkBefore, sinkAfter)
        #expect(allClose(sinkBefore, sinkAfter, atol: 1e-5).item(Bool.self))
    }

    @Test func noEvictionWhenWithinCapacity() {
        MLXRandom.seed(4)
        let cache = StreamingKVCache(
            keep: 4, windowSize: 16, ropeDimensions: dims, ropeBase: base)
        let n = 10  // < capacity (20)
        let keys = rope(MLXRandom.normal([1, 2, n, dims]).asType(.float32), offset: 0)
        let values = MLXRandom.normal([1, 2, n, dims]).asType(.float32)
        _ = cache.update(keys: keys, values: values)

        #expect(!cache.needsEviction)
        cache.evictToWindow()  // no-op
        #expect(cache.offset == n)
        #expect(cache.state[0].dim(2) == n)
    }

    /// Repeated eviction (decode loop) must keep positions bounded and stay numerically sane.
    @Test func repeatedEvictionStaysBounded() {
        MLXRandom.seed(5)
        let keep = 4, windowSize = 8
        let capacity = keep + windowSize
        let cache = StreamingKVCache(
            keep: keep, windowSize: windowSize, ropeDimensions: dims, ropeBase: base)

        // Simulate a long decode: append one (already rope'd) token at a time.
        for step in 0 ..< 100 {
            let k = rope(MLXRandom.normal([1, 2, 1, dims]).asType(.float32), offset: cache.offset)
            let v = MLXRandom.normal([1, 2, 1, dims]).asType(.float32)
            _ = cache.update(keys: k, values: v)
            cache.evictToWindow()
            #expect(cache.offset <= capacity, "offset escaped capacity at step \(step)")
        }
        // After warm-up the cache is pinned at capacity.
        #expect(cache.offset == capacity)
        let st = cache.state
        eval(st)
        #expect(st[0].dim(2) == capacity)
        // Sanity: no NaNs crept in through repeated rotations.
        // allClose uses equalNaN=false, so a self-comparison fails iff a NaN is present.
        #expect(allClose(st[0], st[0]).item(Bool.self), "NaN detected after repeated eviction")
    }

    // MARK: - Serialization

    @Test func serializationRoundTripPreservesConfigAndState() throws {
        MLXRandom.seed(6)
        let cache = StreamingKVCache(
            keep: 3, windowSize: 12, ropeDimensions: dims, ropeBase: base,
            ropeTraditional: false, ropeScale: 1.0)
        let keys = rope(MLXRandom.normal([1, 8, 10, dims]).asType(.float32), offset: 0)
        let values = MLXRandom.normal([1, 8, 10, dims]).asType(.float32)
        _ = cache.update(keys: keys, values: values)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathExtension("safetensors")
        try savePromptCache(url: url, cache: [cache], metadata: [:])
        let (loaded, _) = try loadPromptCache(url: url)

        #expect(loaded.count == 1)
        let restored = try #require(loaded[0] as? StreamingKVCache)
        #expect(restored.keep == 3)
        #expect(restored.windowSize == 12)
        #expect(restored.metaState == cache.metaState)
        #expect(restored.offset == cache.offset)

        for (a, b) in zip(restored.state, cache.state) {
            eval(a, b)
            #expect(allClose(a, b, atol: 1e-4).item(Bool.self))
        }

        // A restored cache must still evict correctly.
        let more = rope(
            MLXRandom.normal([1, 8, 20, dims]).asType(.float32), offset: restored.offset)
        _ = restored.update(keys: more, values: MLXRandom.normal([1, 8, 20, dims]).asType(.float32))
        restored.evictToWindow()
        #expect(restored.offset == restored.capacity)
    }

    // MARK: - Partial eviction via evict(tokenCount:)

    /// evict(tokenCount:) should drop exactly the requested number of tokens and leave
    /// the rest intact. The returned count must equal the requested amount.
    @Test func evictExactTokenCount() {
        MLXRandom.seed(10)
        let keep = 4, windowSize = 20
        let N = 30  // 30 tokens in cache, capacity = 24, so 6 over
        let raw = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let values = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let bakedKeys = rope(raw, offset: 0)
        eval(raw, values, bakedKeys)

        let cache = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base, ropeTraditional: traditional, ropeScale: scale)
        _ = cache.update(keys: bakedKeys, values: values)
        #expect(cache.offset == N)
        #expect(cache.evictableCount == N - keep)  // 26

        // Evict only 10 tokens (not all the way to capacity).
        let evicted = cache.evict(tokenCount: 10)
        #expect(evicted == 10)
        #expect(cache.offset == N - 10)  // 20

        // Verify sinks untouched.
        let sinkAfter = cache.state[0][.ellipsis, ..<keep, 0...]
        let sinkExpected = bakedKeys[.ellipsis, ..<keep, 0...]
        eval(sinkAfter, sinkExpected)
        #expect(allClose(sinkAfter, sinkExpected, atol: 1e-5).item(Bool.self))

        // Verify surviving window keys match re-baked ground truth.
        let winRaw = raw[.ellipsis, (keep + 10)..., 0...]
        let expectedWinKeys = rope(winRaw, offset: keep)
        let gotWinKeys = cache.state[0][.ellipsis, keep..., 0...]
        eval(expectedWinKeys, gotWinKeys)
        #expect(
            allClose(gotWinKeys, expectedWinKeys, atol: 1e-4).item(Bool.self),
            "partial evict keys mismatch")
    }

    /// evict(tokenCount:) with a count larger than evictable should clamp and not crash.
    @Test func evictClampsToAvailable() {
        MLXRandom.seed(11)
        let keep = 4, windowSize = 16
        let N = 12  // less than capacity (20), only 8 non-sink tokens
        let cache = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base)
        let keys = rope(MLXRandom.normal([1, 2, N, dims]).asType(.float32), offset: 0)
        let values = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        _ = cache.update(keys: keys, values: values)

        #expect(cache.evictableCount == N - keep)  // 8

        // Request evicting 100 tokens — should clamp to 8.
        let evicted = cache.evict(tokenCount: 100)
        #expect(evicted == N - keep)  // 8
        #expect(cache.offset == keep) // only sinks remain
    }

    /// evictToWindow should produce exactly the same result as evict(tokenCount: overflow).
    @Test func evictToWindowMatchesExplicitEvict() {
        MLXRandom.seed(12)
        let keep = 4, windowSize = 16
        let capacity = keep + windowSize
        let N = 30
        let raw = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let values = MLXRandom.normal([1, 2, N, dims]).asType(.float32)
        let bakedKeys = rope(raw, offset: 0)
        eval(raw, values, bakedKeys)

        // Path A: evictToWindow()
        let cacheA = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base, ropeTraditional: traditional, ropeScale: scale)
        _ = cacheA.update(keys: bakedKeys, values: values)
        cacheA.evictToWindow()

        // Path B: evict(tokenCount: N - capacity)
        let cacheB = StreamingKVCache(
            keep: keep, windowSize: windowSize,
            ropeDimensions: dims, ropeBase: base, ropeTraditional: traditional, ropeScale: scale)
        _ = cacheB.update(keys: bakedKeys, values: values)
        cacheB.evict(tokenCount: N - capacity)

        #expect(cacheA.offset == cacheB.offset)
        eval(cacheA.state[0], cacheA.state[1], cacheB.state[0], cacheB.state[1])
        #expect(allClose(cacheA.state[0], cacheB.state[0], atol: 1e-6).item(Bool.self))
        #expect(allClose(cacheA.state[1], cacheB.state[1], atol: 1e-6).item(Bool.self))
    }

    @Test func copyIsIndependent() {
        MLXRandom.seed(7)
        let cache = StreamingKVCache(
            keep: 4, windowSize: 8, ropeDimensions: dims, ropeBase: base)
        let keys = rope(MLXRandom.normal([1, 2, 6, dims]).asType(.float32), offset: 0)
        _ = cache.update(keys: keys, values: MLXRandom.normal([1, 2, 6, dims]).asType(.float32))

        let copied = try! #require(cache.copy() as? StreamingKVCache)
        #expect(copied.keep == cache.keep)
        #expect(copied.windowSize == cache.windowSize)
        #expect(copied.offset == cache.offset)

        // Mutating the copy must not affect the original.
        let before = cache.offset
        _ = copied.update(
            keys: rope(MLXRandom.normal([1, 2, 3, dims]).asType(.float32), offset: copied.offset),
            values: MLXRandom.normal([1, 2, 3, dims]).asType(.float32))
        #expect(cache.offset == before)
    }
}
