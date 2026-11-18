//! Bit-exact NumPy 2.0.2 default_rng integer sampling for Hy4 Flash Engram multipliers.

// Released checkpoints do not store Engram multipliers; they are regenerated from seed + 10007 * layer_id (1-based).

// Pinned to NumPy 2.0.2 PCG64 XSL-RR, SeedSequence pooling and Lemire bounded sampling; other versions unverified.


const std = @import("std");

/// NumPy: max(1, (np.iinfo(np.int64).max // compressed) // 2).
pub fn bound(compressed: u64) u64 {
    if (compressed == 0) return 1;
    const i64max: u64 = @intCast(std.math.maxInt(i64));
    return @max(1, (i64max / compressed) / 2);
}

/// The three odd Engram multipliers for a 1-based layer id.
pub fn multipliers(
    gpa: std.mem.Allocator,
    seed: u64,
    layer_id: u32,
    compressed: u64,
) ![3]u64 {
    _ = gpa; // Fixed-size stack buffers: no allocator needed by this port.
    const b = bound(compressed);
    const s = seed +% (10007 * @as(u64, layer_id));
    var pcg = Pcg64.init(s);
    var out: [3]u64 = undefined;
    for (&out) |*slot| {
        // integers(0, b) => inclusive rng = b - 1 in NumPy's C API.
        slot.* = pcg.boundedLemire64(b - 1) * 2 + 1;
    }
    return out;
}

// ---------------------------------------------------------------------------

const INIT_A: u32 = 0x43b0d7e5;
const MULT_A: u32 = 0x931e8875;
const INIT_B: u32 = 0x8b51f9dd;
const MULT_B: u32 = 0x58f38ded;
const MIX_MULT_L: u32 = 0xca01f9dd;
const MIX_MULT_R: u32 = 0x4973f715;
const XSHIFT: u5 = 16;

fn hashMix(value_in: u32, hash_const: *u32) u32 {
    var value = value_in ^ hash_const.*;
    hash_const.* *%= MULT_A;
    value *%= hash_const.*;
    value ^= value >> XSHIFT;
    return value;
}

fn mix(x: u32, y: u32) u32 {
    var result = MIX_MULT_L *% x -% MIX_MULT_R *% y;
    result ^= result >> XSHIFT;
    return result;
}

/// SeedSequence(seed).generate_state(4, np.uint64) -> four 64-bit words.
fn seedWords(seed: u64) [4]u64 {
    // _int_to_uint32_array: low 32-bit words first. A u64 seed yields <=2.
    const entropy: [2]u32 = .{ @truncate(seed), @truncate(seed >> 32) };
    const entropy_len: usize = if (seed == 0) 1 else if (entropy[1] != 0) 2 else 1;

    var pool: [4]u32 = undefined;
    var hc: u32 = INIT_A;
    for (0..4) |i| {
        pool[i] = hashMix(if (i < entropy_len) entropy[i] else 0, &hc);
    }
    // Mix all pool bits together so late entropy affects earlier words.
    for (0..4) |src| {
        for (0..4) |dst| {
            if (src != dst) pool[dst] = mix(pool[dst], hashMix(pool[src], &hc));
        }
    }
    // A u64 seed has at most two entropy words, so the extra-entropy loop in

    var words32: [8]u32 = undefined;
    var hc2: u32 = INIT_B;
    for (0..8) |i| {
        var v = pool[i % 4] ^ hc2;
        hc2 *%= MULT_B;
        v *%= hc2;
        v ^= v >> XSHIFT;
        words32[i] = v;
    }

    // generate_state(4, uint64): 8 uint32 words viewed as little-endian u64,
    var words: [4]u64 = undefined;
    for (0..4) |i| {
        words[i] = (@as(u64, words32[2 * i + 1]) << 32) | @as(u64, words32[2 * i]);
    }
    return words;
}

// ---------------------------------------------------------------------------

const MULT_128: u128 = (@as(u128, 2549297995355413924) << 64) | 4865540595714422341;

const Pcg64 = struct {
    state: u128,
    inc: u128,

    fn init(seed: u64) Pcg64 {
        const words = seedWords(seed);
        const initstate = (@as(u128, words[0]) << 64) | words[1];
        const initseq = (@as(u128, words[2]) << 64) | words[3];
        var self = Pcg64{
            .state = 0,
            .inc = (initseq << 1) | 1,
        };
        self.step();
        self.state +%= initstate;
        self.step();
        return self;
    }

    fn step(self: *Pcg64) void {
        self.state = self.state *% MULT_128 +% self.inc;
    }

    fn next(self: *Pcg64) u64 {
        self.step();
        const high: u64 = @truncate(self.state >> 64);
        const low: u64 = @truncate(self.state);
        const xored = high ^ low;
        const rot: u6 = @intCast(high >> 58);
        if (rot == 0) return xored;
        return (xored >> rot) | (xored << @as(u6, @intCast(64 - @as(u16, rot))));
    }

    /// NumPy bounded_lemire_uint64 with inclusive rng; rng+1 = upper bound.
    fn boundedLemire64(self: *Pcg64, rng: u64) u64 {
        const rng_excl = rng +% 1;
        var m: u128 = @as(u128, self.next()) *% @as(u128, rng_excl);
        var leftover: u64 = @truncate(m);
        if (leftover < rng_excl) {
            const threshold = (~rng) % rng_excl;
            while (leftover < threshold) {
                m = @as(u128, self.next()) *% @as(u128, rng_excl);
                leftover = @truncate(m);
            }
        }
        return @truncate(m >> 64);
    }
};

// ---------------------------------------------------------------------------

const compressed_vocab: u64 = 92620;

test "hy4 host: numpy default_rng bound matches the pinned 2.0.2 value" {
    try std.testing.expectEqual(@as(u64, 49791470723681), bound(compressed_vocab));
}

test "hy4 host: layer 2 Engram multipliers match the pinned PR values" {
    const got = try multipliers(std.testing.allocator, 0, 2, compressed_vocab);
    const want = [3]u64{ 81853575799317, 39690085502883, 24229915018165 };
    try std.testing.expectEqualSlices(u64, &want, &got);
}

test "hy4 host: layer 12 Engram multipliers match numpy 2.0.2" {
    const got = try multipliers(std.testing.allocator, 0, 12, compressed_vocab);
    const want = [3]u64{ 19457948045719, 44729556510419, 84779352922315 };
    try std.testing.expectEqualSlices(u64, &want, &got);
}

test "hy4 host: layer 22 Engram multipliers match numpy 2.0.2" {
    const got = try multipliers(std.testing.allocator, 0, 22, compressed_vocab);
    const want = [3]u64{ 72974230057307, 21530602121221, 55998057243097 };
    try std.testing.expectEqualSlices(u64, &want, &got);
}

test "hy4 host: layer 32 Engram multipliers match numpy 2.0.2" {
    const got = try multipliers(std.testing.allocator, 0, 32, compressed_vocab);
    const want = [3]u64{ 29128509562787, 16745155128523, 22474588302733 };
    try std.testing.expectEqualSlices(u64, &want, &got);
}

test "hy4 host: nonzero and wide seeds match numpy 2.0.2" {
    const cases = [_]struct { seed: u64, layer: u32, compressed: u64, want: [3]u64 }{
        .{ .seed = 12345, .layer = 7, .compressed = 92620, .want = .{ 275835071427, 7872696305635, 26831527372685 } },
        .{ .seed = 1099511627783, .layer = 3, .compressed = 92620,
            .want = .{ 323713822167, 96961610215025, 74825263230143 } },
        .{ .seed = 9223372036854775907, .layer = 9, .compressed = 92620,
            .want = .{ 33031356444871, 75079811088047, 46716261047845 } },
    };
    for (cases) |case| {
        const got = try multipliers(std.testing.allocator, case.seed, case.layer, case.compressed);
        try std.testing.expectEqualSlices(u64, &case.want, &got);
    }
}

test "hy4 host: bound tracks the compressed vocabulary size" {
    try std.testing.expectEqual(@as(u64, 46116860184273), bound(100000));
    const got = try multipliers(std.testing.allocator, 0, 1, 100000);
    const want = [3]u64{ 75936276614145, 4795930018387, 35633158494689 };
    try std.testing.expectEqualSlices(u64, &want, &got);
}

test "hy4 host: all multipliers are odd and inside the bound" {
    const b = bound(compressed_vocab);
    for ([_]u32{ 2, 12, 22, 32 }) |layer| {
        const got = try multipliers(std.testing.allocator, 0, layer, compressed_vocab);
        for (got) |m| {
            try std.testing.expectEqual(@as(u64, 1), m & 1);
            try std.testing.expect(m <= 2 * b - 1);
        }
    }
}
