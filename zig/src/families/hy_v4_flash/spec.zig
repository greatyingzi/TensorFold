//! Draft acceptance protocol for Hy4 Flash speculative decoding.

// The drafter is not wired yet; this module pins the rule the engine must honor.
const std = @import("std");
const ops = @import("ops.zig");

/// Longest draft prefix whose tokens equal the target's greedy picks.

// `greedy` holds one target pick per draft position, in draft order.
pub fn acceptPrefix(draft: []const u32, greedy: []const u32) !usize {
    if (draft.len != greedy.len) return error.DraftLengthMismatch;
    var n: usize = 0;
    while (n < draft.len and draft[n] == greedy[n]) : (n += 1) {}
    return n;
}

/// Greedy token from target logits at one position.
pub fn pick(logits: []const f32) !u32 {
    return @intCast(try ops.argmax(logits));
}

test "hy4 host: draft acceptance keeps only the matching prefix" {
    const draft = [_]u32{ 3, 5, 7, 9 };
    const greedy = [_]u32{ 3, 5, 8, 9 };
    try std.testing.expectEqual(@as(usize, 2), try acceptPrefix(&draft, &greedy));
    try std.testing.expectEqual(@as(usize, 4), try acceptPrefix(&draft, &draft));
    try std.testing.expectError(error.DraftLengthMismatch, acceptPrefix(&draft, greedy[0..3]));
    const logits = [_]f32{ 0.1, 0.9, 0.2 };
    try std.testing.expectEqual(@as(u32, 1), try pick(&logits));
}
