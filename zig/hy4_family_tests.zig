//! Host contracts for Hy4 Flash bring-up: no device, no checkpoint.
test "hy4 host: imports" {
    _ = @import("src/families/hy_v4_flash/config.zig");
    _ = @import("src/families/hy_v4_flash/tensor_names.zig");
    _ = @import("src/families/hy_v4_flash/index_test.zig");
    _ = @import("src/families/hy_v4_flash/graph.zig");
    _ = @import("src/families/hy_v4_flash/capability.zig");
    _ = @import("src/families/hy_v4_flash/engram_rng.zig");
    _ = @import("src/families/hy_v4_flash/drafter.zig");
    _ = @import("src/families/hy_v4_flash/loader.zig");
    _ = @import("src/families/hy_v4_flash/shapes.zig");
    _ = @import("src/families/hy_v4_flash/weights.zig");
    _ = @import("src/families/hy_v4_flash/ops.zig");
    _ = @import("src/families/hy_v4_flash/forward_smoke.zig");
    _ = @import("src/families/hy_v4_flash/moe.zig");
    _ = @import("src/families/hy_v4_flash/ihc.zig");
    _ = @import("src/families/hy_v4_flash/indexer.zig");
    _ = @import("src/families/hy_v4_flash/engram.zig");
    _ = @import("src/families/hy_v4_flash/rotary.zig");
    _ = @import("src/families/hy_v4_flash/full_forward.zig");
    _ = @import("src/families/hy_v4_flash/attention.zig");
    _ = @import("src/families/hy_v4_flash/session.zig");
    _ = @import("src/families/hy_v4_flash/recall.zig");
    _ = @import("src/families/hy_v4_flash/affine4.zig");
}
