//! Host contracts for Hy4 Flash bring-up: no device, no checkpoint.
test "hy4 host: imports" {
    _ = @import("src/families/hy_v4_flash/config.zig");
    _ = @import("src/families/hy_v4_flash/tensor_names.zig");
    _ = @import("src/families/hy_v4_flash/index_test.zig");
}
