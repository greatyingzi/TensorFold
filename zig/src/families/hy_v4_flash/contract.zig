//! Hy4 Flash declarations shared by the host export and the CUDA registration.
//! Kept import-free so it can be exported from tensorfold.zig without a module seam.

pub const model_type = "hy_v4_flash";
pub const formats = [_][]const u8{"bf16"};
pub const default_context: i64 = 32768;
pub const max_segments: u32 = 1;
