# Hy4 Flash serving integration plan (research receipts)

## Registration surface
- `core/registry.zig` is the quantized-kernel registry, NOT model families. Do not touch it unless adding kernels.
- Family contract: `core/engine_api.zig`: Family (:263-269), Open (:275-353), Opened (:370-377), Engine.VTable (:413-445).
- Lane backend: `core/lanes/backend.zig`: Backend (:75-87), VTable (:88-158). Mandatory: prefill, first, queue, read, verify, keep, draft, release.
- CUDA registration: `native/cuda_families.zig`: registry tuple (:17) + api.Family list (:18-28); dispatch in `native/cuda.zig`:159-181; `openWith` (:182+) requires model_type, formats, default_context, max_segments, Options, Loaded, open. Build modules in `zig/build/cuda_engines.zig`:80-92.
- Server is family-agnostic: `server/main.zig`:154 reads model_type, :182 starts tokenizer+engine; `server/engines.zig`:32-39 publishes capabilities, :53-69 opens engine. HTTP routes in `server/serve.zig`/`chat.zig`.

## Smallest templates
- Metal chat family: Flint — `zig/src/families/flint/` (12 files) + `native/flint.zig` + registration in `native/metal.zig`:70,171 + export in `zig/src/tensorfold.zig`.
- CUDA chat family: Dory — `zig/src/families/dory/` (18 files, cuda_native.zig:14-43,109+) + `build/dory.zig`:5 + `cuda_engines.zig`:91 + `cuda_families.zig`:17.

## Tokenizer
- Checkpoint already has tokenizer.json, tokenizer_config.json, chat_template.jinja, generation_config.json — serving can use the standard HfText path (`server/hf_text.zig`:68-75, template search :167-198). No builtIn hook needed.

## Build / launch
- CUDA: `zig build -Dgpu=spark -Dnvcc=/usr/local/cuda/bin/nvcc` → `zig-out/native/bin/tensorfold-native`.
- Serve: `tensorfold-native serve MODEL --name local-model --parallel N --context N --port P`.

## Hy4 gaps (as of this round)
- No api.Family entry; no backend.zig implementing lanes.backend.Backend.VTable; no native/hy4*.zig host open().
- Not exported in `zig/src/tensorfold.zig`, not in any platform family registry/build module table.
- session.zig is token-by-token CPU forward; no prefill/chunk/verify/keep/lane engine; sparse indexer recall not wired; vision tower not forward-wired.
- capability.zig admits only bf16; EXL3/affine4 readers not exposed through formats/registration.
- Minimal serving path: real EXL3 GPU forward (in progress) → backend VTable + native open → CUDA family registration + build export → (tokenizer already in pack) → build tensorfold-native → serve → quality/throughput validation.
