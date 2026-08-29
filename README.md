# ggml-rust

A Rust 1.98 foundation for an independent GGML-compatible runtime.

This project is an independent implementation. The compatibility target is the public ggml C ABI and current little-endian ggml loader behavior for GGUF v2 and v3. The current foundation includes a strict parser for untrusted GGUF metadata and tensor tables, checked row-major CPU tensor operations, and a validated Llama decoder reference path. Like the current ggml loader, it rejects byte-swapped files and nested metadata arrays. Backend adapters remain separate layers.

The workspace requires Rust 1.98 or newer and pins Rust 1.98.0, with Clippy and rustfmt, for reproducible development builds.

Current workspace:

- llama-runtime: checked Llama metadata/layout admission, tokenizer metadata,
  KV-cache CPU decoding, linear RoPE scaling, and deterministic greedy or
  seeded constrained sampling generation
- `ggml-gguf`: bounded GGUF parser with byte-slice and safe seekable-reader validation paths
- `ggml-mmap`: explicitly size-bounded, read-only model file mapping
- `ggml-model`: digest-bound GGUF tensor index with F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_0, and Q8_K CPU materialization, plus direct row-vector matmul for all supported quantized formats without F32 materialization
- `ggml-tensor`: checked row-major CPU tensor primitives for decoder layers
- `gguf-inspect`: memory-mapped command-line metadata and tensor-table inspector

Build and test:

```sh
cargo test --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo run -p gguf-inspect -- model.gguf
cargo run -p gguf-inspect -- --max-file-bytes 1099511627776 model.gguf
```

`gguf-inspect` and `ggml-model` default to a 1 TiB file mapping limit. A mapped model must remain immutable while the command is running. The mapping holds a shared advisory file lock, but an uncooperative process can still truncate or rewrite the file. `ggml-mmap` therefore exposes file-backed mapping as an explicit unsafe boundary instead of claiming that the advisory lock makes construction safe.

The runtime keeps stable compatibility surfaces while internal modules migrate.
MLXcelerator consumes the Llama admission and CPU generation contracts. Its
separate native MLX backend keeps decoder linear layers on the Apple GPU while
the checked host path remains the fallback and parity reference.
