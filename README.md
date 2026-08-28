# ggml-rust

A Rust 1.98 foundation for an independent GGML-compatible runtime.

This project is an independent implementation. The compatibility target is the public ggml C ABI and current little-endian ggml loader behavior for GGUF v2 and v3. The initial milestone is a strict parser for untrusted GGUF metadata and tensor tables. Like the current ggml loader, this milestone rejects byte-swapped files and nested metadata arrays. Graph execution, allocation, CPU kernels, and backend adapters are not implemented yet and follow in separate layers.

The workspace requires Rust 1.98 or newer and pins Rust 1.98.0, with Clippy and rustfmt, for reproducible development builds.

Current workspace:

- `ggml-gguf`: bounded GGUF parser with byte-slice and safe seekable-reader validation paths
- `ggml-mmap`: explicitly size-bounded, read-only model file mapping
- `ggml-model`: digest-bound GGUF tensor index with F32, F16, Q4_0, Q4_K, and Q8_0 CPU materialization
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

The runtime will keep stable compatibility surfaces while internal modules migrate. A separate `llama-rust` runtime will consume this workspace once model loading and graph execution are ready.
