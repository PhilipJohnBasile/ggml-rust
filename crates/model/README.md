# ggml-model

Digest-bound GGUF model indexing for the Rust-native runtime.

`GgufModel::open` validates the complete GGUF layout, records an owned tensor
index, and binds the model to the SHA-256 digest of the mapped file. The CPU
materialization path supports F32, F16, Q4_0, and Q8_0 tensors and returns
`ggml-tensor` values. Quantized formats are decoded into owned F32 values.
