# ggml-model

Digest-bound GGUF model indexing for the Rust-native runtime.

`GgufModel::open` validates the complete GGUF layout, records an owned tensor
index, and binds the model to the SHA-256 digest of the mapped file. The first
materialization path supports F32 tensors and returns `ggml-tensor` values.
Quantized storage types remain explicit errors until their decoders are added.
