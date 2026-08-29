# ggml-model

Digest-bound GGUF model indexing for the Rust-native runtime.

`GgufModel::open` validates the complete GGUF layout, records an owned tensor
index, and binds the model to the SHA-256 digest of the mapped file. The CPU
materialization path supports F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q2_K, Q3_K, Q4_K,
Q5_K, Q6_K, Q8_0, Q8_K, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ4_NL, and IQ4_XS tensors and returns `ggml-tensor` values.
Quantized formats are decoded into owned F32 values. The `matmul_f32_quantized`
path computes row-vector products directly from all supported quantized matrices
without materializing an F32 matrix, while `load_affine_quantized` converts
eligible matrices directly to MLX's packed affine representation.
