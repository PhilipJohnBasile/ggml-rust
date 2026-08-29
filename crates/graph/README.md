# ggml-graph

Safe Rust graph construction and evaluation for the native GGML runtime.

The graph stores MLX-style lazy tensor operations and evaluates them in
insertion order on the checked `ggml-tensor` CPU backend. It currently covers
inputs, constants, elementwise arithmetic, matrix multiplication, reshape,
transpose, broadcasting, RMSNorm, LayerNorm with independent optional weight
and bias, SiLU, stable softmax, and grouped-query scaled dot-product attention.
The same operation boundary is intended to be lowered to Apple GPU kernels as
the native device backend is added.
