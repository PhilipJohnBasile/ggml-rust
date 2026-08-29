# ggml-tensor

Checked CPU tensor primitives for the Rust-native GGML runtime.

The initial implementation owns row-major `f32` data and provides the
operations needed by the first decoder layers: matrix multiplication,
elementwise arithmetic, RMSNorm, SiLU, stable softmax over the last dimension,
rank-2 transpose, right-aligned broadcasting, and grouped-query scaled
dot-product attention with causal masking. Shape products, rank assumptions,
and non-finite results are validated rather than left to unchecked indexing.
