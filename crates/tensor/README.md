# ggml-tensor

Checked CPU tensor primitives for the Rust-native GGML runtime.

The initial implementation owns row-major `f32` data and provides the
operations needed by the first decoder layers: matrix multiplication,
elementwise arithmetic, RMSNorm, SiLU, and stable softmax over the last
dimension. Shape products, rank assumptions, and non-finite results are
validated rather than left to unchecked indexing.
