# llama-runtime

llama-runtime is the model-contract layer for the Rust-native Llama runtime.
It turns scalar architecture metadata in a validated GGUF file into a checked
LlamaConfig, then validates canonical Llama tensor names and shapes before
execution code is allowed to consume the model. It can now load supported
weights into the checked CPU tensor engine, maintain a bounded per-layer KV
cache, apply rotary embeddings, and perform deterministic greedy generation.

The CPU decoder is the reference path. It does not yet provide MLX or Metal
kernels, all GGML quantization formats, or a production sampling policy.
