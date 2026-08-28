# llama-runtime

llama-runtime is the model-contract layer for the Rust-native Llama runtime.
It turns scalar architecture metadata in a validated GGUF file into a checked
LlamaConfig, then validates canonical Llama tensor names and shapes before
execution code is allowed to consume the model.

This crate does not claim tokenizer support, KV-cache management, decoder
execution, or text generation yet. Those layers will consume this validated
contract and must preserve its content-bound model identity.
