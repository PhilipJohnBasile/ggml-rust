# ggml-agent-memory

Versioned, content-addressed memory primitives for long-running and multi-agent
Rust runtimes. The crate keeps shared knowledge read-only to agents, gives an
agent write access to its own scope, and provides compare-and-swap writes with
history and attribution. `DreamingPass` clones a stable input state so an
out-of-band curator can consolidate session transcripts without mutating the
live store until an atomic state-hash commit.
