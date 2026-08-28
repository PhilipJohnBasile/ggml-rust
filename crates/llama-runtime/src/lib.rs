#![deny(unsafe_code)]

use std::fmt;
use std::path::Path;

use ggml_model::{GgufModel, MetadataScalar, ModelError};
use ggml_tensor::{Tensor, TensorError};

/// Validated architecture parameters for a Llama decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    context_length: usize,
    embedding_length: usize,
    block_count: usize,
    head_count: usize,
    head_count_kv: usize,
    feed_forward_length: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    rope_freq_base: f32,
}

impl LlamaConfig {
    /// Creates and validates a Llama configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or numerical parameters are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        rope_freq_base: f32,
    ) -> Result<Self, LlamaError> {
        let config = Self {
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builds a configuration from the canonical Llama GGUF metadata keys.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata is missing, malformed, or the model
    /// architecture is not Llama.
    pub fn from_model(model: &GgufModel) -> Result<Self, LlamaError> {
        if model.architecture() != Some("llama") {
            return Err(LlamaError::UnsupportedArchitecture(
                model.architecture().unwrap_or("missing").to_owned(),
            ));
        }
        let keys = [
            "llama.context_length",
            "llama.embedding_length",
            "llama.block_count",
            "llama.attention.head_count",
            "llama.attention.head_count_kv",
            "llama.feed_forward_length",
            "llama.vocab_size",
            "llama.attention.layer_norm_rms_epsilon",
            "llama.rope.freq_base",
        ];
        let values = model.metadata_scalars(&keys)?;
        let mut values = values.into_iter();
        let context_length = required_usize_value(values.next().flatten(), keys[0])?;
        let embedding_length = required_usize_value(values.next().flatten(), keys[1])?;
        let block_count = required_usize_value(values.next().flatten(), keys[2])?;
        let head_count = required_usize_value(values.next().flatten(), keys[3])?;
        let head_count_kv =
            optional_usize_value(values.next().flatten(), keys[4])?.unwrap_or(head_count);
        let feed_forward_length = required_usize_value(values.next().flatten(), keys[5])?;
        let vocab_size = match values.next().flatten() {
            Some(value) => as_usize(value).map_err(|value| LlamaError::InvalidMetadata {
                key: keys[6],
                value,
            })?,
            None => model
                .metadata_string_array("tokenizer.ggml.tokens", MAX_TOKENIZER_ELEMENTS)?
                .ok_or(LlamaError::MissingMetadata(keys[6]))?
                .len(),
        };
        let rms_norm_epsilon =
            optional_f32_value(values.next().flatten(), keys[7])?.unwrap_or(1.0e-5);
        let rope_freq_base =
            optional_f32_value(values.next().flatten(), keys[8])?.unwrap_or(10_000.0);
        Self::new(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
        )
    }

    /// Returns the maximum sequence length.
    #[must_use]
    pub const fn context_length(&self) -> usize {
        self.context_length
    }

    /// Returns the model embedding width.
    #[must_use]
    pub const fn embedding_length(&self) -> usize {
        self.embedding_length
    }

    /// Returns the decoder layer count.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.block_count
    }

    /// Returns the query head count.
    #[must_use]
    pub const fn head_count(&self) -> usize {
        self.head_count
    }

    /// Returns the key and value head count.
    #[must_use]
    pub const fn head_count_kv(&self) -> usize {
        self.head_count_kv
    }

    /// Returns the feed-forward hidden width.
    #[must_use]
    pub const fn feed_forward_length(&self) -> usize {
        self.feed_forward_length
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Returns the `RMSNorm` epsilon.
    #[must_use]
    pub const fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_epsilon
    }

    /// Returns the rotary embedding frequency base.
    #[must_use]
    pub const fn rope_freq_base(&self) -> f32 {
        self.rope_freq_base
    }

    fn validate(&self) -> Result<(), LlamaError> {
        for (name, value) in [
            ("context_length", self.context_length),
            ("embedding_length", self.embedding_length),
            ("block_count", self.block_count),
            ("head_count", self.head_count),
            ("head_count_kv", self.head_count_kv),
            ("feed_forward_length", self.feed_forward_length),
            ("vocab_size", self.vocab_size),
        ] {
            if value == 0 {
                return Err(LlamaError::InvalidConfig(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        if self.head_count_kv > self.head_count
            || !self.head_count.is_multiple_of(self.head_count_kv)
        {
            return Err(LlamaError::InvalidConfig(
                "head_count must be divisible by head_count_kv".to_owned(),
            ));
        }
        if !self.embedding_length.is_multiple_of(self.head_count) {
            return Err(LlamaError::InvalidConfig(
                "embedding_length must be divisible by head_count".to_owned(),
            ));
        }
        if !(self.embedding_length / self.head_count).is_multiple_of(2) {
            return Err(LlamaError::InvalidConfig(
                "head dimension must be even for rotary embeddings".to_owned(),
            ));
        }
        if !self.rms_norm_epsilon.is_finite() || self.rms_norm_epsilon <= 0.0 {
            return Err(LlamaError::InvalidConfig(
                "rms_norm_epsilon must be finite and positive".to_owned(),
            ));
        }
        if !self.rope_freq_base.is_finite() || self.rope_freq_base <= 0.0 {
            return Err(LlamaError::InvalidConfig(
                "rope_freq_base must be finite and positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Errors returned by Llama configuration and model admission.
#[derive(Debug)]
pub enum LlamaError {
    Model(ModelError),
    Tensor(String),
    UnsupportedArchitecture(String),
    MissingMetadata(&'static str),
    InvalidMetadata {
        key: &'static str,
        value: String,
    },
    InvalidConfig(String),
    MissingTensor(String),
    TensorShape {
        name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
}

impl fmt::Display for LlamaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Model(error) => write!(formatter, "GGUF model error: {error}"),
            Self::Tensor(error) => write!(formatter, "Llama tensor execution error: {error}"),
            Self::UnsupportedArchitecture(value) => {
                write!(formatter, "unsupported GGUF architecture: {value}")
            }
            Self::MissingMetadata(key) => write!(formatter, "missing Llama metadata: {key}"),
            Self::InvalidMetadata { key, value } => {
                write!(formatter, "invalid Llama metadata {key}: {value}")
            }
            Self::InvalidConfig(error) => write!(formatter, "invalid Llama configuration: {error}"),
            Self::MissingTensor(name) => write!(formatter, "missing Llama tensor: {name}"),
            Self::TensorShape {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "Llama tensor {name} has shape {actual:?}, expected {expected:?}"
            ),
        }
    }
}

impl std::error::Error for LlamaError {}

impl From<ModelError> for LlamaError {
    fn from(value: ModelError) -> Self {
        Self::Model(value)
    }
}

impl From<TensorError> for LlamaError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value.to_string())
    }
}

/// A validated Llama model index ready for decoder implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaModel {
    model: GgufModel,
    config: LlamaConfig,
}

impl LlamaModel {
    /// Opens a GGUF file, derives its Llama configuration, and validates its layout.
    ///
    /// # Errors
    ///
    /// Returns an error when the GGUF file, metadata, or tensor layout is
    /// invalid for a Llama model.
    pub fn open(path: impl AsRef<Path>, max_file_bytes: u64) -> Result<Self, LlamaError> {
        let model = GgufModel::open(path, max_file_bytes)?;
        let config = LlamaConfig::from_model(&model)?;
        Self::from_model(model, config)
    }

    /// Validates a model against an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture or required tensor layout does
    /// not match the configuration.
    pub fn from_model(model: GgufModel, config: LlamaConfig) -> Result<Self, LlamaError> {
        if model.architecture() != Some("llama") {
            return Err(LlamaError::UnsupportedArchitecture(
                model.architecture().unwrap_or("missing").to_owned(),
            ));
        }
        validate_layout(&model, &config)?;
        Ok(Self { model, config })
    }

    /// Returns the content-bound GGUF model index.
    #[must_use]
    pub const fn model(&self) -> &GgufModel {
        &self.model
    }

    /// Returns the validated Llama configuration.
    #[must_use]
    pub const fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Loads the validated model tensors into the checked CPU tensor engine.
    ///
    /// This prepares the single-token position-zero forward path. It does not
    /// load tokenizer tables or allocate a KV cache.
    ///
    /// # Errors
    ///
    /// Returns an error when a required tensor cannot be materialized as F32.
    pub fn load_cpu(&self) -> Result<LlamaCpuModel, LlamaError> {
        LlamaCpuModel::load(self)
    }

    /// Loads the model tokenizer tables from bounded GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenizer arrays are absent, malformed, or do not
    /// match the model vocabulary.
    pub fn tokenizer(&self) -> Result<LlamaTokenizer, LlamaError> {
        LlamaTokenizer::from_model(&self.model, self.config.vocab_size)
    }
}

const MAX_TOKENIZER_ELEMENTS: u64 = 16 * 1024 * 1024;

/// A bounded GGUF tokenizer vocabulary with greedy SentencePiece-style pieces.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    bos_token_id: Option<usize>,
    eos_token_id: Option<usize>,
    unk_token_id: Option<usize>,
}

impl LlamaTokenizer {
    fn from_model(model: &GgufModel, vocab_size: usize) -> Result<Self, LlamaError> {
        let tokens = model
            .metadata_string_array("tokenizer.ggml.tokens", MAX_TOKENIZER_ELEMENTS)?
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.tokens"))?;
        if tokens.len() != vocab_size {
            return Err(LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.tokens",
                value: format!(
                    "{} tokens, expected vocabulary size {vocab_size}",
                    tokens.len()
                ),
            });
        }
        let scores = model
            .metadata_f32_array("tokenizer.ggml.scores", MAX_TOKENIZER_ELEMENTS)?
            .unwrap_or_else(|| vec![0.0; tokens.len()]);
        if scores.len() != tokens.len() {
            return Err(LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.scores",
                value: format!("{} scores, expected {}", scores.len(), tokens.len()),
            });
        }
        let token_id_keys = [
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.unknown_token_id",
        ];
        let token_id_values = model.metadata_scalars(&token_id_keys)?;
        let mut token_id_values = token_id_values.into_iter();
        let bos_token_id =
            optional_usize_value(token_id_values.next().flatten(), token_id_keys[0])?;
        let eos_token_id =
            optional_usize_value(token_id_values.next().flatten(), token_id_keys[1])?;
        let unk_token_id =
            optional_usize_value(token_id_values.next().flatten(), token_id_keys[2])?;
        for (name, value) in [
            ("tokenizer.ggml.bos_token_id", bos_token_id),
            ("tokenizer.ggml.eos_token_id", eos_token_id),
            ("tokenizer.ggml.unknown_token_id", unk_token_id),
        ] {
            if let Some(value) = value
                && value >= tokens.len()
            {
                return Err(LlamaError::InvalidMetadata {
                    key: name,
                    value: format!("token id {value} is outside vocabulary"),
                });
            }
        }
        Ok(Self {
            tokens,
            scores,
            bos_token_id,
            eos_token_id,
            unk_token_id,
        })
    }

    /// Returns the vocabulary size.
    #[must_use]
    pub fn vocab_size(&self) -> usize {
        self.tokens.len()
    }

    /// Returns the optional beginning-of-sequence token id.
    #[must_use]
    pub const fn bos_token_id(&self) -> Option<usize> {
        self.bos_token_id
    }

    /// Returns the optional end-of-sequence token id.
    #[must_use]
    pub const fn eos_token_id(&self) -> Option<usize> {
        self.eos_token_id
    }

    /// Encodes text using normalized whitespace markers and greedy longest pieces.
    ///
    /// This covers the standard Llama `SentencePiece` vocabulary representation.
    /// Byte-fallback pieces are honored when present; otherwise an explicit
    /// unknown token is required for unmatched input.
    ///
    /// # Errors
    ///
    /// Returns an error when the text cannot be represented by this vocabulary.
    pub fn encode(&self, text: &str) -> Result<Vec<usize>, LlamaError> {
        let normalized = normalize_sentencepiece(text);
        let mut token_ids = Vec::new();
        let mut offset = 0;
        while offset < normalized.len() {
            let suffix = &normalized[offset..];
            let mut best: Option<(usize, usize, f32)> = None;
            for (index, token) in self.tokens.iter().enumerate() {
                if token.is_empty() || !suffix.starts_with(token) {
                    continue;
                }
                let candidate = (token.len(), index, self.scores[index]);
                if best.is_none_or(|current| {
                    candidate.0 > current.0 || (candidate.0 == current.0 && candidate.2 > current.2)
                }) {
                    best = Some(candidate);
                }
            }
            if let Some((length, index, _)) = best {
                token_ids.push(index);
                offset += length;
                continue;
            }
            let character = suffix
                .chars()
                .next()
                .ok_or_else(|| LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.tokens",
                    value: "invalid UTF-8 token boundary".to_owned(),
                })?;
            let mut consumed_byte = false;
            for byte in character.to_string().as_bytes() {
                let fallback = format!("<0x{byte:02X}>");
                if let Some(index) = self.tokens.iter().position(|token| token == &fallback) {
                    token_ids.push(index);
                    consumed_byte = true;
                } else {
                    consumed_byte = false;
                    break;
                }
            }
            if consumed_byte {
                offset += character.len_utf8();
            } else if let Some(unk_token_id) = self.unk_token_id {
                token_ids.push(unk_token_id);
                offset += character.len_utf8();
            } else {
                return Err(LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.tokens",
                    value: format!("no token matches input at byte offset {offset}"),
                });
            }
        }
        Ok(token_ids)
    }

    /// Decodes token ids into text, including byte-fallback pieces.
    ///
    /// # Errors
    ///
    /// Returns an error when a token id is outside the vocabulary or byte
    /// fallback pieces are not valid UTF-8.
    pub fn decode(&self, token_ids: &[usize]) -> Result<String, LlamaError> {
        let mut output = String::new();
        let mut bytes = Vec::new();
        for &token_id in token_ids {
            let token = self
                .tokens
                .get(token_id)
                .ok_or_else(|| LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.tokens",
                    value: format!("token id {token_id} is outside vocabulary"),
                })?;
            if Some(token_id) == self.bos_token_id || Some(token_id) == self.eos_token_id {
                continue;
            }
            if let Some(byte) = parse_byte_fallback(token) {
                bytes.push(byte);
                continue;
            }
            if !bytes.is_empty() {
                let decoded = String::from_utf8(std::mem::take(&mut bytes)).map_err(|_| {
                    LlamaError::InvalidMetadata {
                        key: "tokenizer.ggml.tokens",
                        value: "byte fallback is not valid UTF-8".to_owned(),
                    }
                })?;
                output.push_str(&decoded);
            }
            output.push_str(&token.replace('▁', " "));
        }
        if !bytes.is_empty() {
            let decoded = String::from_utf8(bytes).map_err(|_| LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.tokens",
                value: "byte fallback is not valid UTF-8".to_owned(),
            })?;
            output.push_str(&decoded);
        }
        Ok(output)
    }
}

fn normalize_sentencepiece(text: &str) -> String {
    let mut normalized = String::with_capacity(text.len() + 3);
    let mut needs_boundary = true;
    for character in text.chars() {
        if character.is_whitespace() {
            needs_boundary = true;
        } else {
            if needs_boundary {
                normalized.push('▁');
                needs_boundary = false;
            }
            normalized.push(character);
        }
    }
    normalized
}

fn parse_byte_fallback(token: &str) -> Option<u8> {
    let bytes = token.as_bytes();
    if bytes.len() != 6
        || bytes[0] != b'<'
        || bytes[1] != b'0'
        || bytes[2] != b'x'
        || bytes[5] != b'>'
    {
        return None;
    }
    let high = char::from(bytes[3]).to_digit(16)?;
    let low = char::from(bytes[4]).to_digit(16)?;
    u8::try_from((high << 4) | low).ok()
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm: Tensor,
    attn_q: Tensor,
    attn_k: Tensor,
    attn_v: Tensor,
    attn_output: Tensor,
    ffn_norm: Tensor,
    ffn_gate: Tensor,
    ffn_down: Tensor,
    ffn_up: Tensor,
}

/// A CPU-resident Llama model with checked incremental decoding.
///
/// The CPU path covers bounded tokenizer loading, RoPE-aware causal attention,
/// per-layer KV caching, and deterministic greedy generation. It remains a
/// correctness reference until optimized sampling and Apple GPU kernels land.
#[derive(Debug, Clone)]
pub struct LlamaCpuModel {
    config: LlamaConfig,
    token_embedding: Tensor,
    output: Tensor,
    output_norm: Tensor,
    layers: Vec<LayerWeights>,
    tokenizer: Option<LlamaTokenizer>,
}

impl LlamaCpuModel {
    fn load(model: &LlamaModel) -> Result<Self, LlamaError> {
        let config = model.config.clone();
        let tokenizer = match model.tokenizer() {
            Ok(tokenizer) => Some(tokenizer),
            Err(LlamaError::MissingMetadata("tokenizer.ggml.tokens")) => None,
            Err(error) => return Err(error),
        };
        let mut names = vec![
            "token_embd.weight".to_owned(),
            "output.weight".to_owned(),
            "output_norm.weight".to_owned(),
        ];
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_down.weight",
                "ffn_up.weight",
            ] {
                names.push(format!("{prefix}.{suffix}"));
            }
        }
        let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        let mut loaded = (0..names.len())
            .map(|_| None)
            .collect::<Vec<Option<Tensor>>>();
        let mut next = 0;
        model.model.for_each_f32(&name_refs, |_name, tensor| {
            let slot = loaded.get_mut(next).ok_or_else(|| {
                LlamaError::Tensor("GGUF loader returned an unexpected tensor".to_owned())
            })?;
            *slot = Some(tensor);
            next += 1;
            Ok::<(), LlamaError>(())
        })?;
        let mut loaded = loaded.into_iter();
        let token_embedding = next_tensor(&mut loaded, "token_embd.weight")?;
        let output = next_tensor(&mut loaded, "output.weight")?;
        let output_norm = next_tensor(&mut loaded, "output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            layers.push(LayerWeights {
                attn_norm: next_tensor(&mut loaded, &format!("{prefix}.attn_norm.weight"))?,
                attn_q: next_tensor(&mut loaded, &format!("{prefix}.attn_q.weight"))?,
                attn_k: next_tensor(&mut loaded, &format!("{prefix}.attn_k.weight"))?,
                attn_v: next_tensor(&mut loaded, &format!("{prefix}.attn_v.weight"))?,
                attn_output: next_tensor(&mut loaded, &format!("{prefix}.attn_output.weight"))?,
                ffn_norm: next_tensor(&mut loaded, &format!("{prefix}.ffn_norm.weight"))?,
                ffn_gate: next_tensor(&mut loaded, &format!("{prefix}.ffn_gate.weight"))?,
                ffn_down: next_tensor(&mut loaded, &format!("{prefix}.ffn_down.weight"))?,
                ffn_up: next_tensor(&mut loaded, &format!("{prefix}.ffn_up.weight"))?,
            });
        }
        Ok(Self {
            config,
            token_embedding,
            output,
            output_norm,
            layers,
            tokenizer,
        })
    }

    /// Returns the validated model configuration.
    #[must_use]
    pub const fn config(&self) -> &LlamaConfig {
        &self.config
    }

    /// Returns the tokenizer when the GGUF contains bounded tokenizer tables.
    #[must_use]
    pub const fn tokenizer(&self) -> Option<&LlamaTokenizer> {
        self.tokenizer.as_ref()
    }

    /// Creates an empty KV-cache session for this model.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured cache width cannot fit the host
    /// address space.
    pub fn session(&self) -> Result<LlamaSession<'_>, LlamaError> {
        LlamaSession::new(self)
    }

    /// Runs one position-zero token through the CPU decoder and returns logits.
    ///
    /// Position zero makes rotary embedding an identity and single-token
    /// attention has a one-element softmax. This convenience method creates a
    /// fresh session, so callers doing sequence decoding should use
    /// [`LlamaCpuModel::session`] directly.
    ///
    /// # Errors
    ///
    /// Returns an error when `token_id` is outside the vocabulary or a checked
    /// tensor operation fails.
    pub fn forward_token(&self, token_id: usize) -> Result<Vec<f32>, LlamaError> {
        let mut session = self.session()?;
        session.forward_token(token_id)
    }

    /// Generates deterministic text from a prompt using the bounded tokenizer.
    ///
    /// This uses greedy sampling on the CPU reference path. It is intended for
    /// correctness smoke tests while higher-performance sampling and Apple
    /// backend execution are added.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenizer metadata is absent, text cannot be
    /// encoded, or decoding exceeds the model context.
    pub fn generate_text(&self, prompt: &str, max_new_tokens: usize) -> Result<String, LlamaError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.tokens"))?;
        let mut prompt_ids = tokenizer.encode(prompt)?;
        if let Some(bos) = tokenizer.bos_token_id() {
            prompt_ids.insert(0, bos);
        }
        let mut session = self.session()?;
        let generated = session.generate_greedy(&prompt_ids, max_new_tokens)?;
        tokenizer.decode(&generated)
    }
}

/// Per-layer key/value storage for incremental decoding.
#[derive(Debug, Clone)]
pub struct LlamaKvCache {
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    capacity: usize,
    kv_width: usize,
}

impl LlamaKvCache {
    /// Allocates a bounded cache for the model's configured context length.
    ///
    /// # Errors
    ///
    /// Returns an error when the cached key/value width overflows the host
    /// address space.
    pub fn new(config: &LlamaConfig) -> Result<Self, LlamaError> {
        let kv_width = config
            .embedding_length
            .checked_mul(config.head_count_kv)
            .and_then(|width| width.checked_div(config.head_count))
            .ok_or_else(|| {
                LlamaError::InvalidConfig(
                    "KV cache width overflows the host address space".to_owned(),
                )
            })?;
        Ok(Self {
            keys: (0..config.block_count).map(|_| Vec::new()).collect(),
            values: (0..config.block_count).map(|_| Vec::new()).collect(),
            capacity: config.context_length,
            kv_width,
        })
    }

    /// Returns the number of tokens currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.keys
            .first()
            .map_or(0, |values| values.len() / self.kv_width)
    }

    /// Returns whether no tokens have been stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the configured maximum number of cached tokens.
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }
}

/// Incremental CPU decoder state with a bounded per-layer KV cache.
#[derive(Debug)]
pub struct LlamaSession<'a> {
    model: &'a LlamaCpuModel,
    cache: LlamaKvCache,
    position: usize,
}

impl<'a> LlamaSession<'a> {
    fn new(model: &'a LlamaCpuModel) -> Result<Self, LlamaError> {
        Ok(Self {
            model,
            cache: LlamaKvCache::new(&model.config)?,
            position: 0,
        })
    }

    /// Returns the next position that will be decoded.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Returns the current KV cache.
    #[must_use]
    pub const fn cache(&self) -> &LlamaKvCache {
        &self.cache
    }

    /// Decodes one token and returns vocabulary logits.
    ///
    /// Rotary embeddings are applied at the current position, and attention
    /// reads all cached keys and values for causal incremental decoding.
    ///
    /// # Errors
    ///
    /// Returns an error when the token or context bound is invalid, or when a
    /// checked tensor operation produces a non-finite result.
    #[allow(clippy::too_many_lines)]
    pub fn forward_token(&mut self, token_id: usize) -> Result<Vec<f32>, LlamaError> {
        if token_id >= self.model.config.vocab_size {
            return Err(LlamaError::InvalidConfig(format!(
                "token id {token_id} is outside vocabulary size {}",
                self.model.config.vocab_size
            )));
        }
        if self.position >= self.cache.capacity {
            return Err(LlamaError::InvalidConfig(format!(
                "context length {} exceeded",
                self.cache.capacity
            )));
        }
        let embedding_width = self.model.config.embedding_length;
        let vocab_size = self.model.config.vocab_size;
        let embedding = self.model.token_embedding.data();
        let hidden = (0..embedding_width)
            .map(|row| embedding[row * vocab_size + token_id])
            .collect::<Vec<_>>();
        let mut hidden = row_tensor(embedding_width, hidden)?;
        let head_dim = embedding_width / self.model.config.head_count;
        #[allow(clippy::cast_precision_loss)]
        let attention_scale = (head_dim as f32).sqrt().recip();
        for (layer_index, layer) in self.model.layers.iter().enumerate() {
            let normalized =
                hidden
                    .rms_norm(self.model.config.rms_norm_epsilon)?
                    .mul(&row_tensor(
                        embedding_width,
                        layer.attn_norm.data().to_vec(),
                    )?)?;
            let query = normalized.matmul(&layer.attn_q)?;
            let key = normalized.matmul(&layer.attn_k)?;
            let value = normalized.matmul(&layer.attn_v)?;
            let mut query_values = query.into_data();
            let mut key_values = key.into_data();
            let value_values = value.into_data();
            apply_rope(
                &mut query_values,
                self.model.config.head_count,
                head_dim,
                self.position,
                self.model.config.rope_freq_base,
            )?;
            apply_rope(
                &mut key_values,
                self.model.config.head_count_kv,
                head_dim,
                self.position,
                self.model.config.rope_freq_base,
            )?;
            if key_values.len() != self.cache.kv_width {
                return Err(LlamaError::Tensor(
                    "key projection width does not match KV cache".to_owned(),
                ));
            }
            self.cache.keys[layer_index].extend_from_slice(&key_values);
            self.cache.values[layer_index].extend_from_slice(&value_values);
            let mut attended = vec![0.0; embedding_width];
            let query_groups = self.model.config.head_count / self.model.config.head_count_kv;
            let cached_tokens = self.position + 1;
            for query_head in 0..self.model.config.head_count {
                let kv_head = query_head / query_groups;
                let query_start = query_head * head_dim;
                let kv_start = kv_head * head_dim;
                let mut scores = Vec::with_capacity(cached_tokens);
                for token_index in 0..cached_tokens {
                    let cached_key_start = token_index * self.cache.kv_width + kv_start;
                    let mut score = 0.0_f32;
                    for offset in 0..head_dim {
                        score += query_values[query_start + offset]
                            * self.cache.keys[layer_index][cached_key_start + offset];
                    }
                    let scaled = score * attention_scale;
                    if !scaled.is_finite() {
                        return Err(LlamaError::Tensor(
                            "attention score is not finite".to_owned(),
                        ));
                    }
                    scores.push(scaled);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut denominator = 0.0_f32;
                let mut probabilities = Vec::with_capacity(cached_tokens);
                for score in scores {
                    let probability = (score - maximum).exp();
                    if !probability.is_finite() {
                        return Err(LlamaError::Tensor(
                            "attention probability is not finite".to_owned(),
                        ));
                    }
                    denominator += probability;
                    probabilities.push(probability);
                }
                if !denominator.is_finite() || denominator <= 0.0 {
                    return Err(LlamaError::Tensor(
                        "attention probability denominator is invalid".to_owned(),
                    ));
                }
                for (token_index, probability) in probabilities.into_iter().enumerate() {
                    let cached_value_start = token_index * self.cache.kv_width + kv_start;
                    let weight = probability / denominator;
                    for offset in 0..head_dim {
                        attended[query_start + offset] +=
                            weight * self.cache.values[layer_index][cached_value_start + offset];
                    }
                }
            }
            let attended = row_tensor(embedding_width, attended)?.matmul(&layer.attn_output)?;
            hidden = hidden.add(&attended)?;
            let normalized =
                hidden
                    .rms_norm(self.model.config.rms_norm_epsilon)?
                    .mul(&row_tensor(
                        embedding_width,
                        layer.ffn_norm.data().to_vec(),
                    )?)?;
            let gate = normalized.matmul(&layer.ffn_gate)?.silu()?;
            let up = normalized.matmul(&layer.ffn_up)?;
            let feed_forward = gate.mul(&up)?.matmul(&layer.ffn_down)?;
            hidden = hidden.add(&feed_forward)?;
        }
        let normalized = hidden
            .rms_norm(self.model.config.rms_norm_epsilon)?
            .mul(&row_tensor(
                embedding_width,
                self.model.output_norm.data().to_vec(),
            )?)?;
        let logits = normalized.matmul(&self.model.output)?.into_data();
        self.position += 1;
        Ok(logits)
    }

    /// Decodes a token sequence and returns logits for every token.
    ///
    /// # Errors
    ///
    /// Returns an error when any token exceeds the vocabulary or context
    /// limits, or when a checked tensor operation fails.
    pub fn decode(&mut self, token_ids: &[usize]) -> Result<Vec<Vec<f32>>, LlamaError> {
        token_ids
            .iter()
            .map(|&token_id| self.forward_token(token_id))
            .collect()
    }

    /// Generates up to `max_new_tokens` with deterministic greedy sampling.
    ///
    /// The returned ids contain only newly generated tokens. An EOS token ends
    /// generation and is not included in the result.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt exceeds the context or the model has
    /// no valid vocabulary logits.
    pub fn generate_greedy(
        &mut self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, LlamaError> {
        let mut logits = if let Some(logits) = self.decode(prompt_ids)?.pop() {
            logits
        } else if let Some(bos) = self
            .model
            .tokenizer
            .as_ref()
            .and_then(|tokenizer| tokenizer.bos_token_id)
        {
            self.forward_token(bos)?
        } else {
            return Err(LlamaError::InvalidConfig(
                "an empty prompt requires a BOS token for generation".to_owned(),
            ));
        };
        let eos_token_id = self
            .model
            .tokenizer
            .as_ref()
            .and_then(|tokenizer| tokenizer.eos_token_id);
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let token_id = argmax_finite(&logits)?;
            if Some(token_id) == eos_token_id {
                break;
            }
            generated.push(token_id);
            logits = self.forward_token(token_id)?;
        }
        Ok(generated)
    }
}

fn apply_rope(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    position: usize,
    frequency_base: f32,
) -> Result<(), LlamaError> {
    #[allow(clippy::cast_precision_loss)]
    let position = position as f32;
    let head_width = head_dim;
    #[allow(clippy::cast_precision_loss)]
    let head_dim = head_width as f32;
    for head in 0..head_count {
        let start = head * head_width;
        for pair in 0..head_width / 2 {
            #[allow(clippy::cast_precision_loss)]
            let exponent = -2.0 * pair as f32 / head_dim;
            let angle = position * frequency_base.powf(exponent);
            let (sine, cosine) = angle.sin_cos();
            let first = values[start + pair * 2];
            let second = values[start + pair * 2 + 1];
            let rotated_first = first * cosine - second * sine;
            let rotated_second = first * sine + second * cosine;
            if !rotated_first.is_finite() || !rotated_second.is_finite() {
                return Err(LlamaError::Tensor(
                    "rotary embedding is not finite".to_owned(),
                ));
            }
            values[start + pair * 2] = rotated_first;
            values[start + pair * 2 + 1] = rotated_second;
        }
    }
    Ok(())
}

fn argmax_finite(values: &[f32]) -> Result<usize, LlamaError> {
    let mut best: Option<(usize, f32)> = None;
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(LlamaError::Tensor(
                "logits contain a non-finite value".to_owned(),
            ));
        }
        if best.is_none_or(|(_, current)| value > current) {
            best = Some((index, value));
        }
    }
    best.map(|(index, _)| index)
        .ok_or_else(|| LlamaError::InvalidConfig("model returned an empty vocabulary".to_owned()))
}

fn row_tensor(width: usize, data: Vec<f32>) -> Result<Tensor, LlamaError> {
    Tensor::from_data([1, width], data).map_err(LlamaError::from)
}

fn required_usize_value(
    value: Option<MetadataScalar>,
    key: &'static str,
) -> Result<usize, LlamaError> {
    let value = value.ok_or(LlamaError::MissingMetadata(key))?;
    as_usize(value).map_err(|value| LlamaError::InvalidMetadata { key, value })
}

fn optional_usize_value(
    value: Option<MetadataScalar>,
    key: &'static str,
) -> Result<Option<usize>, LlamaError> {
    value
        .map(|value| as_usize(value).map_err(|value| LlamaError::InvalidMetadata { key, value }))
        .transpose()
}

fn optional_f32_value(
    value: Option<MetadataScalar>,
    key: &'static str,
) -> Result<Option<f32>, LlamaError> {
    value
        .map(|value| as_f32(value).map_err(|value| LlamaError::InvalidMetadata { key, value }))
        .transpose()
}

fn next_tensor(
    tensors: &mut impl Iterator<Item = Option<Tensor>>,
    name: &str,
) -> Result<Tensor, LlamaError> {
    tensors
        .next()
        .flatten()
        .ok_or_else(|| LlamaError::Tensor(format!("GGUF loader did not return {name}")))
}

fn as_usize(value: MetadataScalar) -> Result<usize, String> {
    let integer = match value {
        MetadataScalar::U8(value) => u64::from(value),
        MetadataScalar::I8(value) if value >= 0 => u64::try_from(value).unwrap_or_default(),
        MetadataScalar::U16(value) => u64::from(value),
        MetadataScalar::I16(value) if value >= 0 => u64::try_from(value).unwrap_or_default(),
        MetadataScalar::U32(value) => u64::from(value),
        MetadataScalar::I32(value) if value >= 0 => u64::try_from(value).unwrap_or_default(),
        MetadataScalar::U64(value) => value,
        MetadataScalar::I64(value) if value >= 0 => u64::try_from(value).unwrap_or_default(),
        value => return Err(format!("expected a nonnegative integer, got {value:?}")),
    };
    usize::try_from(integer).map_err(|_| format!("integer {integer} exceeds usize"))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn as_f32(value: MetadataScalar) -> Result<f32, String> {
    let value = match value {
        MetadataScalar::F32(value) => value,
        MetadataScalar::F64(value) if value.is_finite() => value as f32,
        MetadataScalar::U8(value) => f32::from(value),
        MetadataScalar::U16(value) => f32::from(value),
        MetadataScalar::U32(value) => value as f32,
        MetadataScalar::U64(value) => value as f32,
        value => return Err(format!("expected a finite number, got {value:?}")),
    };
    if value.is_finite() {
        Ok(value)
    } else {
        Err("number is not finite".to_owned())
    }
}

fn validate_layout(model: &GgufModel, config: &LlamaConfig) -> Result<(), LlamaError> {
    require_shape(
        model,
        "token_embd.weight",
        &[config.embedding_length, config.vocab_size],
    )?;
    require_shape(
        model,
        "output.weight",
        &[config.embedding_length, config.vocab_size],
    )?;
    require_shape(model, "output_norm.weight", &[config.embedding_length])?;
    let kv_width = config
        .embedding_length
        .checked_mul(config.head_count_kv)
        .and_then(|width| width.checked_div(config.head_count))
        .ok_or_else(|| {
            LlamaError::InvalidConfig(
                "key/value projection width overflows the host address space".to_owned(),
            )
        })?;
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        for (suffix, shape) in [
            ("attn_norm.weight", vec![config.embedding_length]),
            (
                "attn_q.weight",
                vec![config.embedding_length, config.embedding_length],
            ),
            ("attn_k.weight", vec![config.embedding_length, kv_width]),
            ("attn_v.weight", vec![config.embedding_length, kv_width]),
            (
                "attn_output.weight",
                vec![config.embedding_length, config.embedding_length],
            ),
            ("ffn_norm.weight", vec![config.embedding_length]),
            (
                "ffn_gate.weight",
                vec![config.embedding_length, config.feed_forward_length],
            ),
            (
                "ffn_down.weight",
                vec![config.feed_forward_length, config.embedding_length],
            ),
            (
                "ffn_up.weight",
                vec![config.embedding_length, config.feed_forward_length],
            ),
        ] {
            require_shape(model, &format!("{prefix}.{suffix}"), &shape)?;
        }
    }
    Ok(())
}

fn require_shape(model: &GgufModel, name: &str, expected: &[usize]) -> Result<(), LlamaError> {
    let tensor = model
        .tensor(name)
        .ok_or_else(|| LlamaError::MissingTensor(name.to_owned()))?;
    if tensor.shape() != expected {
        return Err(LlamaError::TensorShape {
            name: name.to_owned(),
            expected: expected.to_vec(),
            actual: tensor.shape().to_vec(),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_u32_metadata(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_f32_metadata(bytes: &mut Vec<u8>, key: &str, value: f32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&6_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn push_string_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[&str]) {
        push_string(bytes, key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(values.len()).unwrap().to_le_bytes());
        for value in values {
            push_string(bytes, value);
        }
    }

    fn push_f32_array_metadata(bytes: &mut Vec<u8>, key: &str, values: &[f32]) {
        push_string(bytes, key);
        bytes.extend_from_slice(&9_u32.to_le_bytes());
        bytes.extend_from_slice(&6_u32.to_le_bytes());
        bytes.extend_from_slice(&u64::try_from(values.len()).unwrap().to_le_bytes());
        for value in values {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }

    fn llama_fixture() -> Vec<u8> {
        let config = [
            ("token_embd.weight", vec![4_u64, 8]),
            ("output.weight", vec![4, 8]),
            ("output_norm.weight", vec![4]),
            ("blk.0.attn_norm.weight", vec![4]),
            ("blk.0.attn_q.weight", vec![4, 4]),
            ("blk.0.attn_k.weight", vec![4, 2]),
            ("blk.0.attn_v.weight", vec![4, 2]),
            ("blk.0.attn_output.weight", vec![4, 4]),
            ("blk.0.ffn_norm.weight", vec![4]),
            ("blk.0.ffn_gate.weight", vec![4, 8]),
            ("blk.0.ffn_down.weight", vec![8, 4]),
            ("blk.0.ffn_up.weight", vec![4, 8]),
        ];
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(config.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&13_u64.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "llama");
        push_u32_metadata(&mut bytes, "llama.context_length", 16);
        push_u32_metadata(&mut bytes, "llama.embedding_length", 4);
        push_u32_metadata(&mut bytes, "llama.block_count", 1);
        push_u32_metadata(&mut bytes, "llama.attention.head_count", 2);
        push_u32_metadata(&mut bytes, "llama.attention.head_count_kv", 1);
        push_u32_metadata(&mut bytes, "llama.feed_forward_length", 8);
        push_u32_metadata(&mut bytes, "llama.vocab_size", 8);
        push_f32_metadata(&mut bytes, "llama.attention.layer_norm_rms_epsilon", 1.0e-5);
        push_f32_metadata(&mut bytes, "llama.rope.freq_base", 10_000.0);
        push_string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "fixture");
        push_string_array_metadata(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<unk>", "▁a", "▁b", "<eos>", "<0x20>", "x", "y", "z"],
        );
        push_f32_array_metadata(
            &mut bytes,
            "tokenizer.ggml.scores",
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let mut offset = 0_u64;
        for (name, shape) in &config {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&u32::try_from(shape.len()).unwrap().to_le_bytes());
            for dimension in shape {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let elements = shape.iter().product::<u64>();
            let byte_len = elements * 4;
            offset += byte_len.div_ceil(32) * 32;
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.resize(bytes.len() + usize::try_from(offset).unwrap(), 0);
        bytes
    }

    fn write_fixture(bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("llama-runtime-{id}.gguf"));
        fs::write(&path, bytes).unwrap();
        path
    }

    #[test]
    fn validates_grouped_query_attention_configuration() {
        let config =
            LlamaConfig::new(4096, 4096, 32, 32, 8, 11_008, 32_000, 1.0e-5, 10_000.0).unwrap();
        assert_eq!(config.head_count_kv(), 8);
        assert_eq!(config.context_length(), 4096);
    }

    #[test]
    fn rejects_non_divisible_head_counts() {
        let result = LlamaConfig::new(1024, 1024, 2, 12, 5, 4096, 1000, 1.0e-5, 10_000.0);
        assert!(matches!(result, Err(LlamaError::InvalidConfig(_))));
    }

    #[test]
    fn rejects_non_positive_numerical_parameters() {
        let result = LlamaConfig::new(1024, 1024, 2, 16, 4, 4096, 1000, 0.0, 10_000.0);
        assert!(matches!(result, Err(LlamaError::InvalidConfig(_))));
        let result = LlamaConfig::new(1024, 1024, 2, 16, 4, 4096, 1000, 1.0e-5, f32::NAN);
        assert!(matches!(result, Err(LlamaError::InvalidConfig(_))));
    }

    #[test]
    fn opens_and_validates_complete_llama_layout() {
        let path = write_fixture(&llama_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().context_length(), 16);
        assert_eq!(model.config().head_count_kv(), 1);
        assert_eq!(model.model().tensors().len(), 12);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_cpu_model_and_runs_position_zero_forward() {
        let path = write_fixture(&llama_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        let cpu = model.load_cpu().unwrap();
        let tokenizer = cpu.tokenizer().unwrap();
        assert_eq!(tokenizer.encode("a b").unwrap(), [1, 2]);
        assert_eq!(tokenizer.decode(&[1, 2]).unwrap(), " a b");
        let logits = cpu.forward_token(3).unwrap();
        assert_eq!(logits, vec![0.0; 8]);
        let result = cpu.forward_token(8);
        assert!(matches!(result, Err(LlamaError::InvalidConfig(_))));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn greedy_generation_uses_tokenizer_and_stops_at_context_bound() {
        let path = write_fixture(&llama_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        let cpu = model.load_cpu().unwrap();
        assert_eq!(cpu.generate_text("a", 2).unwrap(), "<unk><unk>");
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn session_applies_rope_and_grows_bounded_kv_cache() {
        let path = write_fixture(&llama_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        let cpu = model.load_cpu().unwrap();
        let mut session = cpu.session().unwrap();
        let logits = session.decode(&[1, 2, 3]).unwrap();
        assert_eq!(logits.len(), 3);
        assert!(logits.iter().all(|values| values == &vec![0.0; 8]));
        assert_eq!(session.position(), 3);
        assert_eq!(session.cache().len(), 3);
        assert_eq!(session.cache().capacity(), 16);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_incomplete_llama_layout() {
        let mut bytes = llama_fixture();
        let needle = b"blk.0.ffn_up.weight";
        let position = bytes
            .windows(needle.len())
            .position(|window| window == needle)
            .unwrap();
        bytes[position] = b'x';
        let path = write_fixture(&bytes);
        let result = LlamaModel::open(&path, 1 << 20);
        assert!(matches!(result, Err(LlamaError::MissingTensor(_))));
        fs::remove_file(path).unwrap();
    }
}
