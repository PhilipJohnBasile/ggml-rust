#![deny(unsafe_code)]

use std::fmt;
use std::path::Path;

use ggml_model::{GgufModel, MetadataScalar, ModelError, QuantizedMatrix};
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

/// Sampling parameters for incremental Llama generation.
///
/// A temperature of zero selects the highest finite logit and preserves the
/// deterministic greedy behavior. A positive temperature enables sampling,
/// optionally restricted by `top_k` and nucleus `top_p`. A zero `top_k` means
/// that no top-k limit is applied. The seed is deterministic, including when
/// it is zero.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaSamplingConfig {
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
}

impl Default for LlamaSamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
        }
    }
}

impl LlamaSamplingConfig {
    /// Creates validated sampling parameters.
    ///
    /// # Errors
    ///
    /// Returns an error when temperature or nucleus probability is invalid.
    pub fn new(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Result<Self, LlamaError> {
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(LlamaError::InvalidConfig(
                "sampling temperature must be finite and nonnegative".to_owned(),
            ));
        }
        if !top_p.is_finite() || top_p <= 0.0 || top_p > 1.0 {
            return Err(LlamaError::InvalidConfig(
                "sampling top_p must be finite and in (0, 1]".to_owned(),
            ));
        }
        Ok(Self {
            temperature,
            top_k,
            top_p,
            seed,
        })
    }

    /// Returns the temperature. Zero selects greedy decoding.
    #[must_use]
    pub const fn temperature(self) -> f32 {
        self.temperature
    }

    /// Returns the top-k limit. Zero means unlimited.
    #[must_use]
    pub const fn top_k(self) -> usize {
        self.top_k
    }

    /// Returns the nucleus probability limit.
    #[must_use]
    pub const fn top_p(self) -> f32 {
        self.top_p
    }

    /// Returns the deterministic sampling seed.
    #[must_use]
    pub const fn seed(self) -> u64 {
        self.seed
    }
}

/// Stateful sampler that applies one validated policy to a stream of logits.
#[derive(Debug, Clone, Copy)]
pub struct LlamaSampler {
    config: LlamaSamplingConfig,
    rng: DeterministicRng,
}

impl LlamaSampler {
    /// Creates a sampler with deterministic state initialized from `config`.
    #[must_use]
    pub fn new(config: LlamaSamplingConfig) -> Self {
        Self {
            rng: DeterministicRng::new(config.seed()),
            config,
        }
    }

    /// Returns the policy used by this sampler.
    #[must_use]
    pub const fn config(&self) -> LlamaSamplingConfig {
        self.config
    }

    /// Selects one token from finite vocabulary logits.
    ///
    /// # Errors
    ///
    /// Returns an error when logits are empty, non-finite, or cannot produce a
    /// valid probability distribution.
    pub fn sample(&mut self, logits: &[f32]) -> Result<usize, LlamaError> {
        sample_logits(logits, self.config, &mut self.rng)
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
        LlamaCpuModel::load(self, false)
    }

    /// Loads the model while retaining supported rank-2 quantized matrices in
    /// encoded form for direct CPU products and embedding column lookup.
    ///
    /// The default [`Self::load_cpu`] path expands weights to F32 for faster
    /// CPU execution. This explicit mode avoids a complete F32 expansion and
    /// is useful for memory-bounded callers and backend parity checks.
    ///
    /// # Errors
    ///
    /// Returns an error when a required tensor cannot be loaded or validated.
    pub fn load_cpu_quantized(&self) -> Result<LlamaCpuModel, LlamaError> {
        LlamaCpuModel::load(self, true)
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

/// A bounded GGUF tokenizer vocabulary with SentencePiece-style pieces.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    bos_token_id: Option<usize>,
    eos_token_id: Option<usize>,
    unk_token_id: Option<usize>,
}

#[derive(Debug, Clone)]
struct EncodingPath {
    score: f32,
    token_count: usize,
    previous: usize,
    token_id: usize,
    kind: u8,
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
        if let Some(index) = scores.iter().position(|score| !score.is_finite()) {
            return Err(LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.scores",
                value: format!("score at token {index} is not finite"),
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

    /// Encodes text using normalized whitespace markers and `SentencePiece`
    /// unigram Viterbi segmentation.
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
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let input = normalized.as_bytes();
        let mut character_ends = vec![None; input.len()];
        for (offset, character) in normalized.char_indices() {
            character_ends[offset] = Some(offset + character.len_utf8());
        }
        let paths = self.encoding_paths(input, &character_ends, false)?;
        let paths = if paths[input.len()].is_some() {
            paths
        } else {
            self.encoding_paths(input, &character_ends, true)?
        };
        let Some(_) = paths[input.len()] else {
            return Err(LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.tokens",
                value: "no token matches normalized input".to_owned(),
            });
        };
        let mut token_ids = Vec::new();
        let mut offset = input.len();
        while offset != 0 {
            let path = paths[offset]
                .as_ref()
                .ok_or_else(|| LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.tokens",
                    value: "tokenization path is incomplete".to_owned(),
                })?;
            token_ids.push(path.token_id);
            offset = path.previous;
        }
        token_ids.reverse();
        Ok(token_ids)
    }

    fn encoding_paths(
        &self,
        input: &[u8],
        character_ends: &[Option<usize>],
        allow_unknown: bool,
    ) -> Result<Vec<Option<EncodingPath>>, LlamaError> {
        let mut paths = vec![None; input.len() + 1];
        paths[0] = Some(EncodingPath {
            score: 0.0,
            token_count: 0,
            previous: 0,
            token_id: 0,
            kind: 0,
        });
        let token_bytes = self.tokens.iter().map(String::as_bytes).collect::<Vec<_>>();
        for offset in 0..input.len() {
            let Some(path) = paths[offset].clone() else {
                continue;
            };
            for (token_id, bytes) in token_bytes.iter().enumerate() {
                if !bytes.is_empty()
                    && offset.saturating_add(bytes.len()) <= input.len()
                    && input[offset..].starts_with(bytes)
                {
                    self.relax_encoding_path(
                        &mut paths,
                        offset,
                        offset + bytes.len(),
                        token_id,
                        0,
                        &path,
                    )?;
                }
                if parse_byte_fallback(&self.tokens[token_id]) == Some(input[offset]) {
                    self.relax_encoding_path(&mut paths, offset, offset + 1, token_id, 1, &path)?;
                }
            }
            if allow_unknown
                && let (Some(unk_token_id), Some(end)) = (self.unk_token_id, character_ends[offset])
            {
                self.relax_encoding_path(&mut paths, offset, end, unk_token_id, 2, &path)?;
            }
        }
        Ok(paths)
    }

    fn relax_encoding_path(
        &self,
        paths: &mut [Option<EncodingPath>],
        previous: usize,
        end: usize,
        token_id: usize,
        kind: u8,
        path: &EncodingPath,
    ) -> Result<(), LlamaError> {
        let score = path.score + self.scores[token_id];
        if !score.is_finite() {
            return Err(LlamaError::InvalidMetadata {
                key: "tokenizer.ggml.scores",
                value: format!("tokenization score overflowed at token {token_id}"),
            });
        }
        let candidate = EncodingPath {
            score,
            token_count: path.token_count.checked_add(1).ok_or_else(|| {
                LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.tokens",
                    value: "tokenization path is too long".to_owned(),
                }
            })?,
            previous,
            token_id,
            kind,
        };
        let replace = paths[end].as_ref().is_none_or(|current| {
            candidate.score.total_cmp(&current.score).is_gt()
                || (candidate.score.total_cmp(&current.score).is_eq()
                    && (candidate.token_count < current.token_count
                        || (candidate.token_count == current.token_count
                            && (candidate.kind < current.kind
                                || (candidate.kind == current.kind
                                    && candidate.token_id < current.token_id)))))
        });
        if replace {
            paths[end] = Some(candidate);
        }
        Ok(())
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
enum CpuMatrix {
    F32(Tensor),
    Quantized(QuantizedMatrix),
}

impl CpuMatrix {
    fn from_model(model: &GgufModel, name: &str, use_quantized: bool) -> Result<Self, LlamaError> {
        let descriptor = model
            .tensor(name)
            .ok_or_else(|| LlamaError::MissingTensor(name.to_owned()))?;
        if use_quantized
            && descriptor.shape().len() == 2
            && matches!(
                descriptor.value_type().raw(),
                2 | 3 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15
            )
        {
            return Ok(Self::Quantized(model.load_quantized(name)?));
        }
        Ok(Self::F32(transpose_ggml_matrix(model.load_f32(name)?)?))
    }

    fn matmul(&self, input: &[f32]) -> Result<Vec<f32>, LlamaError> {
        match self {
            Self::F32(tensor) => row_tensor(input.len(), input.to_vec())?
                .matmul(tensor)
                .map(Tensor::into_data)
                .map_err(LlamaError::from),
            Self::Quantized(matrix) => matrix.matmul_f32(input).map_err(LlamaError::from),
        }
    }

    fn matmul_tensor(&self, input: &Tensor) -> Result<Tensor, LlamaError> {
        let data = self.matmul(input.data())?;
        Tensor::from_data([1, self.shape()[1]], data).map_err(LlamaError::from)
    }

    fn column(&self, column: usize) -> Result<Vec<f32>, LlamaError> {
        match self {
            Self::F32(tensor) => {
                let shape = tensor.shape();
                if shape.len() != 2 || column >= shape[1] {
                    return Err(LlamaError::Tensor(format!(
                        "matrix column {column} is outside shape {shape:?}"
                    )));
                }
                Ok((0..shape[0])
                    .map(|row| tensor.data()[row * shape[1] + column])
                    .collect())
            }
            Self::Quantized(matrix) => matrix.column(column).map_err(LlamaError::from),
        }
    }

    fn shape(&self) -> [usize; 2] {
        match self {
            Self::F32(tensor) => [tensor.shape()[0], tensor.shape()[1]],
            Self::Quantized(matrix) => [matrix.rows(), matrix.columns()],
        }
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    attn_norm: Tensor,
    attn_q: CpuMatrix,
    attn_k: CpuMatrix,
    attn_v: CpuMatrix,
    attn_output: CpuMatrix,
    ffn_norm: Tensor,
    ffn_gate: CpuMatrix,
    ffn_down: CpuMatrix,
    ffn_up: CpuMatrix,
}

/// A CPU-resident Llama model with checked incremental decoding.
///
/// The CPU path covers bounded tokenizer loading, direct products for
/// supported quantized matrices, RoPE-aware causal attention, per-layer KV
/// caching, and deterministic seeded sampling. It remains the correctness
/// fallback while Apple GPU kernels and broader architecture coverage evolve.
#[derive(Debug, Clone)]
pub struct LlamaCpuModel {
    config: LlamaConfig,
    token_embedding: CpuMatrix,
    output: CpuMatrix,
    output_norm: Tensor,
    layers: Vec<LayerWeights>,
    tokenizer: Option<LlamaTokenizer>,
    use_quantized: bool,
}

impl LlamaCpuModel {
    fn load(model: &LlamaModel, use_quantized: bool) -> Result<Self, LlamaError> {
        let config = model.config.clone();
        let tokenizer = match model.tokenizer() {
            Ok(tokenizer) => Some(tokenizer),
            Err(LlamaError::MissingMetadata("tokenizer.ggml.tokens")) => None,
            Err(error) => return Err(error),
        };
        let token_embedding =
            CpuMatrix::from_model(&model.model, "token_embd.weight", use_quantized)?;
        let output = CpuMatrix::from_model(&model.model, "output.weight", use_quantized)?;
        let output_norm = model.model.load_f32("output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            layers.push(LayerWeights {
                attn_norm: model
                    .model
                    .load_f32(&format!("{prefix}.attn_norm.weight"))?,
                attn_q: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.attn_q.weight"),
                    use_quantized,
                )?,
                attn_k: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.attn_k.weight"),
                    use_quantized,
                )?,
                attn_v: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.attn_v.weight"),
                    use_quantized,
                )?,
                attn_output: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.attn_output.weight"),
                    use_quantized,
                )?,
                ffn_norm: model.model.load_f32(&format!("{prefix}.ffn_norm.weight"))?,
                ffn_gate: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.ffn_gate.weight"),
                    use_quantized,
                )?,
                ffn_down: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.ffn_down.weight"),
                    use_quantized,
                )?,
                ffn_up: CpuMatrix::from_model(
                    &model.model,
                    &format!("{prefix}.ffn_up.weight"),
                    use_quantized,
                )?,
            });
        }
        Ok(Self {
            config,
            token_embedding,
            output,
            output_norm,
            layers,
            tokenizer,
            use_quantized,
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

    /// Returns whether supported quantized matrices remain encoded in this
    /// CPU model rather than being expanded to F32.
    #[must_use]
    pub const fn uses_quantized_weights(&self) -> bool {
        self.use_quantized
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
        self.generate_text_with_sampling(prompt, max_new_tokens, LlamaSamplingConfig::default())
    }

    /// Generates text with validated temperature, top-k, top-p, and seed
    /// controls.
    ///
    /// # Errors
    ///
    /// Returns an error when tokenizer metadata is absent, text cannot be
    /// encoded, sampling parameters are invalid, or decoding exceeds the
    /// model context.
    pub fn generate_text_with_sampling(
        &self,
        prompt: &str,
        max_new_tokens: usize,
        sampling: LlamaSamplingConfig,
    ) -> Result<String, LlamaError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.tokens"))?;
        let mut prompt_ids = tokenizer.encode(prompt)?;
        if let Some(bos) = tokenizer.bos_token_id() {
            prompt_ids.insert(0, bos);
        }
        let mut session = self.session()?;
        let generated = session.generate_with_sampling(&prompt_ids, max_new_tokens, sampling)?;
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
        let hidden = self.model.token_embedding.column(token_id)?;
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
            let query = layer.attn_q.matmul_tensor(&normalized)?;
            let key = layer.attn_k.matmul_tensor(&normalized)?;
            let value = layer.attn_v.matmul_tensor(&normalized)?;
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
            let attended = layer
                .attn_output
                .matmul_tensor(&row_tensor(embedding_width, attended)?)?;
            hidden = hidden.add(&attended)?;
            let normalized =
                hidden
                    .rms_norm(self.model.config.rms_norm_epsilon)?
                    .mul(&row_tensor(
                        embedding_width,
                        layer.ffn_norm.data().to_vec(),
                    )?)?;
            let gate = layer.ffn_gate.matmul_tensor(&normalized)?.silu()?;
            let up = layer.ffn_up.matmul_tensor(&normalized)?;
            let feed_forward = layer.ffn_down.matmul_tensor(&gate.mul(&up)?)?;
            hidden = hidden.add(&feed_forward)?;
        }
        let normalized = hidden
            .rms_norm(self.model.config.rms_norm_epsilon)?
            .mul(&row_tensor(
                embedding_width,
                self.model.output_norm.data().to_vec(),
            )?)?;
        let logits = self.model.output.matmul_tensor(&normalized)?.into_data();
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
        self.generate_with_sampling(prompt_ids, max_new_tokens, LlamaSamplingConfig::default())
    }

    /// Generates up to `max_new_tokens` with validated sampling controls.
    ///
    /// The returned ids contain only newly generated tokens. An EOS token ends
    /// generation and is not included in the result. Sampling uses a small
    /// deterministic PRNG owned by this call, so the same logits,
    /// configuration, and seed produce the same token sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when the prompt exceeds the context, the model has no
    /// valid vocabulary logits, or sampling produces a non-finite value.
    pub fn generate_with_sampling(
        &mut self,
        prompt_ids: &[usize],
        max_new_tokens: usize,
        sampling: LlamaSamplingConfig,
    ) -> Result<Vec<usize>, LlamaError> {
        let mut sampler = LlamaSampler::new(sampling);
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
            let token_id = sampler.sample(&logits)?;
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

#[derive(Debug, Clone, Copy)]
struct DeterministicRng {
    state: u64,
}

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9e37_79b9_7f4a_7c15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut value = self.state;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.state = value;
        value
    }

    #[allow(clippy::cast_precision_loss)]
    fn next_unit_f32(&mut self) -> f32 {
        let mantissa = self.next_u64() >> 40;
        mantissa as f32 / 16_777_216.0
    }
}

fn sample_logits(
    values: &[f32],
    config: LlamaSamplingConfig,
    rng: &mut DeterministicRng,
) -> Result<usize, LlamaError> {
    if values.is_empty() {
        return Err(LlamaError::InvalidConfig(
            "model returned an empty vocabulary".to_owned(),
        ));
    }
    for value in values {
        if !value.is_finite() {
            return Err(LlamaError::Tensor(
                "logits contain a non-finite value".to_owned(),
            ));
        }
    }
    if config.temperature() == 0.0 {
        return argmax_finite(values);
    }
    let mut candidates = values
        .iter()
        .copied()
        .enumerate()
        .map(|(index, value)| {
            let scaled = value / config.temperature();
            (index, scaled)
        })
        .collect::<Vec<_>>();
    if candidates.iter().any(|(_, value)| !value.is_finite()) {
        return Err(LlamaError::Tensor(
            "temperature-scaled logits contain a non-finite value".to_owned(),
        ));
    }
    candidates.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    if config.top_k() != 0 {
        candidates.truncate(config.top_k());
    }
    let maximum = candidates[0].1;
    let mut weighted = candidates
        .into_iter()
        .map(|(index, value)| {
            let weight = (value - maximum).exp();
            (index, weight)
        })
        .collect::<Vec<_>>();
    let total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    if !total.is_finite() || total <= 0.0 {
        return Err(LlamaError::Tensor(
            "sampling probability total is invalid".to_owned(),
        ));
    }
    let mut cumulative = 0.0_f32;
    let mut keep = 0;
    for (_, weight) in &weighted {
        cumulative += *weight / total;
        keep += 1;
        if cumulative >= config.top_p() {
            break;
        }
    }
    weighted.truncate(keep.max(1));
    let selected_total = weighted.iter().map(|(_, weight)| *weight).sum::<f32>();
    if !selected_total.is_finite() || selected_total <= 0.0 {
        return Err(LlamaError::Tensor(
            "selected sampling probability total is invalid".to_owned(),
        ));
    }
    let target = rng.next_unit_f32() * selected_total;
    let mut accumulated = 0.0_f32;
    for (index, weight) in &weighted {
        accumulated += *weight;
        if target < accumulated {
            return Ok(*index);
        }
    }
    weighted
        .last()
        .map(|(index, _)| *index)
        .ok_or_else(|| LlamaError::InvalidConfig("sampling returned no candidate".to_owned()))
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

fn transpose_ggml_matrix(tensor: Tensor) -> Result<Tensor, LlamaError> {
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Ok(tensor);
    }
    let rows = shape[0];
    let columns = shape[1];
    let data = tensor.into_data();
    let mut transposed = vec![0.0; data.len()];
    for row in 0..rows {
        for column in 0..columns {
            transposed[row * columns + column] = data[column * rows + row];
        }
    }
    Tensor::from_data([rows, columns], transposed).map_err(LlamaError::from)
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
    fn validates_sampling_parameters() {
        assert!(LlamaSamplingConfig::new(-1.0, 0, 1.0, 1).is_err());
        assert!(LlamaSamplingConfig::new(f32::NAN, 0, 1.0, 1).is_err());
        assert!(LlamaSamplingConfig::new(1.0, 0, 0.0, 1).is_err());
        assert!(LlamaSamplingConfig::new(1.0, 0, 1.1, 1).is_err());
        let config = LlamaSamplingConfig::new(0.8, 12, 0.95, 42).unwrap();
        assert!((config.temperature() - 0.8).abs() < f32::EPSILON);
        assert_eq!(config.top_k(), 12);
        assert!((config.top_p() - 0.95).abs() < f32::EPSILON);
        assert_eq!(config.seed(), 42);
    }

    #[test]
    fn sampling_is_seeded_and_respects_top_k() {
        let config = LlamaSamplingConfig::new(1.0, 1, 1.0, 42).unwrap();
        let mut first_rng = DeterministicRng::new(config.seed());
        let mut second_rng = DeterministicRng::new(config.seed());
        assert_eq!(
            sample_logits(&[0.0, 1.0, 2.0, 3.0], config, &mut first_rng).unwrap(),
            3
        );
        assert_eq!(
            sample_logits(&[0.0, 1.0, 2.0, 3.0], config, &mut second_rng).unwrap(),
            3
        );
    }

    #[test]
    fn sampling_nucleus_keeps_the_highest_candidate() {
        let config = LlamaSamplingConfig::new(1.0, 0, 0.5, 7).unwrap();
        let mut rng = DeterministicRng::new(config.seed());
        assert_eq!(
            sample_logits(&[10.0, 0.0, 0.0], config, &mut rng).unwrap(),
            0
        );
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
    fn tokenizer_uses_unigram_scores_over_longest_piece() {
        let tokenizer = LlamaTokenizer {
            tokens: vec![
                "<unk>".to_owned(),
                "▁ab".to_owned(),
                "▁a".to_owned(),
                "b".to_owned(),
            ],
            scores: vec![0.0, -10.0, -1.0, -1.0],
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: Some(0),
        };
        assert_eq!(tokenizer.encode("ab").unwrap(), [2, 3]);
    }

    #[test]
    fn tokenizer_consumes_utf8_with_byte_fallback_pieces() {
        let tokenizer = LlamaTokenizer {
            tokens: vec!["▁".to_owned(), "<0xC3>".to_owned(), "<0xA9>".to_owned()],
            scores: vec![0.0, 0.0, 0.0],
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
        };
        assert_eq!(tokenizer.encode("é").unwrap(), [0, 1, 2]);
        assert_eq!(tokenizer.decode(&[0, 1, 2]).unwrap(), " é");
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
    fn normalizes_first_dimension_contiguous_ggml_matrices() {
        let tensor = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let normalized = transpose_ggml_matrix(tensor).unwrap();
        assert_eq!(normalized.shape(), &[2, 3]);
        assert_eq!(normalized.data(), &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0]);
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
