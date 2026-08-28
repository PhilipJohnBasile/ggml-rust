#![deny(unsafe_code)]

use std::fmt;
use std::path::Path;

use ggml_model::{GgufModel, MetadataScalar, ModelError};

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
        let context_length = required_usize(model, "llama.context_length")?;
        let embedding_length = required_usize(model, "llama.embedding_length")?;
        let block_count = required_usize(model, "llama.block_count")?;
        let head_count = required_usize(model, "llama.attention.head_count")?;
        let head_count_kv =
            optional_usize(model, "llama.attention.head_count_kv")?.unwrap_or(head_count);
        let feed_forward_length = required_usize(model, "llama.feed_forward_length")?;
        let vocab_size = required_usize(model, "llama.vocab_size")?;
        let rms_norm_epsilon =
            optional_f32(model, "llama.attention.layer_norm_rms_epsilon")?.unwrap_or(1.0e-5);
        let rope_freq_base = optional_f32(model, "llama.rope.freq_base")?.unwrap_or(10_000.0);
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
}

fn required_usize(model: &GgufModel, key: &'static str) -> Result<usize, LlamaError> {
    let value = model
        .metadata_scalar(key)?
        .ok_or(LlamaError::MissingMetadata(key))?;
    as_usize(value).map_err(|value| LlamaError::InvalidMetadata { key, value })
}

fn optional_usize(model: &GgufModel, key: &'static str) -> Result<Option<usize>, LlamaError> {
    model
        .metadata_scalar(key)?
        .map(|value| as_usize(value).map_err(|value| LlamaError::InvalidMetadata { key, value }))
        .transpose()
}

fn optional_f32(model: &GgufModel, key: &'static str) -> Result<Option<f32>, LlamaError> {
    model
        .metadata_scalar(key)?
        .map(|value| as_f32(value).map_err(|value| LlamaError::InvalidMetadata { key, value }))
        .transpose()
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
        bytes.extend_from_slice(&11_u64.to_le_bytes());
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
        let mut offset = 0_u64;
        for (name, shape) in &config {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&(shape.len() as u32).to_le_bytes());
            for dimension in shape {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&0_u32.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let elements = shape.iter().product::<u64>();
            let byte_len = elements * 4;
            offset += (byte_len + 31) / 32 * 32;
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.resize(bytes.len() + offset as usize, 0);
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
