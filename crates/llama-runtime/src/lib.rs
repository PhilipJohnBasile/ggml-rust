#![deny(unsafe_code)]

use std::collections::HashMap;
use std::fmt;
use std::path::Path;

use ggml_model::{GgufModel, GgufReadSession, MetadataScalar, ModelError, QuantizedMatrix};
use ggml_tensor::{Tensor, TensorError};

/// Position scaling applied before rotary phase calculation.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum LlamaRopeScaling {
    /// Use the unscaled token position.
    #[default]
    None,
    /// Divide token positions by the configured linear factor.
    Linear { factor: f32 },
    /// Apply `YaRN` frequency interpolation and magnitude scaling.
    Yarn {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_context_length: usize,
        attention_factor: f32,
        ext_factor: f32,
    },
}

impl LlamaRopeScaling {
    /// Returns the position divisor used by rotary embeddings.
    #[must_use]
    pub const fn factor(self) -> f32 {
        match self {
            Self::None => 1.0,
            Self::Linear { factor } | Self::Yarn { factor, .. } => factor,
        }
    }

    /// Returns the GGUF scaling kind.
    #[must_use]
    pub const fn kind(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Linear { .. } => "linear",
            Self::Yarn { .. } => "yarn",
        }
    }

    #[allow(clippy::cast_precision_loss)]
    fn phase(
        self,
        position: f32,
        pair: usize,
        head_dim: f32,
        rotary_dimension_count: usize,
        frequency_base: f32,
        frequency_factor: f32,
    ) -> (f32, f32) {
        #[allow(clippy::cast_precision_loss)]
        let exponent = -2.0 * pair as f32 / head_dim;
        let theta_extrap = position * frequency_base.powf(exponent) / frequency_factor;
        match self {
            Self::None => (theta_extrap, 1.0),
            Self::Linear { factor } => (theta_extrap / factor, 1.0),
            Self::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_length,
                attention_factor,
                ext_factor,
            } => {
                let freq_scale = 1.0 / factor;
                let theta_interp = freq_scale * theta_extrap;
                let low = rope_yarn_correction_dim(
                    beta_fast,
                    rotary_dimension_count,
                    frequency_base,
                    original_context_length,
                )
                .floor()
                .max(0.0);
                let high = rope_yarn_correction_dim(
                    beta_slow,
                    rotary_dimension_count,
                    frequency_base,
                    original_context_length,
                )
                .ceil()
                .min(rotary_dimension_count.saturating_sub(1) as f32);
                let ramp = rope_yarn_ramp(low, high, pair * 2) * ext_factor;
                let theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
                let magnitude = attention_factor * (1.0 + 0.1 * (1.0 / freq_scale).ln());
                (theta, magnitude)
            }
        }
    }

    fn validate(self) -> Result<(), LlamaError> {
        if !self.factor().is_finite() || self.factor() <= 0.0 {
            return Err(LlamaError::InvalidConfig(
                "rope scaling factor must be finite and positive".to_owned(),
            ));
        }
        if let Self::Yarn {
            beta_fast,
            beta_slow,
            original_context_length,
            attention_factor,
            ext_factor,
            ..
        } = self
        {
            if !beta_fast.is_finite() || beta_fast <= 0.0 {
                return Err(LlamaError::InvalidConfig(
                    "YaRN beta_fast must be finite and positive".to_owned(),
                ));
            }
            if !beta_slow.is_finite() || beta_slow <= 0.0 {
                return Err(LlamaError::InvalidConfig(
                    "YaRN beta_slow must be finite and positive".to_owned(),
                ));
            }
            if original_context_length == 0 {
                return Err(LlamaError::InvalidConfig(
                    "YaRN original context length must be greater than zero".to_owned(),
                ));
            }
            if !attention_factor.is_finite() || attention_factor <= 0.0 {
                return Err(LlamaError::InvalidConfig(
                    "YaRN attention factor must be finite and positive".to_owned(),
                ));
            }
            if !ext_factor.is_finite() || ext_factor < 0.0 {
                return Err(LlamaError::InvalidConfig(
                    "YaRN extension factor must be finite and nonnegative".to_owned(),
                ));
            }
        }
        Ok(())
    }
}

/// Validated architecture parameters for a Llama decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaConfig {
    context_length: usize,
    embedding_length: usize,
    block_count: usize,
    head_count: usize,
    head_count_kv: usize,
    key_length: usize,
    value_length: usize,
    feed_forward_length: usize,
    vocab_size: usize,
    rms_norm_epsilon: f32,
    rope_freq_base: f32,
    rope_dimension_count: usize,
    rope_scaling: LlamaRopeScaling,
    attention_temperature_scale: f32,
    attention_temperature_context: usize,
    attention_window: Option<usize>,
    attention_window_pattern: Option<Vec<bool>>,
}

#[derive(Debug, Clone, Copy)]
struct LlamaMetadataKeys {
    context_length: &'static str,
    embedding_length: &'static str,
    block_count: &'static str,
    head_count: &'static str,
    head_count_kv: &'static str,
    attention_key_length: &'static str,
    attention_value_length: &'static str,
    feed_forward_length: &'static str,
    vocab_size: &'static str,
    rms_norm_epsilon: &'static str,
    rope_freq_base: &'static str,
    rope_dimension_count: &'static str,
    rope_scaling_type: &'static str,
    rope_scaling_factor: &'static str,
    rope_scaling_attn_factor: &'static str,
    rope_scaling_original_context_length: &'static str,
    rope_scaling_yarn_log_multiplier: &'static str,
    rope_scaling_yarn_ext_factor: &'static str,
    rope_scaling_yarn_attn_factor: &'static str,
    rope_scaling_yarn_beta_fast: &'static str,
    rope_scaling_yarn_beta_slow: &'static str,
    rope_scaling_beta_fast_legacy: &'static str,
    rope_scaling_beta_slow_legacy: &'static str,
    rope_scaling_mscale_all_dim_legacy: &'static str,
    rope_scaling_beta_legacy: &'static str,
    attention_temperature_scale: &'static str,
    attention_window: &'static str,
    attention_window_pattern: &'static str,
}

trait MetadataSource {
    fn metadata_scalar(&self, key: &str) -> Result<Option<MetadataScalar>, ModelError>;
    fn metadata_scalars(&self, keys: &[&str]) -> Result<Vec<Option<MetadataScalar>>, ModelError>;
    fn metadata_string_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<String>>, ModelError>;
    fn metadata_f32_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<f32>>, ModelError>;
    fn metadata_bool_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<bool>>, ModelError>;
}

impl MetadataSource for GgufModel {
    fn metadata_scalar(&self, key: &str) -> Result<Option<MetadataScalar>, ModelError> {
        self.metadata_scalar(key)
    }

    fn metadata_scalars(&self, keys: &[&str]) -> Result<Vec<Option<MetadataScalar>>, ModelError> {
        self.metadata_scalars(keys)
    }

    fn metadata_string_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<String>>, ModelError> {
        self.metadata_string_array(key, max_elements)
    }

    fn metadata_f32_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<f32>>, ModelError> {
        self.metadata_f32_array(key, max_elements)
    }

    fn metadata_bool_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<bool>>, ModelError> {
        self.metadata_bool_array(key, max_elements)
    }
}

impl MetadataSource for GgufReadSession<'_> {
    fn metadata_scalar(&self, key: &str) -> Result<Option<MetadataScalar>, ModelError> {
        self.metadata_scalar(key)
    }

    fn metadata_scalars(&self, keys: &[&str]) -> Result<Vec<Option<MetadataScalar>>, ModelError> {
        self.metadata_scalars(keys)
    }

    fn metadata_string_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<String>>, ModelError> {
        self.metadata_string_array(key, max_elements)
    }

    fn metadata_f32_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<f32>>, ModelError> {
        self.metadata_f32_array(key, max_elements)
    }

    fn metadata_bool_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<bool>>, ModelError> {
        self.metadata_bool_array(key, max_elements)
    }
}

#[allow(clippy::too_many_lines)]
fn metadata_keys(architecture: &str) -> Option<LlamaMetadataKeys> {
    match architecture {
        "llama" => Some(LlamaMetadataKeys {
            context_length: "llama.context_length",
            embedding_length: "llama.embedding_length",
            block_count: "llama.block_count",
            head_count: "llama.attention.head_count",
            head_count_kv: "llama.attention.head_count_kv",
            attention_key_length: "llama.attention.key_length",
            attention_value_length: "llama.attention.value_length",
            feed_forward_length: "llama.feed_forward_length",
            vocab_size: "llama.vocab_size",
            rms_norm_epsilon: "llama.attention.layer_norm_rms_epsilon",
            rope_freq_base: "llama.rope.freq_base",
            rope_dimension_count: "llama.rope.dimension_count",
            rope_scaling_type: "llama.rope.scaling.type",
            rope_scaling_factor: "llama.rope.scaling.factor",
            rope_scaling_attn_factor: "llama.rope.scaling.attn_factor",
            rope_scaling_original_context_length: "llama.rope.scaling.original_context_length",
            rope_scaling_yarn_log_multiplier: "llama.rope.scaling.yarn_log_multiplier",
            rope_scaling_yarn_ext_factor: "llama.rope.scaling.yarn_ext_factor",
            rope_scaling_yarn_attn_factor: "llama.rope.scaling.yarn_attn_factor",
            rope_scaling_yarn_beta_fast: "llama.rope.scaling.yarn_beta_fast",
            rope_scaling_yarn_beta_slow: "llama.rope.scaling.yarn_beta_slow",
            rope_scaling_beta_fast_legacy: "llama.rope.scaling.beta_fast",
            rope_scaling_beta_slow_legacy: "llama.rope.scaling.beta_slow",
            rope_scaling_mscale_all_dim_legacy: "llama.rope.scaling.mscale_all_dim",
            rope_scaling_beta_legacy: "llama.rope.scaling_beta",
            attention_temperature_scale: "llama.attention.temperature_scale",
            attention_window: "llama.attention.sliding_window",
            attention_window_pattern: "llama.attention.sliding_window_pattern",
        }),
        "qwen2" => Some(LlamaMetadataKeys {
            context_length: "qwen2.context_length",
            embedding_length: "qwen2.embedding_length",
            block_count: "qwen2.block_count",
            head_count: "qwen2.attention.head_count",
            head_count_kv: "qwen2.attention.head_count_kv",
            attention_key_length: "qwen2.attention.key_length",
            attention_value_length: "qwen2.attention.value_length",
            feed_forward_length: "qwen2.feed_forward_length",
            vocab_size: "qwen2.vocab_size",
            rms_norm_epsilon: "qwen2.attention.layer_norm_rms_epsilon",
            rope_freq_base: "qwen2.rope.freq_base",
            rope_dimension_count: "qwen2.rope.dimension_count",
            rope_scaling_type: "qwen2.rope.scaling.type",
            rope_scaling_factor: "qwen2.rope.scaling.factor",
            rope_scaling_attn_factor: "qwen2.rope.scaling.attn_factor",
            rope_scaling_original_context_length: "qwen2.rope.scaling.original_context_length",
            rope_scaling_yarn_log_multiplier: "qwen2.rope.scaling.yarn_log_multiplier",
            rope_scaling_yarn_ext_factor: "qwen2.rope.scaling.yarn_ext_factor",
            rope_scaling_yarn_attn_factor: "qwen2.rope.scaling.yarn_attn_factor",
            rope_scaling_yarn_beta_fast: "qwen2.rope.scaling.yarn_beta_fast",
            rope_scaling_yarn_beta_slow: "qwen2.rope.scaling.yarn_beta_slow",
            rope_scaling_beta_fast_legacy: "qwen2.rope.scaling.beta_fast",
            rope_scaling_beta_slow_legacy: "qwen2.rope.scaling.beta_slow",
            rope_scaling_mscale_all_dim_legacy: "qwen2.rope.scaling.mscale_all_dim",
            rope_scaling_beta_legacy: "qwen2.rope.scaling_beta",
            attention_temperature_scale: "qwen2.attention.temperature_scale",
            attention_window: "qwen2.attention.sliding_window",
            attention_window_pattern: "qwen2.attention.sliding_window_pattern",
        }),
        "qwen3" => Some(LlamaMetadataKeys {
            context_length: "qwen3.context_length",
            embedding_length: "qwen3.embedding_length",
            block_count: "qwen3.block_count",
            head_count: "qwen3.attention.head_count",
            head_count_kv: "qwen3.attention.head_count_kv",
            attention_key_length: "qwen3.attention.key_length",
            attention_value_length: "qwen3.attention.value_length",
            feed_forward_length: "qwen3.feed_forward_length",
            vocab_size: "qwen3.vocab_size",
            rms_norm_epsilon: "qwen3.attention.layer_norm_rms_epsilon",
            rope_freq_base: "qwen3.rope.freq_base",
            rope_dimension_count: "qwen3.rope.dimension_count",
            rope_scaling_type: "qwen3.rope.scaling.type",
            rope_scaling_factor: "qwen3.rope.scaling.factor",
            rope_scaling_attn_factor: "qwen3.rope.scaling.attn_factor",
            rope_scaling_original_context_length: "qwen3.rope.scaling.original_context_length",
            rope_scaling_yarn_log_multiplier: "qwen3.rope.scaling.yarn_log_multiplier",
            rope_scaling_yarn_ext_factor: "qwen3.rope.scaling.yarn_ext_factor",
            rope_scaling_yarn_attn_factor: "qwen3.rope.scaling.yarn_attn_factor",
            rope_scaling_yarn_beta_fast: "qwen3.rope.scaling.yarn_beta_fast",
            rope_scaling_yarn_beta_slow: "qwen3.rope.scaling.yarn_beta_slow",
            rope_scaling_beta_fast_legacy: "qwen3.rope.scaling.beta_fast",
            rope_scaling_beta_slow_legacy: "qwen3.rope.scaling.beta_slow",
            rope_scaling_mscale_all_dim_legacy: "qwen3.rope.scaling.mscale_all_dim",
            rope_scaling_beta_legacy: "qwen3.rope.scaling_beta",
            attention_temperature_scale: "qwen3.attention.temperature_scale",
            attention_window: "qwen3.attention.sliding_window",
            attention_window_pattern: "qwen3.attention.sliding_window_pattern",
        }),
        "mistral" => Some(LlamaMetadataKeys {
            context_length: "mistral.context_length",
            embedding_length: "mistral.embedding_length",
            block_count: "mistral.block_count",
            head_count: "mistral.attention.head_count",
            head_count_kv: "mistral.attention.head_count_kv",
            attention_key_length: "mistral.attention.key_length",
            attention_value_length: "mistral.attention.value_length",
            feed_forward_length: "mistral.feed_forward_length",
            vocab_size: "mistral.vocab_size",
            rms_norm_epsilon: "mistral.attention.layer_norm_rms_epsilon",
            rope_freq_base: "mistral.rope.freq_base",
            rope_dimension_count: "mistral.rope.dimension_count",
            rope_scaling_type: "mistral.rope.scaling.type",
            rope_scaling_factor: "mistral.rope.scaling.factor",
            rope_scaling_attn_factor: "mistral.rope.scaling.attn_factor",
            rope_scaling_original_context_length: "mistral.rope.scaling.original_context_length",
            rope_scaling_yarn_log_multiplier: "mistral.rope.scaling.yarn_log_multiplier",
            rope_scaling_yarn_ext_factor: "mistral.rope.scaling.yarn_ext_factor",
            rope_scaling_yarn_attn_factor: "mistral.rope.scaling.yarn_attn_factor",
            rope_scaling_yarn_beta_fast: "mistral.rope.scaling.yarn_beta_fast",
            rope_scaling_yarn_beta_slow: "mistral.rope.scaling.yarn_beta_slow",
            rope_scaling_beta_fast_legacy: "mistral.rope.scaling.beta_fast",
            rope_scaling_beta_slow_legacy: "mistral.rope.scaling.beta_slow",
            rope_scaling_mscale_all_dim_legacy: "mistral.rope.scaling.mscale_all_dim",
            rope_scaling_beta_legacy: "mistral.rope.scaling_beta",
            attention_temperature_scale: "mistral.attention.temperature_scale",
            attention_window: "mistral.attention.sliding_window",
            attention_window_pattern: "mistral.attention.sliding_window_pattern",
        }),
        "mistral3" => Some(LlamaMetadataKeys {
            context_length: "mistral3.context_length",
            embedding_length: "mistral3.embedding_length",
            block_count: "mistral3.block_count",
            head_count: "mistral3.attention.head_count",
            head_count_kv: "mistral3.attention.head_count_kv",
            attention_key_length: "mistral3.attention.key_length",
            attention_value_length: "mistral3.attention.value_length",
            feed_forward_length: "mistral3.feed_forward_length",
            vocab_size: "mistral3.vocab_size",
            rms_norm_epsilon: "mistral3.attention.layer_norm_rms_epsilon",
            rope_freq_base: "mistral3.rope.freq_base",
            rope_dimension_count: "mistral3.rope.dimension_count",
            rope_scaling_type: "mistral3.rope.scaling.type",
            rope_scaling_factor: "mistral3.rope.scaling.factor",
            rope_scaling_attn_factor: "mistral3.rope.scaling.attn_factor",
            rope_scaling_original_context_length: "mistral3.rope.scaling.original_context_length",
            rope_scaling_yarn_log_multiplier: "mistral3.rope.scaling.yarn_log_multiplier",
            rope_scaling_yarn_ext_factor: "mistral3.rope.scaling.yarn_ext_factor",
            rope_scaling_yarn_attn_factor: "mistral3.rope.scaling.yarn_attn_factor",
            rope_scaling_yarn_beta_fast: "mistral3.rope.scaling.yarn_beta_fast",
            rope_scaling_yarn_beta_slow: "mistral3.rope.scaling.yarn_beta_slow",
            rope_scaling_beta_fast_legacy: "mistral3.rope.scaling.beta_fast",
            rope_scaling_beta_slow_legacy: "mistral3.rope.scaling.beta_slow",
            rope_scaling_mscale_all_dim_legacy: "mistral3.rope.scaling.mscale_all_dim",
            rope_scaling_beta_legacy: "mistral3.rope.scaling_beta",
            attention_temperature_scale: "mistral3.attention.temperature_scale",
            attention_window: "mistral3.attention.sliding_window",
            attention_window_pattern: "mistral3.attention.sliding_window_pattern",
        }),
        _ => None,
    }
}

fn validate_architecture_metadata<S>(source: &S, architecture: &str) -> Result<(), LlamaError>
where
    S: MetadataSource + ?Sized,
{
    if architecture == "qwen3" && metadata_key_is_present(source, "qwen3.classifier.output_labels")?
    {
        return Err(LlamaError::InvalidConfig(
            "unsupported qwen3 classifier or reranker metadata qwen3.classifier.output_labels"
                .to_owned(),
        ));
    }
    if architecture == "qwen3" {
        let pooling_type = optional_usize_value(
            source.metadata_scalar("qwen3.pooling_type")?,
            "qwen3.pooling_type",
        )?
        .unwrap_or(0);
        if pooling_type != 0 {
            return Err(LlamaError::InvalidConfig(format!(
                "unsupported qwen3.pooling_type {pooling_type}; causal decoding requires zero"
            )));
        }
    }
    if architecture == "mistral3" {
        let expert_count = optional_usize_value(
            source.metadata_scalar("mistral3.expert_count")?,
            "mistral3.expert_count",
        )?
        .unwrap_or(0);
        if expert_count > 0 {
            return Err(LlamaError::InvalidConfig(format!(
                "unsupported mistral3.expert_count {expert_count}; mixture-of-experts execution is unavailable"
            )));
        }
    }
    Ok(())
}

fn metadata_key_is_present<S>(source: &S, key: &str) -> Result<bool, LlamaError>
where
    S: MetadataSource + ?Sized,
{
    match source.metadata_scalar(key) {
        Ok(value) => Ok(value.is_some()),
        Err(ModelError::MetadataArray(array_key)) if array_key == key => Ok(true),
        Err(error) => Err(LlamaError::Model(error)),
    }
}

fn validate_architecture_tensors(model: &GgufModel, architecture: &str) -> Result<(), LlamaError> {
    if architecture == "qwen3" && model.tensor("cls.output.weight").is_some() {
        return Err(LlamaError::InvalidConfig(
            "unsupported qwen3 classifier or reranker tensor cls.output.weight".to_owned(),
        ));
    }
    if metadata_keys(architecture).is_some() {
        for name in ["rope_factors_long.weight", "rope_factors_short.weight"] {
            if model.tensor(name).is_some() {
                return Err(LlamaError::InvalidConfig(format!(
                    "unsupported decoder rotary tensor {name}"
                )));
            }
        }
    }
    Ok(())
}

fn load_rope_freq_factors(
    model: &GgufModel,
    config: &LlamaConfig,
    session: &GgufReadSession<'_>,
) -> Result<Option<Vec<f32>>, LlamaError> {
    const NAME: &str = "rope_freqs.weight";
    let Some(descriptor) = model.tensor(NAME) else {
        return Ok(None);
    };
    let expected = vec![config.rope_dimension_count / 2];
    if descriptor.shape() != expected {
        return Err(LlamaError::TensorShape {
            name: NAME.to_owned(),
            expected,
            actual: descriptor.shape().to_vec(),
        });
    }
    if descriptor.value_type().raw() != 0 {
        return Err(LlamaError::InvalidConfig(format!(
            "{NAME} must use F32 storage, got {}",
            descriptor.value_type().name()
        )));
    }
    let raw = session.load_raw(NAME)?;
    let (chunks, remainder) = raw.encoded_bytes().as_chunks::<4>();
    let mut factors = Vec::with_capacity(expected[0]);
    for (index, bytes) in chunks.iter().enumerate() {
        let factor = f32::from_le_bytes(*bytes);
        if !factor.is_finite() || factor <= 0.0 {
            return Err(LlamaError::InvalidConfig(format!(
                "{NAME} factor at pair {index} must be finite and positive"
            )));
        }
        factors.push(factor);
    }
    if !remainder.is_empty() || factors.len() != expected[0] {
        return Err(LlamaError::Tensor(format!(
            "{NAME} encoded byte length does not match its shape"
        )));
    }
    Ok(Some(factors))
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
        let rope_dimension_count = embedding_length.checked_div(head_count).unwrap_or(0);
        Self::new_with_rope_dimension(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
        )
    }

    /// Creates a validated Llama configuration with an explicit rotary width.
    ///
    /// `rope_dimension_count` may be smaller than the attention head width for
    /// architectures that rotate only a prefix of each head.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or numerical parameters are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rope_dimension(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        rope_freq_base: f32,
        rope_dimension_count: usize,
    ) -> Result<Self, LlamaError> {
        Self::new_with_rope_scaling(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
            LlamaRopeScaling::None,
        )
    }

    /// Creates a validated Llama configuration with rotary width and scaling.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, numerical parameters, or the scaling
    /// factor are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rope_scaling(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        rope_freq_base: f32,
        rope_dimension_count: usize,
        rope_scaling: LlamaRopeScaling,
    ) -> Result<Self, LlamaError> {
        Self::new_with_rope_scaling_and_attention_window(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
            rope_scaling,
            None,
        )
    }

    /// Creates a validated Llama-compatible configuration with rotary scaling
    /// and an optional sliding attention window.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions, numerical parameters, scaling, or the
    /// attention window are invalid.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_rope_scaling_and_attention_window(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        rope_freq_base: f32,
        rope_dimension_count: usize,
        rope_scaling: LlamaRopeScaling,
        attention_window: Option<usize>,
    ) -> Result<Self, LlamaError> {
        Self::new_with_rope_scaling_and_attention_window_and_pattern(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
            rope_scaling,
            attention_window,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_rope_scaling_and_attention_window_and_pattern(
        context_length: usize,
        embedding_length: usize,
        block_count: usize,
        head_count: usize,
        head_count_kv: usize,
        feed_forward_length: usize,
        vocab_size: usize,
        rms_norm_epsilon: f32,
        rope_freq_base: f32,
        rope_dimension_count: usize,
        rope_scaling: LlamaRopeScaling,
        attention_window: Option<usize>,
        attention_window_pattern: Option<Vec<bool>>,
    ) -> Result<Self, LlamaError> {
        let config = Self {
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            key_length: embedding_length.checked_div(head_count).unwrap_or(0),
            value_length: embedding_length.checked_div(head_count).unwrap_or(0),
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
            rope_scaling,
            attention_temperature_scale: 0.0,
            attention_temperature_context: 0,
            attention_window,
            attention_window_pattern,
        };
        config.validate()?;
        Ok(config)
    }

    /// Builds a configuration from canonical Llama-compatible GGUF metadata.
    ///
    /// # Errors
    ///
    /// Returns an error when metadata is missing, malformed, or the model
    /// architecture is unsupported or not Llama-compatible.
    #[allow(clippy::too_many_lines)]
    pub fn from_model(model: &GgufModel) -> Result<Self, LlamaError> {
        let architecture = model.architecture().unwrap_or("missing");
        Self::from_metadata(model, architecture)
    }

    #[allow(clippy::too_many_lines)]
    fn from_metadata<S>(source: &S, architecture: &str) -> Result<Self, LlamaError>
    where
        S: MetadataSource + ?Sized,
    {
        let keys = metadata_keys(architecture)
            .ok_or_else(|| LlamaError::UnsupportedArchitecture(architecture.to_owned()))?;
        validate_architecture_metadata(source, architecture)?;
        let metadata = [
            keys.context_length,
            keys.embedding_length,
            keys.block_count,
            keys.head_count,
            keys.head_count_kv,
            keys.feed_forward_length,
            keys.vocab_size,
            keys.rms_norm_epsilon,
            keys.rope_freq_base,
        ];
        let values = source.metadata_scalars(&metadata)?;
        let mut values = values.into_iter();
        let context_length = required_usize_value(values.next().flatten(), keys.context_length)?;
        let embedding_length =
            required_usize_value(values.next().flatten(), keys.embedding_length)?;
        let block_count = required_usize_value(values.next().flatten(), keys.block_count)?;
        let head_count = required_usize_value(values.next().flatten(), keys.head_count)?;
        let head_count_kv = optional_usize_value(values.next().flatten(), keys.head_count_kv)?
            .unwrap_or(head_count);
        let key_length = optional_usize_value(
            source.metadata_scalar(keys.attention_key_length)?,
            keys.attention_key_length,
        )?
        .unwrap_or_else(|| embedding_length.checked_div(head_count).unwrap_or(0));
        let value_length = optional_usize_value(
            source.metadata_scalar(keys.attention_value_length)?,
            keys.attention_value_length,
        )?
        .unwrap_or(key_length);
        let feed_forward_length =
            required_usize_value(values.next().flatten(), keys.feed_forward_length)?;
        let vocab_size = match values.next().flatten() {
            Some(value) => as_usize(value).map_err(|value| LlamaError::InvalidMetadata {
                key: keys.vocab_size,
                value,
            })?,
            None => source
                .metadata_string_array("tokenizer.ggml.tokens", MAX_TOKENIZER_ELEMENTS)?
                .ok_or(LlamaError::MissingMetadata(keys.vocab_size))?
                .len(),
        };
        let rms_norm_epsilon =
            optional_f32_value(values.next().flatten(), keys.rms_norm_epsilon)?.unwrap_or(1.0e-5);
        let rope_freq_base =
            optional_f32_value(values.next().flatten(), keys.rope_freq_base)?.unwrap_or(10_000.0);
        let rope_dimension_count = source
            .metadata_scalar(keys.rope_dimension_count)?
            .map(|value| {
                as_usize(value).map_err(|value| LlamaError::InvalidMetadata {
                    key: keys.rope_dimension_count,
                    value,
                })
            })
            .transpose()?
            .unwrap_or_else(|| embedding_length.checked_div(head_count).unwrap_or(0));
        let rope_scaling_type = source.metadata_scalar(keys.rope_scaling_type)?;
        let rope_scaling_factor = optional_f32_value(
            source.metadata_scalar(keys.rope_scaling_factor)?,
            keys.rope_scaling_factor,
        )?;
        let rope_scaling = parse_model_rope_scaling(
            source,
            &keys,
            rope_scaling_type,
            rope_scaling_factor,
            context_length,
        )?;
        let attention_temperature_scale = optional_f32_value(
            source
                .metadata_scalar(keys.attention_temperature_scale)?
                .or(source.metadata_scalar(keys.rope_scaling_beta_legacy)?),
            keys.attention_temperature_scale,
        )?
        .unwrap_or(0.0);
        let attention_temperature_context = if attention_temperature_scale == 0.0 {
            0
        } else {
            optional_usize_value(
                source.metadata_scalar(keys.rope_scaling_original_context_length)?,
                keys.rope_scaling_original_context_length,
            )?
            .unwrap_or(context_length)
        };
        let attention_window = optional_usize_value(
            source.metadata_scalar(keys.attention_window)?,
            keys.attention_window,
        )?;
        let attention_window_pattern =
            parse_attention_window_pattern(source, keys.attention_window_pattern, block_count)?;
        let mut config = Self::new_with_rope_scaling_and_attention_window_and_pattern(
            context_length,
            embedding_length,
            block_count,
            head_count,
            head_count_kv,
            feed_forward_length,
            vocab_size,
            rms_norm_epsilon,
            rope_freq_base,
            rope_dimension_count,
            rope_scaling,
            attention_window,
            attention_window_pattern,
        )?;
        config.attention_temperature_scale = attention_temperature_scale;
        config.attention_temperature_context = attention_temperature_context;
        config.key_length = key_length;
        config.value_length = value_length;
        config.validate()?;
        Ok(config)
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

    /// Returns the key/query width of one attention head.
    #[must_use]
    pub const fn key_length(&self) -> usize {
        self.key_length
    }

    /// Returns the value width of one attention head.
    #[must_use]
    pub const fn value_length(&self) -> usize {
        self.value_length
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

    /// Returns the number of values rotated in each attention head.
    #[must_use]
    pub const fn rope_dimension_count(&self) -> usize {
        self.rope_dimension_count
    }

    /// Returns the configured rotary position scaling.
    #[must_use]
    pub const fn rope_scaling(&self) -> LlamaRopeScaling {
        self.rope_scaling
    }

    /// Returns the rotary position divisor, or `1.0` when scaling is disabled.
    #[must_use]
    pub const fn rope_scaling_factor(&self) -> f32 {
        self.rope_scaling.factor()
    }

    /// Returns the rotary angle and magnitude for one pair of head values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rope_phase(&self, position: f32, pair: usize, head_dim: usize) -> (f32, f32) {
        self.rope_scaling.phase(
            position,
            pair,
            head_dim as f32,
            self.rope_dimension_count,
            self.rope_freq_base,
            1.0,
        )
    }

    /// Returns the `YaRN` or model-specific attention magnitude applied to `RoPE`.
    #[must_use]
    pub const fn rope_attention_factor(&self) -> f32 {
        match self.rope_scaling {
            LlamaRopeScaling::Yarn {
                attention_factor, ..
            } => attention_factor,
            _ => 1.0,
        }
    }

    /// Returns the configured attention-temperature scale coefficient.
    #[must_use]
    pub const fn attention_temperature_scale(&self) -> f32 {
        self.attention_temperature_scale
    }

    /// Returns the dynamic query multiplier used by Mistral3 temperature tuning.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn attention_temperature_multiplier(&self, position: usize) -> f32 {
        if self.attention_temperature_scale == 0.0 {
            return 1.0;
        }
        let floor = self.attention_temperature_context.max(1);
        ((position / floor) as f32 + 1.0).ln() * self.attention_temperature_scale + 1.0
    }

    /// Returns the optional causal attention window in tokens.
    #[must_use]
    pub const fn attention_window(&self) -> Option<usize> {
        self.attention_window
    }

    /// Returns the optional per-layer sliding-window activation pattern.
    #[must_use]
    pub fn attention_window_pattern(&self) -> Option<&[bool]> {
        self.attention_window_pattern.as_deref()
    }

    /// Returns the first cached token visible to the current attention step.
    #[must_use]
    pub fn attention_start(&self, cached_tokens: usize) -> usize {
        self.attention_window
            .map_or(0, |window| cached_tokens.saturating_sub(window))
    }

    /// Returns the first cached token visible to one layer's attention step.
    #[must_use]
    pub fn attention_start_for_layer(&self, layer: usize, cached_tokens: usize) -> usize {
        if self
            .attention_window_pattern
            .as_ref()
            .and_then(|pattern| pattern.get(layer))
            .is_some_and(|uses_window| !uses_window)
        {
            0
        } else {
            self.attention_start(cached_tokens)
        }
    }

    /// Returns the physical KV-cache capacity required by one decoder layer.
    ///
    /// Layers using a sliding window can use a bounded ring instead of
    /// reserving the full model context. Dense layers retain the full context
    /// so mixed local and global attention remains exact.
    #[must_use]
    pub fn kv_cache_capacity_for_layer(&self, layer: usize) -> usize {
        let uses_window = self
            .attention_window_pattern
            .as_ref()
            .and_then(|pattern| pattern.get(layer))
            .copied()
            .unwrap_or(true);
        if uses_window {
            self.attention_window.map_or(self.context_length, |window| {
                window.min(self.context_length)
            })
        } else {
            self.context_length
        }
    }

    /// Converts a token index to the rotary position used by this model.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn scaled_rope_position(&self, position: usize) -> f32 {
        position as f32 / self.rope_scaling_factor()
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
        if self.key_length == 0 || self.value_length == 0 {
            return Err(LlamaError::InvalidConfig(
                "attention key and value lengths must be greater than zero".to_owned(),
            ));
        }
        if self.key_length != self.value_length {
            return Err(LlamaError::InvalidConfig(
                "attention key and value lengths must match for this decoder".to_owned(),
            ));
        }
        let head_dim = self.key_length;
        if !head_dim.is_multiple_of(2) {
            return Err(LlamaError::InvalidConfig(
                "head dimension must be even for rotary embeddings".to_owned(),
            ));
        }
        if self.rope_dimension_count == 0
            || self.rope_dimension_count > head_dim
            || !self.rope_dimension_count.is_multiple_of(2)
        {
            return Err(LlamaError::InvalidConfig(
                "rope_dimension_count must be positive, even, and no larger than head dimension"
                    .to_owned(),
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
        self.rope_scaling.validate()?;
        if !self.attention_temperature_scale.is_finite() {
            return Err(LlamaError::InvalidConfig(
                "attention temperature scale must be finite".to_owned(),
            ));
        }
        if self.attention_temperature_scale != 0.0 && self.attention_temperature_context == 0 {
            return Err(LlamaError::InvalidConfig(
                "attention temperature context must be greater than zero when temperature scaling is enabled".to_owned(),
            ));
        }
        if self.attention_window == Some(0) {
            return Err(LlamaError::InvalidConfig(
                "attention_window must be greater than zero".to_owned(),
            ));
        }
        if let Some(pattern) = &self.attention_window_pattern {
            if pattern.len() != self.block_count {
                return Err(LlamaError::InvalidConfig(format!(
                    "attention_window_pattern has {} entries, expected {}",
                    pattern.len(),
                    self.block_count
                )));
            }
            if self.attention_window.is_none() {
                return Err(LlamaError::InvalidConfig(
                    "attention_window_pattern requires attention_window".to_owned(),
                ));
            }
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
    rope_freq_factors: Option<Vec<f32>>,
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
        let architecture = model.architecture().unwrap_or("missing");
        validate_architecture_tensors(&model, architecture)?;
        let session = model.read_session()?;
        let config = LlamaConfig::from_metadata(&session, architecture)?;
        let rope_freq_factors = load_rope_freq_factors(&model, &config, &session)?;
        session.verify_unchanged()?;
        validate_layout(&model, &config)?;
        Ok(Self {
            model,
            config,
            rope_freq_factors,
        })
    }

    /// Validates a model against an explicit configuration.
    ///
    /// # Errors
    ///
    /// Returns an error when the architecture or required tensor layout does
    /// not match the configuration.
    pub fn from_model(model: GgufModel, config: LlamaConfig) -> Result<Self, LlamaError> {
        let architecture = model.architecture().unwrap_or("missing");
        if metadata_keys(architecture).is_none() {
            return Err(LlamaError::UnsupportedArchitecture(architecture.to_owned()));
        }
        validate_architecture_tensors(&model, architecture)?;
        let session = model.read_session()?;
        validate_architecture_metadata(&session, architecture)?;
        let rope_freq_factors = load_rope_freq_factors(&model, &config, &session)?;
        session.verify_unchanged()?;
        validate_layout(&model, &config)?;
        Ok(Self {
            model,
            config,
            rope_freq_factors,
        })
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

    /// Returns the validated optional per-pair rotary frequency factors.
    ///
    /// GGML divides each pair's unscaled rotary phase by the corresponding
    /// factor before applying linear or `YaRN` interpolation.
    #[must_use]
    pub fn rope_freq_factors(&self) -> Option<&[f32]> {
        self.rope_freq_factors.as_deref()
    }

    /// Loads and validates rotary frequency factors through an existing GGUF
    /// read transaction.
    ///
    /// This lets a device backend admit the optional `rope_freqs.weight`
    /// tensor without opening another mapping or hashing the model again. The
    /// caller remains responsible for ending the transaction with
    /// [`GgufReadSession::verify_unchanged`].
    ///
    /// # Errors
    ///
    /// Returns an error when the session belongs to another model, or when the
    /// factor tensor has the wrong type, shape, or values.
    pub fn rope_freq_factors_from_session(
        &self,
        session: &GgufReadSession<'_>,
    ) -> Result<Option<Vec<f32>>, LlamaError> {
        if session.model() != &self.model {
            return Err(LlamaError::Tensor(
                "GGUF read session belongs to another model".to_owned(),
            ));
        }
        load_rope_freq_factors(&self.model, &self.config, session)
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
        let session = self.model.read_session()?;
        let tokenizer = self.tokenizer_from_session(&session)?;
        session.verify_unchanged()?;
        Ok(tokenizer)
    }

    /// Loads tokenizer tables through an existing validated GGUF read
    /// transaction.
    ///
    /// This lets a device backend share one model mapping across tokenizer
    /// admission and weight upload. The caller remains responsible for ending
    /// the transaction with [`GgufReadSession::verify_unchanged`].
    ///
    /// # Errors
    ///
    /// Returns an error when the session belongs to another model or tokenizer
    /// metadata is absent, malformed, or inconsistent with the vocabulary.
    pub fn tokenizer_from_session(
        &self,
        session: &GgufReadSession<'_>,
    ) -> Result<LlamaTokenizer, LlamaError> {
        if session.model() != &self.model {
            return Err(LlamaError::Tensor(
                "GGUF read session belongs to another model".to_owned(),
            ));
        }
        LlamaTokenizer::from_metadata(session, self.config.vocab_size)
    }
}

const MAX_TOKENIZER_ELEMENTS: u64 = 16 * 1024 * 1024;
const MAX_CONFIG_ARRAY_ELEMENTS: u64 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TokenizerKind {
    SentencePiece,
    Gpt2Bpe,
    TekkenBpe,
}

/// A bounded GGUF tokenizer vocabulary with `SentencePiece` and GPT-2 BPE modes.
#[derive(Debug, Clone, PartialEq)]
pub struct LlamaTokenizer {
    tokens: Vec<String>,
    scores: Vec<f32>,
    bos_token_id: Option<usize>,
    eos_token_id: Option<usize>,
    unk_token_id: Option<usize>,
    kind: TokenizerKind,
    token_ids: HashMap<String, usize>,
    merge_ranks: HashMap<String, usize>,
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
    #[allow(clippy::too_many_lines)]
    fn from_metadata<S>(source: &S, vocab_size: usize) -> Result<Self, LlamaError>
    where
        S: MetadataSource + ?Sized,
    {
        let tokens = source
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
        let scores = source
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
        let token_id_values = source.metadata_scalars(&token_id_keys)?;
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
        let kind = match source.metadata_scalar("tokenizer.ggml.model")? {
            None => TokenizerKind::SentencePiece,
            Some(MetadataScalar::String(value)) if value == "gpt2" => {
                match source.metadata_scalar("tokenizer.ggml.pre")? {
                    Some(MetadataScalar::String(pre)) if pre == "tekken" => {
                        TokenizerKind::TekkenBpe
                    }
                    None | Some(MetadataScalar::String(_)) => TokenizerKind::Gpt2Bpe,
                    Some(value) => {
                        return Err(LlamaError::InvalidMetadata {
                            key: "tokenizer.ggml.pre",
                            value: format!("expected a string, got {value:?}"),
                        });
                    }
                }
            }
            Some(MetadataScalar::String(value)) if value == "llama" || value == "sentencepiece" => {
                TokenizerKind::SentencePiece
            }
            Some(value) => {
                return Err(LlamaError::InvalidMetadata {
                    key: "tokenizer.ggml.model",
                    value: format!("unsupported tokenizer model {value:?}"),
                });
            }
        };
        let token_ids = if matches!(kind, TokenizerKind::Gpt2Bpe | TokenizerKind::TekkenBpe) {
            tokens
                .iter()
                .enumerate()
                .map(|(id, token)| (token.clone(), id))
                .collect()
        } else {
            HashMap::new()
        };
        let merge_ranks = if matches!(kind, TokenizerKind::Gpt2Bpe | TokenizerKind::TekkenBpe) {
            let merges = source
                .metadata_string_array("tokenizer.ggml.merges", MAX_TOKENIZER_ELEMENTS)?
                .ok_or(LlamaError::MissingMetadata("tokenizer.ggml.merges"))?;
            let mut ranks = HashMap::with_capacity(merges.len());
            for (rank, merge) in merges.into_iter().enumerate() {
                let mut pieces = merge.splitn(2, ' ');
                let left = pieces.next().unwrap_or_default();
                let right = pieces.next().unwrap_or_default();
                if left.is_empty() || right.is_empty() {
                    return Err(LlamaError::InvalidMetadata {
                        key: "tokenizer.ggml.merges",
                        value: format!("merge {rank} is not a pair"),
                    });
                }
                ranks.insert(bpe_pair_key(left, right), rank);
            }
            ranks
        } else {
            HashMap::new()
        };
        Ok(Self {
            tokens,
            scores,
            bos_token_id,
            eos_token_id,
            unk_token_id,
            kind,
            token_ids,
            merge_ranks,
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
        if matches!(self.kind, TokenizerKind::Gpt2Bpe | TokenizerKind::TekkenBpe) {
            return self.encode_gpt2_bpe(text);
        }
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

    fn encode_gpt2_bpe(&self, text: &str) -> Result<Vec<usize>, LlamaError> {
        let encoder = byte_encoder_table();
        let mut token_ids = Vec::new();
        let chunks = if self.kind == TokenizerKind::TekkenBpe {
            gpt2_pretokenize_tekken(text)
        } else {
            gpt2_pretokenize(text)
        };
        for chunk in chunks {
            let symbols = chunk
                .as_bytes()
                .iter()
                .map(|byte| encoder[usize::from(*byte)].to_string())
                .collect::<Vec<_>>();
            let symbols = self.merge_bpe_symbols(symbols);
            for symbol in symbols {
                if let Some(&token_id) = self.token_ids.get(&symbol) {
                    token_ids.push(token_id);
                    continue;
                }
                for character in symbol.chars() {
                    let piece = character.to_string();
                    let token_id = self.token_ids.get(&piece).copied().or(self.unk_token_id);
                    let Some(token_id) = token_id else {
                        return Err(LlamaError::InvalidMetadata {
                            key: "tokenizer.ggml.tokens",
                            value: format!("no BPE token matches {piece:?}"),
                        });
                    };
                    token_ids.push(token_id);
                }
            }
        }
        Ok(token_ids)
    }

    fn merge_bpe_symbols(&self, mut symbols: Vec<String>) -> Vec<String> {
        while symbols.len() > 1 {
            let mut best: Option<(usize, usize)> = None;
            for index in 0..symbols.len() - 1 {
                let key = bpe_pair_key(&symbols[index], &symbols[index + 1]);
                let Some(&rank) = self.merge_ranks.get(&key) else {
                    continue;
                };
                if best.is_none_or(|(_, best_rank)| rank < best_rank) {
                    best = Some((index, rank));
                }
            }
            let Some((index, _)) = best else {
                break;
            };
            let right = symbols.remove(index + 1);
            symbols[index].push_str(&right);
        }
        symbols
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
    /// Invalid UTF-8 assembled from byte-fallback pieces is replaced with
    /// U+FFFD so arbitrary sampled token sequences remain decodable.
    ///
    /// # Errors
    ///
    /// Returns an error when a token id is outside the vocabulary.
    pub fn decode(&self, token_ids: &[usize]) -> Result<String, LlamaError> {
        if self.kind == TokenizerKind::Gpt2Bpe {
            return self.decode_gpt2_bpe(token_ids);
        }
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
                output.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
            }
            output.push_str(&token.replace('▁', " "));
        }
        if !bytes.is_empty() {
            output.push_str(&String::from_utf8_lossy(&bytes));
        }
        Ok(output)
    }

    fn decode_gpt2_bpe(&self, token_ids: &[usize]) -> Result<String, LlamaError> {
        let mut bytes = Vec::new();
        let mut output = String::new();
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
            let mut decoded = true;
            for character in token.chars() {
                if let Some(byte) = unicode_to_byte(character) {
                    bytes.push(byte);
                } else {
                    decoded = false;
                    break;
                }
            }
            if !decoded {
                output.push_str(&String::from_utf8_lossy(&bytes));
                bytes.clear();
                output.push_str(token);
            }
        }
        output.push_str(&String::from_utf8_lossy(&bytes));
        Ok(output)
    }
}

fn bpe_pair_key(left: &str, right: &str) -> String {
    let mut key = String::with_capacity(left.len() + right.len() + 1);
    key.push_str(left);
    key.push('\0');
    key.push_str(right);
    key
}

fn byte_encoder_table() -> [char; 256] {
    let mut table = ['\0'; 256];
    let mut byte = 0_u16;
    let mut mapped = 0_u32;
    while byte < 256 {
        let value = u8::try_from(byte).unwrap_or_default();
        if (33..=126).contains(&value)
            || (161..=172).contains(&value)
            || (174..=255).contains(&value)
        {
            table[usize::from(value)] = char::from(value);
        } else {
            table[usize::from(value)] = char::from_u32(256 + mapped).unwrap_or('\u{fffd}');
            mapped += 1;
        }
        byte += 1;
    }
    table
}

fn unicode_to_byte(character: char) -> Option<u8> {
    let table = byte_encoder_table();
    table
        .iter()
        .position(|candidate| *candidate == character)
        .and_then(|index| u8::try_from(index).ok())
}

fn gpt2_pretokenize(text: &str) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let mut prefix = String::new();
        if characters[index] == ' '
            && index + 1 < characters.len()
            && !characters[index + 1].is_whitespace()
        {
            prefix.push(' ');
            index += 1;
        }
        let start = index;
        if index + 1 < characters.len() && characters[index] == '\'' {
            let suffixes = ["re", "ve", "ll", "s", "t", "m", "d"];
            if let Some(suffix) = suffixes.iter().find(|suffix| {
                characters[index + 1..]
                    .iter()
                    .take(suffix.len())
                    .collect::<String>()
                    .eq_ignore_ascii_case(suffix)
            }) {
                index += 1 + suffix.len();
                let mut chunk = prefix;
                chunk.extend(characters[start..index].iter());
                chunks.push(chunk);
                continue;
            }
        }
        let category = characters[index];
        if category.is_alphabetic() {
            index += 1;
            while index < characters.len() && characters[index].is_alphabetic() {
                index += 1;
            }
        } else if category.is_numeric() {
            index += 1;
        } else if category.is_whitespace() {
            index += 1;
            while index < characters.len() && characters[index].is_whitespace() {
                index += 1;
            }
        } else {
            index += 1;
            while index < characters.len()
                && !characters[index].is_whitespace()
                && !characters[index].is_alphabetic()
                && !characters[index].is_numeric()
            {
                index += 1;
            }
        }
        let mut chunk = prefix;
        chunk.extend(characters[start..index].iter());
        chunks.push(chunk);
    }
    chunks
}

/// Splits the byte-level BPE stream used by Mistral's Tekken tokenizer.
///
/// Tekken keeps numbers as single-character chunks and does not special-case
/// English contractions. Keeping those boundaries here is important because
/// the subsequent merge table is byte-ranked and cannot repair a different
/// pre-tokenization boundary.
fn gpt2_pretokenize_tekken(text: &str) -> Vec<String> {
    let characters = text.chars().collect::<Vec<_>>();
    let mut chunks = Vec::new();
    let mut index = 0;
    while index < characters.len() {
        let start = index;
        let character = characters[index];

        if character == '\r' || character == '\n' {
            index += 1;
            while index < characters.len()
                && (characters[index] == '\r' || characters[index] == '\n')
            {
                index += 1;
            }
        } else if character.is_whitespace()
            && !(character == ' '
                && index + 1 < characters.len()
                && characters[index + 1].is_alphabetic())
        {
            index += 1;
            while index < characters.len()
                && characters[index].is_whitespace()
                && characters[index] != '\r'
                && characters[index] != '\n'
            {
                index += 1;
            }
        } else if character.is_numeric() {
            index += 1;
        } else if character.is_alphabetic() {
            index = tekken_word_end(&characters, index);
        } else if index + 1 < characters.len() && characters[index + 1].is_alphabetic() {
            // The Tekken pattern allows one non-letter prefix on a word. This
            // is what makes a contraction such as "can't" become "can" +
            // "'t" instead of one GPT-style contraction chunk.
            index += 1;
            index = tekken_word_end(&characters, index);
        } else {
            index += 1;
            while index < characters.len()
                && !characters[index].is_whitespace()
                && !characters[index].is_alphabetic()
                && !characters[index].is_numeric()
                && characters[index] != '\r'
                && characters[index] != '\n'
            {
                index += 1;
            }
        }
        chunks.push(characters[start..index].iter().collect());
    }
    chunks
}

fn tekken_word_end(characters: &[char], mut index: usize) -> usize {
    let starts_lowercase = characters[index].is_lowercase();
    let mut saw_lowercase = false;
    while index < characters.len() && characters[index].is_alphabetic() {
        let character = characters[index];
        if starts_lowercase {
            if character.is_uppercase() && saw_lowercase {
                break;
            }
        } else if character.is_lowercase() {
            saw_lowercase = true;
        } else if saw_lowercase && character.is_uppercase() {
            break;
        }
        if character.is_lowercase() {
            saw_lowercase = true;
        }
        index += 1;
    }
    index
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
    fn from_tensor(tensor: Tensor) -> Result<Self, LlamaError> {
        Ok(Self::F32(transpose_ggml_matrix(tensor)?))
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
    attn_q_norm: Option<Vec<f32>>,
    attn_q_bias: Option<Vec<f32>>,
    attn_k: CpuMatrix,
    attn_k_norm: Option<Vec<f32>>,
    attn_k_bias: Option<Vec<f32>>,
    attn_v: CpuMatrix,
    attn_v_bias: Option<Vec<f32>>,
    attn_output: CpuMatrix,
    attn_output_bias: Option<Vec<f32>>,
    ffn_norm: Tensor,
    ffn_gate: CpuMatrix,
    ffn_gate_bias: Option<Vec<f32>>,
    ffn_down: CpuMatrix,
    ffn_down_bias: Option<Vec<f32>>,
    ffn_up: CpuMatrix,
    ffn_up_bias: Option<Vec<f32>>,
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
    rope_freq_factors: Option<Vec<f32>>,
    token_embedding: CpuMatrix,
    output: CpuMatrix,
    output_bias: Option<Vec<f32>>,
    output_norm: Tensor,
    layers: Vec<LayerWeights>,
    tokenizer: Option<LlamaTokenizer>,
    use_quantized: bool,
}

impl LlamaCpuModel {
    fn load(model: &LlamaModel, use_quantized: bool) -> Result<Self, LlamaError> {
        let config = model.config.clone();
        let rope_freq_factors = model.rope_freq_factors.clone();
        let tokenizer = match model.tokenizer() {
            Ok(tokenizer) => Some(tokenizer),
            Err(LlamaError::MissingMetadata("tokenizer.ggml.tokens")) => None,
            Err(error) => return Err(error),
        };
        if !use_quantized {
            return Self::load_f32_weights(model, config, rope_freq_factors, tokenizer);
        }
        Self::load_quantized_weights(model, config, rope_freq_factors, tokenizer)
    }

    #[allow(clippy::too_many_lines)]
    fn load_quantized_weights(
        model: &LlamaModel,
        config: LlamaConfig,
        rope_freq_factors: Option<Vec<f32>>,
        tokenizer: Option<LlamaTokenizer>,
    ) -> Result<Self, LlamaError> {
        let has_output_weight = model.model.tensor("output.weight").is_some();
        let mut matrix_names = vec!["token_embd.weight".to_owned()];
        if has_output_weight {
            matrix_names.push("output.weight".to_owned());
        }
        let mut vector_names = vec!["output_norm.weight".to_owned()];
        let has_output_bias = model.model.tensor("output.bias").is_some();
        if has_output_bias {
            vector_names.push("output.bias".to_owned());
        }
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            vector_names.extend([
                format!("{prefix}.attn_norm.weight"),
                format!("{prefix}.ffn_norm.weight"),
            ]);
            for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                let name = format!("{prefix}.{suffix}");
                if model.model.tensor(&name).is_some() {
                    vector_names.push(name);
                }
            }
            for suffix in [
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
                "ffn_gate.bias",
                "ffn_down.bias",
                "ffn_up.bias",
            ] {
                let name = format!("{prefix}.{suffix}");
                if model.model.tensor(&name).is_some() {
                    vector_names.push(name);
                }
            }
            matrix_names.extend([
                format!("{prefix}.attn_q.weight"),
                format!("{prefix}.attn_k.weight"),
                format!("{prefix}.attn_v.weight"),
                format!("{prefix}.attn_output.weight"),
                format!("{prefix}.ffn_gate.weight"),
                format!("{prefix}.ffn_down.weight"),
                format!("{prefix}.ffn_up.weight"),
            ]);
        }
        let mut quantized_names = Vec::new();
        let mut f32_names = Vec::new();
        for name in &matrix_names {
            let descriptor = model
                .model
                .tensor(name)
                .ok_or_else(|| LlamaError::MissingTensor(name.clone()))?;
            if descriptor.shape().len() == 2
                && matches!(
                    descriptor.value_type().raw(),
                    2 | 3
                        | 6
                        | 7
                        | 8
                        | 9
                        | 10
                        | 11
                        | 12
                        | 13
                        | 14
                        | 15
                        | 16
                        | 17
                        | 18
                        | 19
                        | 20
                        | 21
                        | 22
                        | 23
                        | 29
                        | 34
                        | 35
                        | 39
                        | 40
                        | 41
                        | 42
                )
            {
                quantized_names.push(name.as_str());
            } else {
                f32_names.push(name.as_str());
            }
        }
        let quantized = model.model.load_quantized_many(&quantized_names)?;
        let f32_matrices = model.model.load_f32_many(&f32_names)?;
        let mut matrices = HashMap::with_capacity(matrix_names.len());
        for (name, matrix) in quantized_names.into_iter().zip(quantized) {
            matrices.insert(name.to_owned(), CpuMatrix::Quantized(matrix));
        }
        for (name, tensor) in f32_names.into_iter().zip(f32_matrices) {
            matrices.insert(name.to_owned(), CpuMatrix::from_tensor(tensor)?);
        }
        let vector_refs = vector_names.iter().map(String::as_str).collect::<Vec<_>>();
        let vector_values = model.model.load_f32_many(&vector_refs)?;
        let mut vectors = vector_names
            .into_iter()
            .zip(vector_values)
            .collect::<HashMap<_, _>>();
        let token_embedding = take_matrix(&mut matrices, "token_embd.weight")?;
        let output = if has_output_weight {
            take_matrix(&mut matrices, "output.weight")?
        } else {
            token_embedding.clone()
        };
        let output_bias = vectors
            .remove("output.bias")
            .map(Tensor::into_data)
            .map(|bias| {
                if bias.len() == config.vocab_size {
                    Ok(bias)
                } else {
                    Err(LlamaError::TensorShape {
                        name: "output.bias".to_owned(),
                        expected: vec![config.vocab_size],
                        actual: vec![bias.len()],
                    })
                }
            })
            .transpose()?;
        let output_norm = take_vector(&mut vectors, "output_norm.weight")?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            layers.push(LayerWeights {
                attn_norm: take_vector(&mut vectors, &format!("{prefix}.attn_norm.weight"))?,
                attn_q: take_matrix(&mut matrices, &format!("{prefix}.attn_q.weight"))?,
                attn_q_norm: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_q_norm.weight"),
                    config.key_length,
                )?,
                attn_q_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_q.bias"),
                    config.head_count * config.key_length,
                )?,
                attn_k: take_matrix(&mut matrices, &format!("{prefix}.attn_k.weight"))?,
                attn_k_norm: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_k_norm.weight"),
                    config.key_length,
                )?,
                attn_k_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_k.bias"),
                    config.head_count_kv * config.value_length,
                )?,
                attn_v: take_matrix(&mut matrices, &format!("{prefix}.attn_v.weight"))?,
                attn_v_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_v.bias"),
                    config.head_count_kv * config.value_length,
                )?,
                attn_output: take_matrix(&mut matrices, &format!("{prefix}.attn_output.weight"))?,
                attn_output_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.attn_output.bias"),
                    config.embedding_length,
                )?,
                ffn_norm: take_vector(&mut vectors, &format!("{prefix}.ffn_norm.weight"))?,
                ffn_gate: take_matrix(&mut matrices, &format!("{prefix}.ffn_gate.weight"))?,
                ffn_gate_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.ffn_gate.bias"),
                    config.feed_forward_length,
                )?,
                ffn_down: take_matrix(&mut matrices, &format!("{prefix}.ffn_down.weight"))?,
                ffn_down_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.ffn_down.bias"),
                    config.embedding_length,
                )?,
                ffn_up: take_matrix(&mut matrices, &format!("{prefix}.ffn_up.weight"))?,
                ffn_up_bias: take_optional_bias(
                    &mut vectors,
                    &format!("{prefix}.ffn_up.bias"),
                    config.feed_forward_length,
                )?,
            });
        }
        Ok(Self {
            config,
            rope_freq_factors,
            token_embedding,
            output,
            output_bias,
            output_norm,
            layers,
            tokenizer,
            use_quantized: true,
        })
    }

    #[allow(clippy::too_many_lines)]
    fn load_f32_weights(
        model: &LlamaModel,
        config: LlamaConfig,
        rope_freq_factors: Option<Vec<f32>>,
        tokenizer: Option<LlamaTokenizer>,
    ) -> Result<Self, LlamaError> {
        let has_output_weight = model.model.tensor("output.weight").is_some();
        let mut names = vec!["token_embd.weight".to_owned()];
        if has_output_weight {
            names.push("output.weight".to_owned());
        }
        names.push("output_norm.weight".to_owned());
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
            for suffix in ["attn_q_norm.weight", "attn_k_norm.weight"] {
                let name = format!("{prefix}.{suffix}");
                if model.model.tensor(&name).is_some() {
                    names.push(name);
                }
            }
            for suffix in [
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
                "ffn_gate.bias",
                "ffn_down.bias",
                "ffn_up.bias",
            ] {
                let name = format!("{prefix}.{suffix}");
                if model.model.tensor(&name).is_some() {
                    names.push(name);
                }
            }
        }
        let has_output_bias = model.model.tensor("output.bias").is_some();
        if has_output_bias {
            names.push("output.bias".to_owned());
        }
        let name_refs = names.iter().map(String::as_str).collect::<Vec<_>>();
        let mut loaded = HashMap::with_capacity(names.len());
        model.model.for_each_f32(&name_refs, |name, tensor| {
            loaded.insert(name.to_owned(), tensor);
            Ok::<(), LlamaError>(())
        })?;
        let token_embedding =
            CpuMatrix::from_tensor(loaded.remove("token_embd.weight").ok_or_else(|| {
                LlamaError::Tensor("GGUF loader did not return token_embd.weight".to_owned())
            })?)?;
        let output = if has_output_weight {
            CpuMatrix::from_tensor(loaded.remove("output.weight").ok_or_else(|| {
                LlamaError::Tensor("GGUF loader did not return output.weight".to_owned())
            })?)?
        } else {
            token_embedding.clone()
        };
        let output_bias = loaded.remove("output.bias").map(Tensor::into_data);
        if let Some(bias) = &output_bias
            && bias.len() != config.vocab_size
        {
            return Err(LlamaError::TensorShape {
                name: "output.bias".to_owned(),
                expected: vec![config.vocab_size],
                actual: vec![bias.len()],
            });
        }
        let output_norm = loaded.remove("output_norm.weight").ok_or_else(|| {
            LlamaError::Tensor("GGUF loader did not return output_norm.weight".to_owned())
        })?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            let prefix = format!("blk.{layer}");
            layers.push(LayerWeights {
                attn_norm: loaded
                    .remove(&format!("{prefix}.attn_norm.weight"))
                    .ok_or_else(|| {
                        LlamaError::Tensor(format!(
                            "GGUF loader did not return {prefix}.attn_norm.weight"
                        ))
                    })?,
                attn_q: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.attn_q.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.attn_q.weight"
                            ))
                        })?,
                )?,
                attn_q_norm: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_q_norm.weight"),
                    config.key_length,
                )?,
                attn_q_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_q.bias"),
                    config.head_count * config.key_length,
                )?,
                attn_k: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.attn_k.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.attn_k.weight"
                            ))
                        })?,
                )?,
                attn_k_norm: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_k_norm.weight"),
                    config.key_length,
                )?,
                attn_k_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_k.bias"),
                    config.head_count_kv * config.value_length,
                )?,
                attn_v: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.attn_v.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.attn_v.weight"
                            ))
                        })?,
                )?,
                attn_v_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_v.bias"),
                    config.head_count_kv * config.value_length,
                )?,
                attn_output: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.attn_output.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.attn_output.weight"
                            ))
                        })?,
                )?,
                attn_output_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.attn_output.bias"),
                    config.embedding_length,
                )?,
                ffn_norm: loaded
                    .remove(&format!("{prefix}.ffn_norm.weight"))
                    .ok_or_else(|| {
                        LlamaError::Tensor(format!(
                            "GGUF loader did not return {prefix}.ffn_norm.weight"
                        ))
                    })?,
                ffn_gate: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.ffn_gate.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.ffn_gate.weight"
                            ))
                        })?,
                )?,
                ffn_gate_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.ffn_gate.bias"),
                    config.feed_forward_length,
                )?,
                ffn_down: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.ffn_down.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.ffn_down.weight"
                            ))
                        })?,
                )?,
                ffn_down_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.ffn_down.bias"),
                    config.embedding_length,
                )?,
                ffn_up: CpuMatrix::from_tensor(
                    loaded
                        .remove(&format!("{prefix}.ffn_up.weight"))
                        .ok_or_else(|| {
                            LlamaError::Tensor(format!(
                                "GGUF loader did not return {prefix}.ffn_up.weight"
                            ))
                        })?,
                )?,
                ffn_up_bias: take_optional_loaded_bias(
                    &mut loaded,
                    &format!("{prefix}.ffn_up.bias"),
                    config.feed_forward_length,
                )?,
            });
        }
        Ok(Self {
            config,
            rope_freq_factors,
            token_embedding,
            output,
            output_bias,
            output_norm,
            layers,
            tokenizer,
            use_quantized: false,
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

/// Bounded ring storage for one decoder layer's key/value rows.
#[derive(Debug, Clone)]
struct LayerKvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    capacity: usize,
    kv_width: usize,
    start_position: usize,
    length: usize,
}

impl LayerKvCache {
    fn new(capacity: usize, kv_width: usize) -> Result<Self, LlamaError> {
        if capacity == 0 || kv_width == 0 {
            return Err(LlamaError::InvalidConfig(
                "KV cache dimensions must be greater than zero".to_owned(),
            ));
        }
        Ok(Self {
            keys: Vec::new(),
            values: Vec::new(),
            capacity,
            kv_width,
            start_position: 0,
            length: 0,
        })
    }

    fn end_position(&self) -> usize {
        self.start_position + self.length
    }

    fn physical_offset(&self, position: usize) -> usize {
        (position % self.capacity) * self.kv_width
    }

    fn append(&mut self, position: usize, keys: &[f32], values: &[f32]) -> Result<(), LlamaError> {
        if keys.len() != self.kv_width || values.len() != self.kv_width {
            return Err(LlamaError::Tensor(
                "KV cache row width does not match configuration".to_owned(),
            ));
        }
        if keys.iter().chain(values).any(|value| !value.is_finite()) {
            return Err(LlamaError::Tensor(
                "KV cache rows must contain finite values".to_owned(),
            ));
        }
        if position != self.end_position() {
            return Err(LlamaError::InvalidConfig(format!(
                "KV cache position {position} is not the next position {}",
                self.end_position()
            )));
        }
        let total_elements = self
            .capacity
            .checked_mul(self.kv_width)
            .ok_or_else(|| LlamaError::InvalidConfig("KV cache size overflows usize".to_owned()))?;
        if self.keys.is_empty() {
            self.keys
                .try_reserve_exact(total_elements)
                .map_err(|error| {
                    LlamaError::Tensor(format!("could not allocate KV key cache: {error}"))
                })?;
            self.keys.resize(total_elements, 0.0);
            self.values
                .try_reserve_exact(total_elements)
                .map_err(|error| {
                    LlamaError::Tensor(format!("could not allocate KV value cache: {error}"))
                })?;
            self.values.resize(total_elements, 0.0);
        }
        let offset = self.physical_offset(position);
        self.keys[offset..offset + self.kv_width].copy_from_slice(keys);
        self.values[offset..offset + self.kv_width].copy_from_slice(values);
        if self.length == self.capacity {
            self.start_position += 1;
        } else {
            self.length += 1;
        }
        Ok(())
    }

    fn row_offset(&self, position: usize) -> Option<usize> {
        (position >= self.start_position && position < self.end_position())
            .then(|| self.physical_offset(position))
    }
}

/// Per-layer key/value storage for incremental decoding.
#[derive(Debug, Clone)]
pub struct LlamaKvCache {
    layers: Vec<LayerKvCache>,
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
            .head_count_kv
            .checked_mul(config.value_length)
            .ok_or_else(|| {
                LlamaError::InvalidConfig(
                    "KV cache width overflows the host address space".to_owned(),
                )
            })?;
        let mut layers = Vec::with_capacity(config.block_count);
        for layer in 0..config.block_count {
            layers.push(LayerKvCache::new(
                config.kv_cache_capacity_for_layer(layer),
                kv_width,
            )?);
        }
        Ok(Self {
            layers,
            capacity: config.context_length,
            kv_width,
        })
    }

    /// Returns the number of tokens currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.length)
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

    /// Returns the number of rows currently retained for one layer.
    #[must_use]
    pub fn layer_len(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(|cache| cache.length)
    }

    /// Returns the absolute position of the oldest retained row for one layer.
    #[must_use]
    pub fn layer_start_position(&self, layer: usize) -> Option<usize> {
        self.layers.get(layer).map(|cache| cache.start_position)
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
    #[allow(clippy::cast_precision_loss)]
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
        let head_dim = self.model.config.key_length;
        let attention_width = self.model.config.head_count * self.model.config.value_length;
        #[allow(clippy::cast_precision_loss)]
        let attention_scale = (head_dim as f32).sqrt().recip()
            * self
                .model
                .config
                .attention_temperature_multiplier(self.position);
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
            let mut value_values = value.into_data();
            add_projection_bias(
                &mut query_values,
                layer.attn_q_bias.as_deref(),
                "attention query",
            )?;
            add_projection_bias(
                &mut key_values,
                layer.attn_k_bias.as_deref(),
                "attention key",
            )?;
            add_projection_bias(
                &mut value_values,
                layer.attn_v_bias.as_deref(),
                "attention value",
            )?;
            apply_projection_rms_norm(
                &mut query_values,
                layer.attn_q_norm.as_deref(),
                self.model.config.head_count,
                head_dim,
                self.model.config.rms_norm_epsilon,
                "attention query",
            )?;
            apply_projection_rms_norm(
                &mut key_values,
                layer.attn_k_norm.as_deref(),
                self.model.config.head_count_kv,
                head_dim,
                self.model.config.rms_norm_epsilon,
                "attention key",
            )?;
            apply_rope_with_scaling_and_factors(
                &mut query_values,
                self.model.config.head_count,
                head_dim,
                self.model.config.rope_dimension_count,
                self.position as f32,
                self.model.config.rope_freq_base,
                self.model.config.rope_scaling,
                self.model.rope_freq_factors.as_deref(),
            )?;
            apply_rope_with_scaling_and_factors(
                &mut key_values,
                self.model.config.head_count_kv,
                head_dim,
                self.model.config.rope_dimension_count,
                self.position as f32,
                self.model.config.rope_freq_base,
                self.model.config.rope_scaling,
                self.model.rope_freq_factors.as_deref(),
            )?;
            if key_values.len() != self.cache.kv_width {
                return Err(LlamaError::Tensor(
                    "key projection width does not match KV cache".to_owned(),
                ));
            }
            let layer_cache = self.cache.layers.get_mut(layer_index).ok_or_else(|| {
                LlamaError::InvalidConfig("KV layer index is out of range".to_owned())
            })?;
            layer_cache.append(self.position, &key_values, &value_values)?;
            let cached_tokens = layer_cache.end_position();
            let attention_start = self
                .model
                .config
                .attention_start_for_layer(layer_index, cached_tokens)
                .max(layer_cache.start_position);
            let retained_tokens = cached_tokens - attention_start;
            let kv_heads = self.model.config.head_count_kv;
            let mut retained_keys = Vec::with_capacity(retained_tokens * self.cache.kv_width);
            let mut retained_values = Vec::with_capacity(retained_tokens * self.cache.kv_width);
            for token_index in attention_start..cached_tokens {
                let row_start = layer_cache
                    .row_offset(token_index)
                    .ok_or_else(|| LlamaError::Tensor("KV cache row is not retained".to_owned()))?;
                for head in 0..kv_heads {
                    let head_start = row_start + head * head_dim;
                    let head_end = head_start + head_dim;
                    retained_keys.extend_from_slice(&layer_cache.keys[head_start..head_end]);
                    retained_values.extend_from_slice(&layer_cache.values[head_start..head_end]);
                }
            }
            let query =
                Tensor::from_data([1, self.model.config.head_count, 1, head_dim], query_values)?;
            let keys = Tensor::from_data([1, kv_heads, retained_tokens, head_dim], retained_keys)?;
            let values = Tensor::from_data(
                [1, kv_heads, retained_tokens, self.model.config.value_length],
                retained_values,
            )?;
            let attended =
                query.scaled_dot_product_attention(&keys, &values, attention_scale, true)?;
            let attended = row_tensor(attention_width, attended.into_data())?;
            let attended = layer.attn_output.matmul_tensor(&attended)?;
            let mut attended = attended.into_data();
            add_projection_bias(
                &mut attended,
                layer.attn_output_bias.as_deref(),
                "attention output",
            )?;
            let attended = row_tensor(attention_width, attended)?;
            hidden = hidden.add(&attended)?;
            let normalized =
                hidden
                    .rms_norm(self.model.config.rms_norm_epsilon)?
                    .mul(&row_tensor(
                        embedding_width,
                        layer.ffn_norm.data().to_vec(),
                    )?)?;
            let mut gate = layer.ffn_gate.matmul_tensor(&normalized)?.into_data();
            add_projection_bias(&mut gate, layer.ffn_gate_bias.as_deref(), "FFN gate")?;
            let gate =
                Tensor::from_data([1, self.model.config.feed_forward_length], gate)?.silu()?;
            let mut up = layer.ffn_up.matmul_tensor(&normalized)?.into_data();
            add_projection_bias(&mut up, layer.ffn_up_bias.as_deref(), "FFN up")?;
            let up = Tensor::from_data([1, self.model.config.feed_forward_length], up)?;
            let mut feed_forward = layer.ffn_down.matmul_tensor(&gate.mul(&up)?)?.into_data();
            add_projection_bias(
                &mut feed_forward,
                layer.ffn_down_bias.as_deref(),
                "FFN down",
            )?;
            let feed_forward = Tensor::from_data([1, embedding_width], feed_forward)?;
            hidden = hidden.add(&feed_forward)?;
        }
        let normalized = hidden
            .rms_norm(self.model.config.rms_norm_epsilon)?
            .mul(&row_tensor(
                embedding_width,
                self.model.output_norm.data().to_vec(),
            )?)?;
        let mut logits = self.model.output.matmul_tensor(&normalized)?.into_data();
        if let Some(output_bias) = &self.model.output_bias {
            for (logit, bias) in logits.iter_mut().zip(output_bias) {
                *logit += *bias;
            }
        }
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
        let mut logits = None;
        for &token_id in prompt_ids {
            logits = Some(self.forward_token(token_id)?);
        }
        let mut logits = if let Some(logits) = logits {
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

#[cfg(test)]
fn apply_rope(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    rope_dimension_count: usize,
    position: f32,
    frequency_base: f32,
) -> Result<(), LlamaError> {
    apply_rope_with_scaling(
        values,
        head_count,
        head_dim,
        rope_dimension_count,
        position,
        frequency_base,
        LlamaRopeScaling::None,
    )
}

#[cfg(test)]
fn apply_rope_with_scaling(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    rope_dimension_count: usize,
    position: f32,
    frequency_base: f32,
    scaling: LlamaRopeScaling,
) -> Result<(), LlamaError> {
    apply_rope_with_scaling_and_factors(
        values,
        head_count,
        head_dim,
        rope_dimension_count,
        position,
        frequency_base,
        scaling,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_rope_with_scaling_and_factors(
    values: &mut [f32],
    head_count: usize,
    head_dim: usize,
    rope_dimension_count: usize,
    position: f32,
    frequency_base: f32,
    scaling: LlamaRopeScaling,
    frequency_factors: Option<&[f32]>,
) -> Result<(), LlamaError> {
    let pair_count = rope_dimension_count / 2;
    if let Some(factors) = frequency_factors {
        if factors.len() != pair_count {
            return Err(LlamaError::Tensor(format!(
                "rotary frequency factor count {} does not match {pair_count} pairs",
                factors.len()
            )));
        }
        if factors
            .iter()
            .any(|factor| !factor.is_finite() || *factor <= 0.0)
        {
            return Err(LlamaError::Tensor(
                "rotary frequency factors must be finite and positive".to_owned(),
            ));
        }
    }
    let head_width = head_dim;
    #[allow(clippy::cast_precision_loss)]
    let head_dim = head_width as f32;
    for head in 0..head_count {
        let start = head * head_width;
        for pair in 0..pair_count {
            let frequency_factor = frequency_factors.map_or(1.0, |factors| factors[pair]);
            let (angle, magnitude) = scaling.phase(
                position,
                pair,
                head_dim,
                rope_dimension_count,
                frequency_base,
                frequency_factor,
            );
            let (sine, cosine) = angle.sin_cos();
            let first = values[start + pair * 2];
            let second = values[start + pair * 2 + 1];
            let rotated_first = (first * cosine - second * sine) * magnitude;
            let rotated_second = (first * sine + second * cosine) * magnitude;
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

fn parse_rope_scaling(
    kind: Option<MetadataScalar>,
    factor: Option<f32>,
) -> Result<LlamaRopeScaling, LlamaError> {
    match (kind, factor) {
        (None, None) => Ok(LlamaRopeScaling::None),
        (None, Some(_)) => Err(LlamaError::InvalidMetadata {
            key: "llama.rope.scaling.type",
            value: "scaling factor is present without a scaling type".to_owned(),
        }),
        (Some(MetadataScalar::String(kind)), factor) => match kind.as_str() {
            "none" => {
                if let Some(factor) = factor
                    && (factor - 1.0).abs() > f32::EPSILON
                {
                    return Err(LlamaError::InvalidMetadata {
                        key: "llama.rope.scaling.factor",
                        value: format!("none scaling requires factor 1.0, got {factor}"),
                    });
                }
                Ok(LlamaRopeScaling::None)
            }
            "linear" => {
                let factor =
                    factor.ok_or(LlamaError::MissingMetadata("llama.rope.scaling.factor"))?;
                let scaling = LlamaRopeScaling::Linear { factor };
                scaling.validate().map_err(|error| match error {
                    LlamaError::InvalidConfig(value) => LlamaError::InvalidMetadata {
                        key: "llama.rope.scaling.factor",
                        value,
                    },
                    other => other,
                })?;
                Ok(scaling)
            }
            _ => Err(LlamaError::InvalidMetadata {
                key: "llama.rope.scaling.type",
                value: format!("unsupported scaling type {kind:?}"),
            }),
        },
        (Some(value), _) => Err(LlamaError::InvalidMetadata {
            key: "llama.rope.scaling.type",
            value: format!("expected a string, got {value:?}"),
        }),
    }
}

fn parse_model_rope_scaling<S>(
    source: &S,
    keys: &LlamaMetadataKeys,
    kind: Option<MetadataScalar>,
    factor: Option<f32>,
    context_length: usize,
) -> Result<LlamaRopeScaling, LlamaError>
where
    S: MetadataSource + ?Sized,
{
    let is_yarn = matches!(kind, Some(MetadataScalar::String(ref value)) if value == "yarn");
    if !is_yarn {
        return parse_rope_scaling(kind, factor);
    }
    let factor = factor.ok_or(LlamaError::MissingMetadata(keys.rope_scaling_factor))?;
    let beta_fast = optional_alias_f32(
        source,
        keys.rope_scaling_yarn_beta_fast,
        keys.rope_scaling_beta_fast_legacy,
    )?
    .unwrap_or(32.0);
    let beta_slow = optional_alias_f32(
        source,
        keys.rope_scaling_yarn_beta_slow,
        keys.rope_scaling_beta_slow_legacy,
    )?
    .unwrap_or(1.0);
    let original_context_length = optional_usize_value(
        source.metadata_scalar(keys.rope_scaling_original_context_length)?,
        keys.rope_scaling_original_context_length,
    )?
    .unwrap_or(context_length);
    let log_multiplier = optional_alias_f32(
        source,
        keys.rope_scaling_yarn_log_multiplier,
        keys.rope_scaling_mscale_all_dim_legacy,
    )?
    .unwrap_or(0.0);
    let ext_factor = optional_f32_value(
        source.metadata_scalar(keys.rope_scaling_yarn_ext_factor)?,
        keys.rope_scaling_yarn_ext_factor,
    )?
    .unwrap_or(1.0);
    let rope_attention_factor = optional_f32_value(
        source.metadata_scalar(keys.rope_scaling_attn_factor)?,
        keys.rope_scaling_attn_factor,
    )?
    .unwrap_or(1.0);
    let configured_yarn_attention_factor = optional_f32_value(
        source.metadata_scalar(keys.rope_scaling_yarn_attn_factor)?,
        keys.rope_scaling_yarn_attn_factor,
    )?;
    let get_mscale = |scale: f32, multiplier: f32| {
        if scale <= 1.0 {
            1.0
        } else {
            0.1 * multiplier * scale.ln() + 1.0
        }
    };
    let mut attention_factor = configured_yarn_attention_factor.unwrap_or_else(|| {
        let mut inferred = if log_multiplier == 0.0 {
            get_mscale(factor, 1.0)
        } else {
            get_mscale(factor, 1.0) / get_mscale(factor, log_multiplier)
        };
        if ext_factor != 0.0 {
            inferred *= 1.0 / (1.0 + 0.1 * factor.ln());
        }
        inferred
    });
    attention_factor *= rope_attention_factor;
    let scaling = LlamaRopeScaling::Yarn {
        factor,
        beta_fast,
        beta_slow,
        original_context_length,
        attention_factor,
        ext_factor,
    };
    scaling.validate().map_err(|error| match error {
        LlamaError::InvalidConfig(value) => LlamaError::InvalidMetadata {
            key: keys.rope_scaling_factor,
            value,
        },
        other => other,
    })?;
    Ok(scaling)
}

fn optional_alias_f32<S>(
    source: &S,
    primary_key: &'static str,
    alias_key: &'static str,
) -> Result<Option<f32>, LlamaError>
where
    S: MetadataSource + ?Sized,
{
    optional_f32_value(
        source
            .metadata_scalar(primary_key)?
            .or(source.metadata_scalar(alias_key)?),
        primary_key,
    )
}

#[allow(clippy::cast_precision_loss)]
fn rope_yarn_correction_dim(
    rotations: f32,
    rotary_dimension_count: usize,
    frequency_base: f32,
    original_context_length: usize,
) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let dimension = rotary_dimension_count as f32;
    dimension * ((original_context_length as f32) / (rotations * 2.0 * std::f32::consts::PI)).ln()
        / (2.0 * frequency_base.ln())
}

fn rope_yarn_ramp(low: f32, high: f32, pair_index: usize) -> f32 {
    #[allow(clippy::cast_precision_loss)]
    let value = (pair_index as f32 - low) / (0.001_f32.max(high - low));
    1.0 - value.clamp(0.0, 1.0)
}

fn parse_attention_window_pattern<S>(
    source: &S,
    key: &'static str,
    block_count: usize,
) -> Result<Option<Vec<bool>>, LlamaError>
where
    S: MetadataSource + ?Sized,
{
    match source.metadata_scalar(key) {
        Ok(Some(value)) => {
            let period =
                as_usize(value).map_err(|value| LlamaError::InvalidMetadata { key, value })?;
            let pattern = (0..block_count)
                .map(|layer| period == 0 || layer % period < period.saturating_sub(1))
                .collect();
            Ok(Some(pattern))
        }
        Ok(None) => Ok(None),
        Err(ModelError::MetadataArray(_)) => source
            .metadata_bool_array(key, MAX_CONFIG_ARRAY_ELEMENTS)
            .map_err(LlamaError::from),
        Err(error) => Err(LlamaError::from(error)),
    }
}

fn take_matrix(
    matrices: &mut HashMap<String, CpuMatrix>,
    name: &str,
) -> Result<CpuMatrix, LlamaError> {
    matrices
        .remove(name)
        .ok_or_else(|| LlamaError::Tensor(format!("GGUF loader did not return {name}")))
}

fn take_vector(vectors: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor, LlamaError> {
    vectors
        .remove(name)
        .ok_or_else(|| LlamaError::Tensor(format!("GGUF loader did not return {name}")))
}

fn take_optional_bias(
    vectors: &mut HashMap<String, Tensor>,
    name: &str,
    width: usize,
) -> Result<Option<Vec<f32>>, LlamaError> {
    vectors
        .remove(name)
        .map(|tensor| {
            let data = tensor.into_data();
            if data.len() != width {
                return Err(LlamaError::TensorShape {
                    name: name.to_owned(),
                    expected: vec![width],
                    actual: vec![data.len()],
                });
            }
            Ok(data)
        })
        .transpose()
}

fn take_optional_loaded_bias(
    loaded: &mut HashMap<String, Tensor>,
    name: &str,
    width: usize,
) -> Result<Option<Vec<f32>>, LlamaError> {
    take_optional_bias(loaded, name, width)
}

fn add_projection_bias(
    values: &mut [f32],
    bias: Option<&[f32]>,
    operation: &str,
) -> Result<(), LlamaError> {
    if let Some(bias) = bias {
        if values.len() != bias.len() {
            return Err(LlamaError::Tensor(format!(
                "{operation} bias width {} does not match projection width {}",
                bias.len(),
                values.len()
            )));
        }
        for (value, offset) in values.iter_mut().zip(bias) {
            *value += *offset;
        }
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(LlamaError::Tensor(format!(
            "{operation} projection is not finite after bias"
        )));
    }
    Ok(())
}

fn apply_projection_rms_norm(
    values: &mut [f32],
    weight: Option<&[f32]>,
    head_count: usize,
    head_dim: usize,
    epsilon: f32,
    operation: &str,
) -> Result<(), LlamaError> {
    let Some(weight) = weight else {
        return Ok(());
    };
    if head_dim == 0
        || values.len() != head_count.saturating_mul(head_dim)
        || weight.len() != head_dim
    {
        return Err(LlamaError::Tensor(format!(
            "{operation} RMSNorm shape does not match projection"
        )));
    }
    if !epsilon.is_finite() || epsilon < 0.0 {
        return Err(LlamaError::InvalidConfig(
            "RMSNorm epsilon must be finite and nonnegative".to_owned(),
        ));
    }
    #[allow(clippy::cast_precision_loss)]
    let scale = 1.0 / head_dim as f32;
    for row in values.chunks_exact_mut(head_dim) {
        let mean_square = row.iter().map(|value| value * value).sum::<f32>() * scale;
        let denominator = (mean_square + epsilon).sqrt();
        if !denominator.is_finite() || denominator == 0.0 {
            return Err(LlamaError::Tensor(format!(
                "{operation} RMSNorm produced a non-finite scale"
            )));
        }
        for (value, factor) in row.iter_mut().zip(weight) {
            *value = *value / denominator * *factor;
            if !value.is_finite() {
                return Err(LlamaError::Tensor(format!(
                    "{operation} RMSNorm produced a non-finite value"
                )));
            }
        }
    }
    Ok(())
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

#[allow(clippy::too_many_lines)]
fn validate_layout(model: &GgufModel, config: &LlamaConfig) -> Result<(), LlamaError> {
    let architecture = model.architecture().unwrap_or_default();
    require_shape(
        model,
        "token_embd.weight",
        &[config.embedding_length, config.vocab_size],
    )?;
    if model.tensor("output.weight").is_some() {
        require_shape(
            model,
            "output.weight",
            &[config.embedding_length, config.vocab_size],
        )?;
    }
    if let Some(output_bias) = model.tensor("output.bias")
        && output_bias.shape() != [config.vocab_size]
    {
        return Err(LlamaError::TensorShape {
            name: "output.bias".to_owned(),
            expected: vec![config.vocab_size],
            actual: output_bias.shape().to_vec(),
        });
    }
    if matches!(architecture, "qwen2" | "qwen3")
        && config.rope_dimension_count() != config.key_length()
    {
        return Err(LlamaError::InvalidConfig(
            "qwen2 and qwen3 require rotary embeddings across the full attention head".to_owned(),
        ));
    }
    require_shape(model, "output_norm.weight", &[config.embedding_length])?;
    let query_width = config
        .head_count
        .checked_mul(config.key_length)
        .ok_or_else(|| {
            LlamaError::InvalidConfig(
                "query projection width overflows the host address space".to_owned(),
            )
        })?;
    let kv_width = config
        .head_count_kv
        .checked_mul(config.value_length)
        .ok_or_else(|| {
            LlamaError::InvalidConfig(
                "key/value projection width overflows the host address space".to_owned(),
            )
        })?;
    for layer in 0..config.block_count {
        let prefix = format!("blk.{layer}");
        for (suffix, shape) in [
            ("attn_norm.weight", vec![config.embedding_length]),
            ("attn_q.weight", vec![config.embedding_length, query_width]),
            ("attn_k.weight", vec![config.embedding_length, kv_width]),
            ("attn_v.weight", vec![config.embedding_length, kv_width]),
            (
                "attn_output.weight",
                vec![
                    config.value_length * config.head_count,
                    config.embedding_length,
                ],
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
        if architecture == "qwen3" {
            require_shape(
                model,
                &format!("{prefix}.attn_q_norm.weight"),
                &[config.key_length],
            )?;
            require_shape(
                model,
                &format!("{prefix}.attn_k_norm.weight"),
                &[config.key_length],
            )?;
        }
        for (suffix, width) in [
            ("attn_q.bias", query_width),
            ("attn_k.bias", kv_width),
            ("attn_v.bias", kv_width),
            ("attn_output.bias", config.embedding_length),
        ] {
            let name = format!("{prefix}.{suffix}");
            if model.tensor(&name).is_some() {
                require_shape(model, &name, &[width])?;
            }
        }
        let name = format!("{prefix}.attn_qkv.bias");
        if model.tensor(&name).is_some() {
            return Err(LlamaError::InvalidConfig(format!(
                "unsupported decoder bias tensor {name}"
            )));
        }
        for (suffix, width) in [
            ("ffn_gate.bias", config.feed_forward_length),
            ("ffn_down.bias", config.embedding_length),
            ("ffn_up.bias", config.feed_forward_length),
        ] {
            let name = format!("{prefix}.{suffix}");
            if model.tensor(&name).is_some() {
                if architecture == "mistral3" {
                    require_shape(model, &name, &[width])?;
                } else {
                    return Err(LlamaError::InvalidConfig(format!(
                        "unsupported decoder bias tensor {name}"
                    )));
                }
            }
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

    fn push_string_metadata(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(bytes, value);
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
        llama_fixture_for("llama")
    }

    #[allow(clippy::too_many_lines)]
    fn llama_fixture_for(architecture: &str) -> Vec<u8> {
        llama_fixture_for_output(architecture, true)
    }

    #[allow(clippy::too_many_lines)]
    fn llama_fixture_for_output(architecture: &str, has_output_weight: bool) -> Vec<u8> {
        llama_fixture_with_additions(architecture, has_output_weight, &[], &[], &[])
    }

    #[allow(clippy::too_many_lines)]
    fn llama_fixture_with_additions<'a>(
        architecture: &str,
        has_output_weight: bool,
        extra_tensors: &[(&'a str, &'a [u64])],
        extra_u32_metadata: &[(&str, u32)],
        extra_string_array_metadata: &[(&str, &[&str])],
    ) -> Vec<u8> {
        llama_fixture_with_encoded_additions(
            architecture,
            has_output_weight,
            extra_tensors,
            extra_u32_metadata,
            extra_string_array_metadata,
            &[],
        )
    }

    #[allow(clippy::too_many_lines)]
    fn llama_fixture_with_encoded_additions<'a>(
        architecture: &str,
        has_output_weight: bool,
        extra_tensors: &[(&'a str, &'a [u64])],
        extra_u32_metadata: &[(&str, u32)],
        extra_string_array_metadata: &[(&str, &[&str])],
        extra_tensor_encodings: &[(&str, u32, &[u8])],
    ) -> Vec<u8> {
        let mut config = vec![
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
        if !has_output_weight {
            config.remove(1);
        }
        if architecture == "qwen3" {
            config.extend([
                ("blk.0.attn_q_norm.weight", vec![2]),
                ("blk.0.attn_k_norm.weight", vec![2]),
            ]);
        }
        if architecture == "mistral3" {
            config.extend([
                ("blk.0.ffn_gate.bias", vec![8]),
                ("blk.0.ffn_down.bias", vec![4]),
                ("blk.0.ffn_up.bias", vec![8]),
            ]);
        }
        config.extend(
            extra_tensors
                .iter()
                .map(|(name, shape)| (*name, shape.to_vec())),
        );
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&(config.len() as u64).to_le_bytes());
        let base_metadata_count: u64 = if architecture == "mistral3" { 19 } else { 13 };
        let extra_metadata_count =
            u64::try_from(extra_u32_metadata.len() + extra_string_array_metadata.len()).unwrap();
        let metadata_count = base_metadata_count + extra_metadata_count;
        bytes.extend_from_slice(&metadata_count.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, architecture);
        push_u32_metadata(&mut bytes, &format!("{architecture}.context_length"), 16);
        push_u32_metadata(&mut bytes, &format!("{architecture}.embedding_length"), 4);
        push_u32_metadata(&mut bytes, &format!("{architecture}.block_count"), 1);
        push_u32_metadata(
            &mut bytes,
            &format!("{architecture}.attention.head_count"),
            2,
        );
        push_u32_metadata(
            &mut bytes,
            &format!("{architecture}.attention.head_count_kv"),
            1,
        );
        push_u32_metadata(
            &mut bytes,
            &format!("{architecture}.feed_forward_length"),
            8,
        );
        push_u32_metadata(&mut bytes, &format!("{architecture}.vocab_size"), 8);
        push_f32_metadata(
            &mut bytes,
            &format!("{architecture}.attention.layer_norm_rms_epsilon"),
            1.0e-5,
        );
        push_f32_metadata(
            &mut bytes,
            &format!("{architecture}.rope.freq_base"),
            10_000.0,
        );
        if architecture == "mistral3" {
            push_string_metadata(&mut bytes, "mistral3.rope.scaling.type", "yarn");
            push_f32_metadata(&mut bytes, "mistral3.rope.scaling.factor", 4.0);
            push_f32_metadata(&mut bytes, "mistral3.rope.scaling.yarn_beta_fast", 32.0);
            push_f32_metadata(&mut bytes, "mistral3.rope.scaling.yarn_beta_slow", 1.0);
            push_u32_metadata(
                &mut bytes,
                "mistral3.rope.scaling.original_context_length",
                8,
            );
            push_f32_metadata(&mut bytes, "mistral3.attention.temperature_scale", 0.1);
        }
        for (key, value) in extra_u32_metadata {
            push_u32_metadata(&mut bytes, key, *value);
        }
        for (key, values) in extra_string_array_metadata {
            push_string_array_metadata(&mut bytes, key, values);
        }
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
            let value_type = extra_tensor_encodings
                .iter()
                .find_map(|(encoded_name, value_type, _)| {
                    (*encoded_name == *name).then_some(*value_type)
                })
                .unwrap_or(0);
            bytes.extend_from_slice(&value_type.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let elements = shape.iter().product::<u64>();
            let byte_len = match value_type {
                0 => elements * 4,
                1 => elements * 2,
                _ => panic!("unsupported fixture tensor type {value_type}"),
            };
            offset += byte_len.div_ceil(32) * 32;
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.resize(data_start + usize::try_from(offset).unwrap(), 0);
        let mut data_offset = 0_usize;
        for (name, shape) in &config {
            let encoding = extra_tensor_encodings
                .iter()
                .find(|(encoded_name, _, _)| *encoded_name == *name);
            let value_type = encoding.map_or(0, |(_, value_type, _)| *value_type);
            let elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
            let byte_len = match value_type {
                0 => elements * 4,
                1 => elements * 2,
                _ => panic!("unsupported fixture tensor type {value_type}"),
            };
            if let Some((_, _, encoded)) = encoding {
                assert_eq!(encoded.len(), byte_len);
                bytes[data_start + data_offset..data_start + data_offset + byte_len]
                    .copy_from_slice(encoded);
            }
            data_offset += byte_len.div_ceil(32) * 32;
        }
        bytes
    }

    fn qwen2_tied_fixture() -> Vec<u8> {
        let config = [
            ("token_embd.weight", vec![4_u64, 8]),
            ("output.bias", vec![8]),
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
        bytes.extend_from_slice(&14_u64.to_le_bytes());
        push_string_metadata(&mut bytes, "general.architecture", "qwen2");
        push_u32_metadata(&mut bytes, "qwen2.context_length", 16);
        push_u32_metadata(&mut bytes, "qwen2.embedding_length", 4);
        push_u32_metadata(&mut bytes, "qwen2.block_count", 1);
        push_u32_metadata(&mut bytes, "qwen2.attention.head_count", 2);
        push_u32_metadata(&mut bytes, "qwen2.attention.head_count_kv", 1);
        push_u32_metadata(&mut bytes, "qwen2.feed_forward_length", 8);
        push_u32_metadata(&mut bytes, "qwen2.vocab_size", 8);
        push_f32_metadata(&mut bytes, "qwen2.attention.layer_norm_rms_epsilon", 1.0e-5);
        push_f32_metadata(&mut bytes, "qwen2.rope.freq_base", 10_000.0);
        push_string_array_metadata(
            &mut bytes,
            "tokenizer.ggml.tokens",
            &["<unk>", "Ġ", "a", "Ġa", "b", "c", "d", "e"],
        );
        push_f32_array_metadata(
            &mut bytes,
            "tokenizer.ggml.scores",
            &[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        push_string_metadata(&mut bytes, "tokenizer.ggml.model", "gpt2");
        push_string_array_metadata(&mut bytes, "tokenizer.ggml.merges", &["Ġ a"]);
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

    fn llama_quantized_fixture() -> Vec<u8> {
        let config = vec![
            ("token_embd.weight", vec![32_u64, 32], 2_u32),
            ("output.weight", vec![32, 32], 2),
            ("output_norm.weight", vec![32], 0),
            ("blk.0.attn_norm.weight", vec![32], 0),
            ("blk.0.attn_q.weight", vec![32, 32], 2),
            ("blk.0.attn_k.weight", vec![32, 8], 2),
            ("blk.0.attn_v.weight", vec![32, 8], 2),
            ("blk.0.attn_output.weight", vec![32, 32], 2),
            ("blk.0.ffn_norm.weight", vec![32], 0),
            ("blk.0.ffn_gate.weight", vec![32, 64], 2),
            ("blk.0.ffn_down.weight", vec![64, 32], 2),
            ("blk.0.ffn_up.weight", vec![32, 64], 2),
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
        push_u32_metadata(&mut bytes, "llama.embedding_length", 32);
        push_u32_metadata(&mut bytes, "llama.block_count", 1);
        push_u32_metadata(&mut bytes, "llama.attention.head_count", 4);
        push_u32_metadata(&mut bytes, "llama.attention.head_count_kv", 1);
        push_u32_metadata(&mut bytes, "llama.feed_forward_length", 64);
        push_u32_metadata(&mut bytes, "llama.vocab_size", 32);
        push_f32_metadata(&mut bytes, "llama.attention.layer_norm_rms_epsilon", 1.0e-5);
        push_f32_metadata(&mut bytes, "llama.rope.freq_base", 10_000.0);
        push_string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "quantized-fixture");
        let tokens = [
            "<unk>", "▁a", "▁b", "<eos>", "x", "y", "z", "q", "r", "s", "t", "u", "v", "w", "m",
            "n", "o", "p", "i", "j", "k", "l", "c", "d", "e", "f", "g", "h", "aa", "bb", "cc",
            "dd",
        ];
        push_string_array_metadata(&mut bytes, "tokenizer.ggml.tokens", &tokens);
        push_f32_array_metadata(&mut bytes, "tokenizer.ggml.scores", &[0.0; 32]);
        let mut offset = 0_u64;
        for (name, shape, value_type) in &config {
            push_string(&mut bytes, name);
            bytes.extend_from_slice(&u32::try_from(shape.len()).unwrap().to_le_bytes());
            for dimension in shape {
                bytes.extend_from_slice(&dimension.to_le_bytes());
            }
            bytes.extend_from_slice(&value_type.to_le_bytes());
            bytes.extend_from_slice(&offset.to_le_bytes());
            let elements = shape.iter().product::<u64>();
            let byte_len = if *value_type == 2 {
                elements / 32 * 18
            } else {
                elements * 4
            };
            offset += byte_len.div_ceil(32) * 32;
        }
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        let data_start = bytes.len();
        bytes.resize(data_start + usize::try_from(offset).unwrap(), 0);
        let mut data_offset = 0_usize;
        for (_, shape, value_type) in &config {
            let elements = usize::try_from(shape.iter().product::<u64>()).unwrap();
            let byte_len = if *value_type == 2 {
                elements / 32 * 18
            } else {
                elements * 4
            };
            let start = data_start + data_offset;
            if *value_type == 2 {
                for block in bytes[start..start + byte_len].as_chunks_mut::<18>().0 {
                    block[..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
                    block[2..].fill(0x99);
                }
            } else {
                for value in bytes[start..start + byte_len].as_chunks_mut::<4>().0 {
                    value.copy_from_slice(&1.0_f32.to_le_bytes());
                }
            }
            data_offset += byte_len.div_ceil(32) * 32;
        }
        bytes
    }

    fn write_fixture(bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("llama-runtime-{id}.gguf"));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn assert_invalid_config(bytes: &[u8], expected: &str) {
        let path = write_fixture(bytes);
        let result = LlamaModel::open(&path, 1 << 20);
        match result {
            Err(LlamaError::InvalidConfig(message)) => assert_eq!(message, expected),
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        fs::remove_file(path).unwrap();
    }

    fn assert_unsupported_rotary_tensor(architecture: &str, name: &str) {
        let shape = [1_u64];
        let bytes = llama_fixture_with_additions(architecture, true, &[(name, &shape)], &[], &[]);
        assert_invalid_config(&bytes, &format!("unsupported decoder rotary tensor {name}"));
    }

    fn rope_freq_fixture(
        architecture: &str,
        shape: &[u64],
        value_type: u32,
        encoded: &[u8],
    ) -> Vec<u8> {
        llama_fixture_with_encoded_additions(
            architecture,
            true,
            &[("rope_freqs.weight", shape)],
            &[],
            &[],
            &[("rope_freqs.weight", value_type, encoded)],
        )
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
    fn accepts_a_partial_rotary_dimension() {
        let config =
            LlamaConfig::new_with_rope_dimension(16, 8, 1, 2, 1, 16, 32, 1.0e-5, 10_000.0, 2)
                .unwrap();
        assert_eq!(config.rope_dimension_count(), 2);
        let invalid =
            LlamaConfig::new_with_rope_dimension(16, 8, 1, 2, 1, 16, 32, 1.0e-5, 10_000.0, 3);
        assert!(matches!(invalid, Err(LlamaError::InvalidConfig(_))));
    }

    #[test]
    fn validates_linear_rope_scaling_and_positions() {
        let config = LlamaConfig::new_with_rope_scaling(
            16,
            8,
            1,
            2,
            1,
            16,
            32,
            1.0e-5,
            10_000.0,
            4,
            LlamaRopeScaling::Linear { factor: 2.0 },
        )
        .unwrap();
        assert_eq!(config.rope_scaling().kind(), "linear");
        assert!((config.rope_scaling_factor() - 2.0).abs() < f32::EPSILON);
        assert!((config.scaled_rope_position(7) - 3.5).abs() < f32::EPSILON);
        assert!(
            LlamaConfig::new_with_rope_scaling(
                16,
                8,
                1,
                2,
                1,
                16,
                32,
                1.0e-5,
                10_000.0,
                4,
                LlamaRopeScaling::Linear { factor: 0.0 },
            )
            .is_err()
        );
    }

    #[test]
    fn applies_yarn_phase_mix_and_magnitude_scaling() {
        let scaling = LlamaRopeScaling::Yarn {
            factor: 4.0,
            beta_fast: 32.0,
            beta_slow: 1.0,
            original_context_length: 4096,
            attention_factor: 1.0,
            ext_factor: 1.0,
        };
        assert_eq!(scaling.kind(), "yarn");
        assert!((scaling.factor() - 4.0).abs() < f32::EPSILON);
        scaling.validate().unwrap();
        let (angle, magnitude) = scaling.phase(4096.0, 0, 128.0, 4, 10_000.0, 1.0);
        assert!(angle.is_finite());
        assert!(magnitude.is_finite());
        let mut values = vec![1.0, 0.0, 0.0, 1.0];
        apply_rope_with_scaling(&mut values, 1, 4, 4, 4096.0, 10_000.0, scaling).unwrap();
        assert!(values.iter().all(|value| value.is_finite()));
        assert!((values[0].hypot(values[1]) - magnitude).abs() < 1.0e-4);
    }

    #[test]
    fn applies_attention_temperature_multiplier_by_original_context() {
        let mut config = LlamaConfig::new(16, 8, 1, 2, 1, 16, 32, 1.0e-5, 10_000.0).unwrap();
        config.attention_temperature_scale = 0.1;
        config.attention_temperature_context = 8;
        assert!((config.attention_temperature_multiplier(0) - 1.0).abs() < f32::EPSILON);
        let expected = 1.0 + 0.1 * 3.0_f32.ln();
        assert!((config.attention_temperature_multiplier(16) - expected).abs() < 1.0e-6);
    }

    #[test]
    fn applies_a_bounded_sliding_attention_window() {
        let config = LlamaConfig::new_with_rope_scaling_and_attention_window(
            16,
            8,
            1,
            2,
            1,
            16,
            32,
            1.0e-5,
            10_000.0,
            4,
            LlamaRopeScaling::None,
            Some(4),
        )
        .unwrap();
        assert_eq!(config.attention_window(), Some(4));
        assert_eq!(config.attention_start(3), 0);
        assert_eq!(config.attention_start(7), 3);
        assert!(
            LlamaConfig::new_with_rope_scaling_and_attention_window(
                16,
                8,
                1,
                2,
                1,
                16,
                32,
                1.0e-5,
                10_000.0,
                4,
                LlamaRopeScaling::None,
                Some(0),
            )
            .is_err()
        );
    }

    #[test]
    fn applies_per_layer_sliding_attention_pattern() {
        let config = LlamaConfig::new_with_rope_scaling_and_attention_window_and_pattern(
            16,
            8,
            2,
            2,
            1,
            16,
            32,
            1.0e-5,
            10_000.0,
            4,
            LlamaRopeScaling::None,
            Some(4),
            Some(vec![true, false]),
        )
        .unwrap();
        assert_eq!(config.attention_window_pattern(), Some(&[true, false][..]));
        assert_eq!(config.attention_start_for_layer(0, 7), 3);
        assert_eq!(config.attention_start_for_layer(1, 7), 0);
        assert_eq!(config.kv_cache_capacity_for_layer(0), 4);
        assert_eq!(config.kv_cache_capacity_for_layer(1), 16);
        assert!(
            LlamaConfig::new_with_rope_scaling_and_attention_window_and_pattern(
                16,
                8,
                2,
                2,
                1,
                16,
                32,
                1.0e-5,
                10_000.0,
                4,
                LlamaRopeScaling::None,
                Some(4),
                Some(vec![true]),
            )
            .is_err()
        );
    }

    #[test]
    fn parses_supported_rope_scaling_metadata() {
        assert_eq!(
            parse_rope_scaling(Some(MetadataScalar::String("linear".to_owned())), Some(4.0))
                .unwrap(),
            LlamaRopeScaling::Linear { factor: 4.0 }
        );
        assert_eq!(
            parse_rope_scaling(Some(MetadataScalar::String("none".to_owned())), None).unwrap(),
            LlamaRopeScaling::None
        );
        assert!(
            parse_rope_scaling(Some(MetadataScalar::String("yarn".to_owned())), Some(2.0)).is_err()
        );
    }

    #[test]
    fn partial_rotary_dimension_leaves_the_tail_unchanged() {
        let mut values = vec![1.0, 2.0, 3.0, 4.0];
        apply_rope(&mut values, 1, 4, 2, 1.0, 10_000.0).unwrap();
        assert_eq!(&values[2..], &[3.0, 4.0]);
    }

    #[test]
    fn rope_frequency_factors_match_identity_and_scale_multiple_positions() {
        for position in [1.0_f32, 2.0] {
            let initial = vec![1.0, 0.0, 1.0, 0.0];
            let mut baseline = initial.clone();
            apply_rope_with_scaling(
                &mut baseline,
                1,
                4,
                4,
                position,
                10_000.0,
                LlamaRopeScaling::None,
            )
            .unwrap();
            let mut identity = initial.clone();
            apply_rope_with_scaling_and_factors(
                &mut identity,
                1,
                4,
                4,
                position,
                10_000.0,
                LlamaRopeScaling::None,
                Some(&[1.0, 1.0]),
            )
            .unwrap();
            assert_eq!(identity, baseline);

            let mut scaled = initial;
            apply_rope_with_scaling_and_factors(
                &mut scaled,
                1,
                4,
                4,
                position,
                10_000.0,
                LlamaRopeScaling::None,
                Some(&[2.0, 4.0]),
            )
            .unwrap();
            let first_angle = position / 2.0;
            let second_angle = position * 10_000.0_f32.powf(-0.5) / 4.0;
            let (first_sine, first_cosine) = first_angle.sin_cos();
            let (second_sine, second_cosine) = second_angle.sin_cos();
            let expected = [first_cosine, first_sine, second_cosine, second_sine];
            for (actual, expected) in scaled.iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-7);
            }
            assert_ne!(scaled, baseline);
        }
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
    fn projection_rms_norm_scales_each_attention_head() {
        let mut values = vec![3.0, 4.0, 1.0, 2.0];
        let weight = [2.0, 0.5];
        apply_projection_rms_norm(&mut values, Some(&weight), 2, 2, 0.0, "attention query")
            .unwrap();
        assert!((values[0] - 1.697_056_3).abs() < 1.0e-6);
        assert!((values[1] - 0.565_685_45).abs() < 1.0e-6);
        assert!((values[2] - 1.264_911).abs() < 1.0e-6);
        assert!((values[3] - 0.632_455_5).abs() < 1.0e-6);
    }

    #[test]
    fn opens_and_validates_complete_llama_layout() {
        let path = write_fixture(&llama_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().context_length(), 16);
        assert_eq!(model.config().head_count_kv(), 1);
        assert_eq!(model.model().tensors().len(), 12);
        assert_eq!(model.rope_freq_factors(), None);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_llama_with_output_tied_to_token_embeddings() {
        let path = write_fixture(&llama_fixture_for_output("llama", false));
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert!(model.model().tensor("output.weight").is_none());
        let cpu = model.load_cpu().unwrap();
        assert_eq!(cpu.forward_token(1).unwrap(), vec![0.0; 8]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn opens_mistral3_text_layout_with_canonical_metadata_prefix() {
        let path = write_fixture(&llama_fixture_for("mistral3"));
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().embedding_length(), 4);
        assert_eq!(model.config().rope_scaling().kind(), "yarn");
        assert!((model.config().rope_scaling_factor() - 4.0).abs() < f32::EPSILON);
        assert!((model.config().attention_temperature_scale() - 0.1).abs() < f32::EPSILON);
        assert_eq!(model.config().kv_cache_capacity_for_layer(0), 16);
        let cpu = model.load_cpu().unwrap();
        assert_eq!(cpu.forward_token(1).unwrap().len(), 8);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn accepts_rope_frequency_factors_for_all_admitted_decoders() {
        let encoded = 2.5_f32.to_le_bytes();
        for architecture in ["llama", "qwen2", "qwen3", "mistral", "mistral3"] {
            let bytes = rope_freq_fixture(architecture, &[1], 0, &encoded);
            let path = write_fixture(&bytes);
            let model = LlamaModel::open(&path, 1 << 20).unwrap();
            assert_eq!(model.rope_freq_factors(), Some(&[2.5][..]));
            let session = model.model().read_session().unwrap();
            assert_eq!(
                model.rope_freq_factors_from_session(&session).unwrap(),
                Some(vec![2.5])
            );
            session.verify_unchanged().unwrap();
            let cpu = model.load_cpu().unwrap();
            assert_eq!(cpu.rope_freq_factors.as_deref(), Some(&[2.5][..]));
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn from_model_admits_and_retains_rope_frequency_factors() {
        let encoded = 3.0_f32.to_le_bytes();
        let path = write_fixture(&rope_freq_fixture("llama", &[1], 0, &encoded));
        let indexed = GgufModel::open(&path, 1 << 20).unwrap();
        let config = LlamaConfig::from_model(&indexed).unwrap();
        let model = LlamaModel::from_model(indexed, config).unwrap();
        assert_eq!(model.rope_freq_factors(), Some(&[3.0][..]));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_wrong_rope_frequency_factor_shape() {
        let encoded = [1.0_f32.to_le_bytes(), 2.0_f32.to_le_bytes()].concat();
        let path = write_fixture(&rope_freq_fixture("llama", &[2], 0, &encoded));
        let result = LlamaModel::open(&path, 1 << 20);
        match result {
            Err(LlamaError::TensorShape {
                name,
                expected,
                actual,
            }) => {
                assert_eq!(name, "rope_freqs.weight");
                assert_eq!(expected, [1]);
                assert_eq!(actual, [2]);
            }
            other => panic!("expected TensorShape, got {other:?}"),
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_f32_rope_frequency_factors() {
        let path = write_fixture(&rope_freq_fixture(
            "llama",
            &[1],
            1,
            &1.0_f32.to_le_bytes()[..2],
        ));
        let result = LlamaModel::open(&path, 1 << 20);
        match result {
            Err(LlamaError::InvalidConfig(message)) => {
                assert_eq!(message, "rope_freqs.weight must use F32 storage, got F16");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_nonfinite_or_nonpositive_rope_frequency_factors() {
        for factor in [f32::NAN, f32::INFINITY, 0.0, -1.0] {
            let path = write_fixture(&rope_freq_fixture("llama", &[1], 0, &factor.to_le_bytes()));
            let result = LlamaModel::open(&path, 1 << 20);
            match result {
                Err(LlamaError::InvalidConfig(message)) => assert_eq!(
                    message,
                    "rope_freqs.weight factor at pair 0 must be finite and positive"
                ),
                other => panic!("expected InvalidConfig, got {other:?}"),
            }
            fs::remove_file(path).unwrap();
        }
    }

    #[test]
    fn rejects_llama_long_rope_factors_tensor() {
        assert_unsupported_rotary_tensor("llama", "rope_factors_long.weight");
    }

    #[test]
    fn rejects_llama_short_rope_factors_tensor() {
        assert_unsupported_rotary_tensor("llama", "rope_factors_short.weight");
    }

    #[test]
    fn rejects_mistral3_long_rope_factors_tensor() {
        assert_unsupported_rotary_tensor("mistral3", "rope_factors_long.weight");
    }

    #[test]
    fn rejects_mistral3_short_rope_factors_tensor() {
        assert_unsupported_rotary_tensor("mistral3", "rope_factors_short.weight");
    }

    #[test]
    fn rejects_mistral3_mixture_of_experts_metadata() {
        let bytes = llama_fixture_with_additions(
            "mistral3",
            true,
            &[],
            &[("mistral3.expert_count", 1)],
            &[],
        );
        assert_invalid_config(
            &bytes,
            "unsupported mistral3.expert_count 1; mixture-of-experts execution is unavailable",
        );
    }

    #[test]
    fn accepts_mistral3_zero_expert_count_metadata() {
        let bytes = llama_fixture_with_additions(
            "mistral3",
            true,
            &[],
            &[("mistral3.expert_count", 0)],
            &[],
        );
        let path = write_fixture(&bytes);
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().block_count(), 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn opens_qwen3_standard_layout() {
        let path = write_fixture(&llama_fixture_for("qwen3"));
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().embedding_length(), 4);
        assert_eq!(model.config().head_count_kv(), 1);
        let cpu = model.load_cpu().unwrap();
        assert_eq!(cpu.forward_token(1).unwrap().len(), 8);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_qwen3_classifier_output_tensor() {
        let bytes = llama_fixture_with_additions(
            "qwen3",
            true,
            &[("cls.output.weight", &[4, 2])],
            &[],
            &[],
        );
        assert_invalid_config(
            &bytes,
            "unsupported qwen3 classifier or reranker tensor cls.output.weight",
        );
    }

    #[test]
    fn rejects_qwen3_classifier_output_labels_metadata() {
        let bytes = llama_fixture_with_additions(
            "qwen3",
            true,
            &[],
            &[],
            &[(
                "qwen3.classifier.output_labels",
                &["not_relevant", "relevant"],
            )],
        );
        assert_invalid_config(
            &bytes,
            "unsupported qwen3 classifier or reranker metadata qwen3.classifier.output_labels",
        );
    }

    #[test]
    fn rejects_qwen3_nonzero_pooling_type() {
        let bytes =
            llama_fixture_with_additions("qwen3", true, &[], &[("qwen3.pooling_type", 4)], &[]);
        assert_invalid_config(
            &bytes,
            "unsupported qwen3.pooling_type 4; causal decoding requires zero",
        );
    }

    #[test]
    fn accepts_qwen3_zero_pooling_type() {
        let bytes =
            llama_fixture_with_additions("qwen3", true, &[], &[("qwen3.pooling_type", 0)], &[]);
        let path = write_fixture(&bytes);
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().block_count(), 1);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_qwen2_with_tied_output_and_gpt2_tokenizer() {
        let path = write_fixture(&qwen2_tied_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        assert_eq!(model.config().vocab_size(), 8);
        assert!(model.model().tensor("output.weight").is_none());
        let tokenizer = model.tokenizer().unwrap();
        assert_eq!(tokenizer.encode(" a").unwrap(), [3]);
        let cpu = model.load_cpu().unwrap();
        assert_eq!(cpu.forward_token(1).unwrap(), vec![0.0; 8]);
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
    fn loads_quantized_cpu_model_without_f32_expansion() {
        let path = write_fixture(&llama_quantized_fixture());
        let model = LlamaModel::open(&path, 1 << 20).unwrap();
        let cpu = model.load_cpu_quantized().unwrap();
        assert!(cpu.uses_quantized_weights());
        let logits = cpu.forward_token(1).unwrap();
        assert_eq!(logits.len(), 32);
        assert!(logits.iter().all(|value| value.is_finite()));
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
            kind: TokenizerKind::SentencePiece,
            token_ids: HashMap::new(),
            merge_ranks: HashMap::new(),
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
            kind: TokenizerKind::SentencePiece,
            token_ids: HashMap::new(),
            merge_ranks: HashMap::new(),
        };
        assert_eq!(tokenizer.encode("é").unwrap(), [0, 1, 2]);
        assert_eq!(tokenizer.decode(&[0, 1, 2]).unwrap(), " é");
        assert_eq!(tokenizer.decode(&[1]).unwrap(), "�");
    }

    #[test]
    fn tokenizer_applies_gpt2_bpe_merges_and_round_trips_bytes() {
        let tokens = ["h", "e", "l", "o", "hello"]
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>();
        let token_ids = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), id))
            .collect::<HashMap<_, _>>();
        let merge_ranks = ["h e", "he l", "hel l", "hell o"]
            .into_iter()
            .enumerate()
            .map(|(rank, merge)| {
                let mut pieces = merge.split(' ');
                (
                    bpe_pair_key(pieces.next().unwrap(), pieces.next().unwrap()),
                    rank,
                )
            })
            .collect::<HashMap<_, _>>();
        let tokenizer = LlamaTokenizer {
            tokens,
            scores: vec![0.0; 5],
            bos_token_id: None,
            eos_token_id: None,
            unk_token_id: None,
            kind: TokenizerKind::Gpt2Bpe,
            token_ids,
            merge_ranks,
        };
        assert_eq!(tokenizer.encode("hello").unwrap(), [4]);
        assert_eq!(tokenizer.decode(&[4]).unwrap(), "hello");
    }

    #[test]
    fn tekken_pretokenizer_keeps_digits_atomic_and_contractions_separate() {
        assert_eq!(
            gpt2_pretokenize_tekken("Hello 123 can't"),
            ["Hello", " ", "1", "2", "3", " can", "'t"]
        );
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
    fn sliding_window_cache_evicts_oldest_rows_without_growing() {
        let config = LlamaConfig::new_with_rope_scaling_and_attention_window(
            8,
            8,
            1,
            2,
            1,
            16,
            32,
            1.0e-5,
            10_000.0,
            4,
            LlamaRopeScaling::None,
            Some(2),
        )
        .unwrap();
        let mut cache = LlamaKvCache::new(&config).unwrap();
        let row_a = [1.0, 2.0, 3.0, 4.0];
        let row_b = [5.0, 6.0, 7.0, 8.0];
        let row_c = [9.0, 10.0, 11.0, 12.0];
        cache.layers[0].append(0, &row_a, &row_a).unwrap();
        cache.layers[0].append(1, &row_b, &row_b).unwrap();
        cache.layers[0].append(2, &row_c, &row_c).unwrap();
        let layer = &cache.layers[0];
        assert_eq!(cache.capacity(), 8);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.layer_len(0), Some(2));
        assert_eq!(cache.layer_start_position(0), Some(1));
        assert_eq!(layer.keys[layer.row_offset(1).unwrap()..][..4], row_b);
        assert_eq!(layer.keys[layer.row_offset(2).unwrap()..][..4], row_c);
        assert!(layer.row_offset(0).is_none());
        assert_eq!(layer.keys.len(), 2 * 4);
        assert_eq!(layer.values.len(), 2 * 4);
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
