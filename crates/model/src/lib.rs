#![deny(unsafe_code)]

use std::fmt;
use std::path::{Path, PathBuf};

use ggml_gguf::{Gguf, TensorType};
use ggml_mmap::MappedFile;
use ggml_tensor::Tensor;
use sha2::{Digest, Sha256};

/// Default maximum size for a mapped GGUF model.
pub const DEFAULT_MODEL_BYTE_LIMIT: u64 = 1 << 40;

/// An indexed tensor in a validated GGUF model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorDescriptor {
    name: String,
    shape: Vec<usize>,
    value_type: TensorType,
    byte_offset: u64,
    byte_len: u64,
}

/// An MLX affine-quantized matrix converted directly from an encoded GGUF
/// matrix.
///
/// The matrix is stored in MLX's row-major `[output, input]` orientation. The
/// packed weights use the same little-endian bit layout as `mlx_quantize`,
/// while scales and biases contain one value per input group. Conversion
/// decodes one group at a time and never materializes the complete matrix as
/// F32 values.
#[derive(Debug, Clone, PartialEq)]
pub struct AffineQuantizedMatrix {
    rows: usize,
    columns: usize,
    group_size: usize,
    bits: usize,
    packed: Vec<u32>,
    scales: Vec<f32>,
    biases: Vec<f32>,
}

/// An owned GGUF quantized matrix that can be used without F32 expansion.
///
/// GGML stores rank-2 matrices with the input dimension first and contiguous
/// columns. The encoded block bytes are retained in their original format;
/// callers can read one input column (for embeddings) or multiply a row vector
/// directly against the matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizedMatrix {
    rows: usize,
    columns: usize,
    value_type: TensorType,
    bytes: Vec<u8>,
}

/// Tensor data selected by [`GgufModel::for_each_tensor`].
#[derive(Debug, Clone, PartialEq)]
pub enum LoadedTensor {
    /// A tensor decoded into the checked CPU F32 representation.
    F32(Tensor),
    /// A rank-2 quantized tensor converted directly to MLX affine layout.
    AffineQuantized(AffineQuantizedMatrix),
}

impl AffineQuantizedMatrix {
    /// Number of output rows in the MLX matrix.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of input columns in the MLX matrix.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Number of input values covered by one affine scale and bias.
    #[must_use]
    pub const fn group_size(&self) -> usize {
        self.group_size
    }

    /// Number of bits used for each packed value.
    #[must_use]
    pub const fn bits(&self) -> usize {
        self.bits
    }

    /// Returns packed MLX quantized weights in row-major `[rows, columns * bits / 32]` layout.
    #[must_use]
    pub fn packed(&self) -> &[u32] {
        &self.packed
    }

    /// Returns MLX affine scales in row-major `[rows, columns / group_size]` layout.
    #[must_use]
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Returns MLX affine biases in row-major `[rows, columns / group_size]` layout.
    #[must_use]
    pub fn biases(&self) -> &[f32] {
        &self.biases
    }
}

impl QuantizedMatrix {
    /// Number of input values expected by a row-vector product.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Number of output values produced by a row-vector product.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the GGUF quantization format used by this matrix.
    #[must_use]
    pub const fn value_type(&self) -> TensorType {
        self.value_type
    }

    /// Returns one logical input column, preserving GGML matrix orientation.
    ///
    /// This is used for token embeddings, where the token id selects a column
    /// and the returned values have length [`Self::rows`].
    ///
    /// # Errors
    ///
    /// Returns an error when `column` is outside the matrix or a decoded value
    /// is malformed or non-finite.
    pub fn column(&self, column: usize) -> Result<Vec<f32>, ModelError> {
        self.decode_column(column)
    }

    /// Computes a row-vector product directly from the encoded matrix.
    ///
    /// The input length must equal [`Self::rows`]. No complete F32 matrix is
    /// allocated; each encoded weight is decoded once while accumulating its
    /// output column.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions or encoded values are invalid, the
    /// input is non-finite, or an accumulated result is non-finite.
    pub fn matmul_f32(&self, input: &[f32]) -> Result<Vec<f32>, ModelError> {
        if input.len() != self.rows {
            return Err(ModelError::Shape(format!(
                "quantized matmul input has {} values, expected {}",
                input.len(),
                self.rows
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::Shape(
                "quantized matmul input contains a non-finite value".to_owned(),
            ));
        }
        let mut output = Vec::with_capacity(self.columns);
        for column in 0..self.columns {
            let weights = self.decode_column(column)?;
            let mut sum = 0.0_f32;
            for (&value, &weight) in input.iter().zip(&weights) {
                sum += value * weight;
                if !sum.is_finite() {
                    return Err(ModelError::Shape(
                        "quantized matmul produced a non-finite value".to_owned(),
                    ));
                }
            }
            output.push(sum);
        }
        Ok(output)
    }

    fn decode_column(&self, column: usize) -> Result<Vec<f32>, ModelError> {
        if column >= self.columns {
            return Err(ModelError::Shape(format!(
                "quantized matrix column {column} is outside {} columns",
                self.columns
            )));
        }
        let (block_values, block_bytes) =
            quantized_block_layout(self.value_type).ok_or_else(|| {
                ModelError::UnsupportedTensorType {
                    name: "<quantized-matrix>".to_owned(),
                    value_type: self.value_type,
                }
            })?;
        if !self.rows.is_multiple_of(block_values) {
            return Err(ModelError::Shape(
                "quantized matrix rows are not block aligned".to_owned(),
            ));
        }
        let column_bytes = self
            .rows
            .checked_div(block_values)
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| {
                ModelError::Shape("quantized matrix byte length overflows".to_owned())
            })?;
        let start = column
            .checked_mul(column_bytes)
            .ok_or_else(|| ModelError::Shape("quantized matrix index overflows".to_owned()))?;
        let end = start
            .checked_add(column_bytes)
            .ok_or_else(|| ModelError::Shape("quantized matrix range overflows".to_owned()))?;
        let bytes = self
            .bytes
            .get(start..end)
            .ok_or_else(|| ModelError::Shape("quantized matrix bytes are truncated".to_owned()))?;
        let values = decode_values(self.value_type, bytes)?;
        if values.len() != self.rows {
            return Err(ModelError::Shape(format!(
                "quantized matrix column decoded {} values, expected {}",
                values.len(),
                self.rows
            )));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::Shape(
                "quantized matrix decoded a non-finite value".to_owned(),
            ));
        }
        Ok(values)
    }
}

/// An owned scalar GGUF metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataScalar {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl TensorDescriptor {
    /// Returns the GGUF tensor name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Returns the logical tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the GGML storage type.
    #[must_use]
    pub const fn value_type(&self) -> TensorType {
        self.value_type
    }

    /// Returns the absolute file offset of the tensor bytes.
    #[must_use]
    pub const fn byte_offset(&self) -> u64 {
        self.byte_offset
    }

    /// Returns the encoded tensor byte length.
    #[must_use]
    pub const fn byte_len(&self) -> u64 {
        self.byte_len
    }
}

/// A validated GGUF model index with content-bound tensor materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufModel {
    path: PathBuf,
    file_size: u64,
    digest: [u8; 32],
    architecture: Option<String>,
    name: Option<String>,
    tensors: Vec<TensorDescriptor>,
    max_file_bytes: u64,
}

/// Errors returned by GGUF model indexing and tensor materialization.
#[derive(Debug)]
pub enum ModelError {
    InvalidLimit,
    Io(String),
    Parse(String),
    TensorNotFound(String),
    UnsupportedTensorType {
        name: String,
        value_type: TensorType,
    },
    Shape(String),
    ContentChanged,
    InvalidUtf8Metadata(&'static str),
    MetadataArray(String),
    MetadataArrayType {
        key: String,
        expected: &'static str,
        actual: String,
    },
    MetadataArrayLimit {
        key: String,
        len: usize,
        max: u64,
    },
}

impl fmt::Display for ModelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLimit => {
                formatter.write_str("model mapping limit must be greater than zero")
            }
            Self::Io(error) => write!(formatter, "could not read GGUF model: {error}"),
            Self::Parse(error) => write!(formatter, "GGUF model is invalid: {error}"),
            Self::TensorNotFound(name) => write!(formatter, "GGUF tensor not found: {name}"),
            Self::UnsupportedTensorType { name, value_type } => {
                write!(
                    formatter,
                    "tensor {name} uses unsupported storage type {value_type}"
                )
            }
            Self::Shape(error) => write!(formatter, "GGUF tensor shape is invalid: {error}"),
            Self::ContentChanged => {
                formatter.write_str("GGUF model bytes changed after the model was opened")
            }
            Self::InvalidUtf8Metadata(key) => {
                write!(formatter, "GGUF metadata {key} is not a UTF-8 string")
            }
            Self::MetadataArray(key) => {
                write!(formatter, "GGUF metadata {key} is an array, not a scalar")
            }
            Self::MetadataArrayType {
                key,
                expected,
                actual,
            } => write!(
                formatter,
                "GGUF metadata {key} array has element type {actual}, expected {expected}"
            ),
            Self::MetadataArrayLimit { key, len, max } => write!(
                formatter,
                "GGUF metadata {key} array has {len} elements, exceeding limit {max}"
            ),
        }
    }
}

impl std::error::Error for ModelError {}

impl GgufModel {
    /// Opens, validates, and indexes one complete GGUF model.
    ///
    /// The file is mapped read-only only for the duration of indexing. The
    /// resulting digest is checked again before each tensor materialization.
    ///
    /// # Errors
    ///
    /// Returns an error when the path is not a regular file, the mapping limit
    /// is exceeded, GGUF validation fails, or a tensor shape cannot fit the
    /// host address space.
    pub fn open(path: impl AsRef<Path>, max_file_bytes: u64) -> Result<Self, ModelError> {
        if max_file_bytes == 0 {
            return Err(ModelError::InvalidLimit);
        }
        let path = std::fs::canonicalize(path.as_ref())
            .map_err(|error| ModelError::Io(error.to_string()))?;
        let metadata =
            std::fs::metadata(&path).map_err(|error| ModelError::Io(error.to_string()))?;
        if !metadata.is_file() {
            return Err(ModelError::Io(
                "model path is not a regular file".to_owned(),
            ));
        }
        let mapped = map_model(&path, max_file_bytes)?;
        let bytes = mapped.as_bytes();
        let digest = digest_bytes(bytes);
        let gguf = Gguf::from_bytes(bytes).map_err(|error| ModelError::Parse(error.to_string()))?;
        let mut tensors = Vec::with_capacity(gguf.tensors().len());
        for (index, tensor) in gguf.tensors().iter().enumerate() {
            let shape = tensor
                .shape()
                .iter()
                .copied()
                .map(|dimension| {
                    usize::try_from(dimension)
                        .map_err(|_| ModelError::Shape(format!("{dimension} exceeds usize")))
                })
                .collect::<Result<Vec<_>, _>>()?;
            if shape.contains(&0) {
                return Err(ModelError::Shape(format!(
                    "tensor {} has a zero dimension",
                    tensor.name
                )));
            }
            let range = gguf.tensor_data_range(index).ok_or_else(|| {
                ModelError::Parse(format!("tensor {} range is outside the file", tensor.name))
            })?;
            tensors.push(TensorDescriptor {
                name: tensor.name.to_owned(),
                shape,
                value_type: tensor.value_type,
                byte_offset: u64::try_from(range.start)
                    .map_err(|_| ModelError::Shape("tensor offset exceeds u64".to_owned()))?,
                byte_len: u64::try_from(range.len())
                    .map_err(|_| ModelError::Shape("tensor byte length exceeds u64".to_owned()))?,
            });
        }
        let architecture = metadata_string(&gguf, "general.architecture")?;
        let name = metadata_string(&gguf, "general.name")?;
        Ok(Self {
            path,
            file_size: u64::try_from(bytes.len())
                .map_err(|_| ModelError::Shape("file exceeds u64".to_owned()))?,
            digest,
            architecture,
            name,
            tensors,
            max_file_bytes,
        })
    }

    /// Returns the canonical model path.
    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Returns the validated model byte length.
    #[must_use]
    pub const fn file_size(&self) -> u64 {
        self.file_size
    }

    /// Returns the model SHA-256 digest as lowercase hexadecimal.
    #[must_use]
    pub fn digest_hex(&self) -> String {
        encode_hex(&self.digest)
    }

    /// Returns the optional `general.architecture` metadata value.
    #[must_use]
    pub fn architecture(&self) -> Option<&str> {
        self.architecture.as_deref()
    }

    /// Returns the optional `general.name` metadata value.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Reads one scalar metadata value while enforcing the model digest.
    ///
    /// Array metadata is intentionally rejected here because callers that
    /// need tokenizer tables must request and validate those arrays through a
    /// dedicated bounded API rather than copying an unbounded value implicitly.
    ///
    /// # Errors
    ///
    /// Returns an error when the model changed, the mapped bytes are invalid,
    /// or the requested metadata value is an array.
    pub fn metadata_scalar(&self, key: &str) -> Result<Option<MetadataScalar>, ModelError> {
        self.with_validated_gguf(|gguf| {
            let Some(value) = gguf.metadata_value(key) else {
                return Ok(None);
            };
            match value {
                ggml_gguf::MetadataValue::Scalar(value) => Ok(Some(owned_scalar(*value))),
                ggml_gguf::MetadataValue::Array(_) => {
                    Err(ModelError::MetadataArray(key.to_owned()))
                }
            }
        })
    }

    /// Reads several scalar metadata values while enforcing the model digest
    /// only once. Returned values preserve the order of `keys`.
    ///
    /// # Errors
    ///
    /// Returns an error when the model changed, the mapped bytes are invalid,
    /// or one of the requested metadata values is an array.
    pub fn metadata_scalars(
        &self,
        keys: &[&str],
    ) -> Result<Vec<Option<MetadataScalar>>, ModelError> {
        self.with_validated_gguf(|gguf| {
            keys.iter()
                .map(|key| {
                    let Some(value) = gguf.metadata_value(key) else {
                        return Ok(None);
                    };
                    match value {
                        ggml_gguf::MetadataValue::Scalar(value) => Ok(Some(owned_scalar(*value))),
                        ggml_gguf::MetadataValue::Array(_) => {
                            Err(ModelError::MetadataArray((*key).to_owned()))
                        }
                    }
                })
                .collect()
        })
    }

    /// Reads a bounded string array from GGUF metadata while enforcing the model digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the model changed, the metadata is malformed, the
    /// array type is not string, or the caller's element bound is exceeded.
    pub fn metadata_string_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<String>>, ModelError> {
        self.with_validated_gguf(|gguf| {
            let Some(value) = gguf.metadata_value(key) else {
                return Ok(None);
            };
            let ggml_gguf::MetadataValue::Array(array) = value else {
                return Err(ModelError::MetadataArray(key.to_owned()));
            };
            if array.element_type() != ggml_gguf::MetadataType::String {
                return Err(ModelError::MetadataArrayType {
                    key: key.to_owned(),
                    expected: "String",
                    actual: format!("{:?}", array.element_type()),
                });
            }
            let length = array.len();
            if u64::try_from(length).unwrap_or(u64::MAX) > max_elements {
                return Err(ModelError::MetadataArrayLimit {
                    key: key.to_owned(),
                    len: length,
                    max: max_elements,
                });
            }
            let mut values = Vec::with_capacity(length);
            for index in 0..length {
                let Some(ggml_gguf::ScalarValue::String(value)) = array.get(index) else {
                    return Err(ModelError::Parse(format!(
                        "GGUF metadata {key} string array contains an invalid element"
                    )));
                };
                values.push(value.to_owned());
            }
            Ok(Some(values))
        })
    }

    /// Reads a bounded F32 metadata array while enforcing the model digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the model changed, the metadata is malformed, the
    /// array type is not F32, or the caller's element bound is exceeded.
    pub fn metadata_f32_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<f32>>, ModelError> {
        self.with_validated_gguf(|gguf| {
            let Some(value) = gguf.metadata_value(key) else {
                return Ok(None);
            };
            let ggml_gguf::MetadataValue::Array(array) = value else {
                return Err(ModelError::MetadataArray(key.to_owned()));
            };
            if array.element_type() != ggml_gguf::MetadataType::F32 {
                return Err(ModelError::MetadataArrayType {
                    key: key.to_owned(),
                    expected: "F32",
                    actual: format!("{:?}", array.element_type()),
                });
            }
            let length = array.len();
            if u64::try_from(length).unwrap_or(u64::MAX) > max_elements {
                return Err(ModelError::MetadataArrayLimit {
                    key: key.to_owned(),
                    len: length,
                    max: max_elements,
                });
            }
            let mut values = Vec::with_capacity(length);
            for index in 0..length {
                let Some(ggml_gguf::ScalarValue::F32(value)) = array.get(index) else {
                    return Err(ModelError::Parse(format!(
                        "GGUF metadata {key} F32 array contains an invalid element"
                    )));
                };
                if !value.is_finite() {
                    return Err(ModelError::Parse(format!(
                        "GGUF metadata {key} contains a non-finite value"
                    )));
                }
                values.push(value);
            }
            Ok(Some(values))
        })
    }

    /// Returns all indexed tensor descriptors in GGUF order.
    #[must_use]
    pub fn tensors(&self) -> &[TensorDescriptor] {
        &self.tensors
    }

    /// Finds one tensor descriptor by exact name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&TensorDescriptor> {
        self.tensors.iter().find(|tensor| tensor.name == name)
    }

    /// Materializes one tensor as F32 values in the checked CPU tensor engine.
    ///
    /// F32, F16, BF16, `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`,
    /// `Q5_K`, `Q6_K`, `Q8_0`, and `Q8_K` storage are supported. Quantized
    /// formats are
    /// decoded on the CPU into owned F32 values; the encoded bytes remain
    /// content-bound to the digest captured by [`Self::open`].
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is missing or uses an unsupported
    /// storage type, the model bytes changed, the tensor range is invalid, or
    /// the shape does not match the decoded values.
    pub fn load_f32(&self, name: &str) -> Result<Tensor, ModelError> {
        let descriptor = self
            .tensor(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_owned()))?;
        self.with_validated_bytes(|bytes| Self::materialize_f32(bytes, descriptor))
    }

    /// Materializes several tensors as F32 values while mapping and hashing
    /// the GGUF file only once.
    ///
    /// The returned tensors preserve the order of `names`. Duplicate names
    /// are allowed and produce duplicate owned tensors.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing or uses an unsupported
    /// storage type, the model bytes changed, a tensor range is invalid, or a
    /// shape does not match its decoded values.
    pub fn load_f32_many(&self, names: &[&str]) -> Result<Vec<Tensor>, ModelError> {
        let mut tensors = Vec::with_capacity(names.len());
        self.for_each_f32(names, |_, tensor| {
            tensors.push(tensor);
            Ok::<(), ModelError>(())
        })?;
        Ok(tensors)
    }

    /// Loads one rank-2 quantized GGUF matrix without expanding it to F32.
    ///
    /// The encoded block bytes are copied into an owned buffer so the returned
    /// matrix remains valid after the file mapping is released. This is the
    /// CPU fallback boundary for direct quantized products and token lookup.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is missing, not a supported quantized
    /// matrix, malformed, or the model bytes changed after indexing.
    pub fn load_quantized(&self, name: &str) -> Result<QuantizedMatrix, ModelError> {
        let mut matrices = self.load_quantized_many(&[name])?;
        matrices
            .pop()
            .ok_or_else(|| ModelError::TensorNotFound(name.to_owned()))
    }

    /// Loads several rank-2 quantized GGUF matrices through one validated
    /// mapping without expanding them to F32.
    ///
    /// The returned matrices preserve the order of `names`. Each encoded
    /// tensor is copied once into an owned buffer, while the GGUF is mapped and
    /// digest-checked only once for the whole batch.
    ///
    /// # Errors
    ///
    /// Returns an error when any tensor is missing, not a supported quantized
    /// matrix, malformed, or the model bytes changed after indexing.
    pub fn load_quantized_many(&self, names: &[&str]) -> Result<Vec<QuantizedMatrix>, ModelError> {
        let descriptors = names
            .iter()
            .map(|name| {
                self.tensor(name)
                    .ok_or_else(|| ModelError::TensorNotFound((*name).to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for descriptor in &descriptors {
            validate_quantized_matrix_descriptor(descriptor)?;
        }
        self.with_validated_bytes(|bytes| {
            descriptors
                .iter()
                .map(|descriptor| materialize_quantized(bytes, descriptor))
                .collect()
        })
    }

    /// Converts one rank-2 quantized GGUF matrix directly to MLX affine
    /// quantization components.
    ///
    /// The GGUF matrix is decoded one input group at a time, so this path
    /// never allocates a complete F32 copy before packing the MLX weights.
    /// The returned matrix uses MLX's `[output, input]` orientation and the
    /// same packed little-endian bit layout as `mlx_quantize`.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is missing, not a supported GGUF
    /// quantized matrix, not rank 2, not aligned to `group_size`, the
    /// quantization parameters are unsupported, the model changed, or the
    /// encoded tensor is malformed.
    pub fn load_affine_quantized(
        &self,
        name: &str,
        group_size: usize,
        bits: usize,
    ) -> Result<AffineQuantizedMatrix, ModelError> {
        let descriptor = self
            .tensor(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_owned()))?;
        self.with_validated_bytes(|bytes| {
            materialize_affine_quantized(bytes, descriptor, group_size, bits)
        })
    }

    /// Loads tensors through one validated mapping, selecting direct MLX
    /// affine conversion for eligible rank-2 GGUF quantized matrices and F32
    /// materialization for all other tensors.
    ///
    /// This is the preferred boundary for backends that can consume MLX
    /// affine weights. Quantized vectors and non-matrix tensors remain F32,
    /// while eligible matrices never pass through a full F32 materialization.
    /// Tensors are delivered in the order of `names`.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing or malformed, the model
    /// changed, the requested affine parameters are unsupported, or the
    /// callback fails.
    pub fn for_each_tensor<F, E>(
        &self,
        names: &[&str],
        group_size: usize,
        bits: usize,
        mut callback: F,
    ) -> Result<(), E>
    where
        F: FnMut(&str, LoadedTensor) -> Result<(), E>,
        E: From<ModelError>,
    {
        validate_affine_quantization(group_size, bits).map_err(E::from)?;
        let descriptors = names
            .iter()
            .map(|name| {
                self.tensor(name)
                    .ok_or_else(|| ModelError::TensorNotFound((*name).to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(E::from)?;
        let mapped = map_model(&self.path, self.max_file_bytes).map_err(E::from)?;
        let bytes = mapped.as_bytes();
        if digest_bytes(bytes) != self.digest {
            return Err(E::from(ModelError::ContentChanged));
        }
        for descriptor in descriptors {
            let loaded = if affine_quantized_candidate(descriptor, group_size) {
                materialize_affine_quantized(bytes, descriptor, group_size, bits)
                    .map(LoadedTensor::AffineQuantized)
                    .map_err(E::from)?
            } else {
                Self::materialize_f32(bytes, descriptor)
                    .map(LoadedTensor::F32)
                    .map_err(E::from)?
            };
            callback(&descriptor.name, loaded)?;
        }
        Ok(())
    }

    /// Computes a row-vector matrix product directly from a quantized GGUF
    /// matrix without materializing the matrix as F32 values.
    ///
    /// The input must have the length of the descriptor's first dimension.
    /// GGML stores matrix values in column-major tensor order, so each output
    /// column is decoded and dotted against the input row. `Q4_0`, `Q4_1`,
    /// `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, and
    /// `Q8_K` are supported. The operation walks the encoded blocks directly
    /// and does not allocate a temporary F32 matrix.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not a supported rank-2 quantized
    /// matrix, dimensions or block alignment are invalid, model bytes have
    /// changed, or a numerical result is non-finite.
    pub fn matmul_f32_quantized(&self, name: &str, input: &[f32]) -> Result<Vec<f32>, ModelError> {
        let descriptor = self
            .tensor(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_owned()))?;
        let (rows, columns) = match descriptor.shape.as_slice() {
            [rows, columns] => (*rows, *columns),
            shape => {
                return Err(ModelError::Shape(format!(
                    "quantized matmul requires a rank-2 matrix, got {shape:?}"
                )));
            }
        };
        if input.len() != rows {
            return Err(ModelError::Shape(format!(
                "quantized matmul input has {} values, expected {rows}",
                input.len()
            )));
        }
        if input.iter().any(|value| !value.is_finite()) {
            return Err(ModelError::Shape(
                "quantized matmul input contains a non-finite value".to_owned(),
            ));
        }
        let (block_values, block_bytes) = quantized_block_layout(descriptor.value_type)
            .ok_or_else(|| ModelError::UnsupportedTensorType {
                name: descriptor.name.clone(),
                value_type: descriptor.value_type,
            })?;
        if !rows.is_multiple_of(block_values) {
            return Err(ModelError::Shape(format!(
                "quantized matmul rows must be a multiple of {block_values}"
            )));
        }
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_div(block_values))
            .and_then(|blocks| blocks.checked_mul(block_bytes))
            .ok_or_else(|| {
                ModelError::Shape("quantized matrix byte length overflows".to_owned())
            })?;
        if usize::try_from(descriptor.byte_len).ok() != Some(expected_bytes) {
            return Err(ModelError::Shape(format!(
                "quantized matrix byte length {}, expected {expected_bytes}",
                descriptor.byte_len
            )));
        }
        self.with_validated_bytes(|bytes| {
            let start = usize::try_from(descriptor.byte_offset)
                .map_err(|_| ModelError::Shape("tensor offset exceeds usize".to_owned()))?;
            let end = start
                .checked_add(expected_bytes)
                .ok_or_else(|| ModelError::Shape("tensor range overflows usize".to_owned()))?;
            let tensor_bytes = bytes
                .get(start..end)
                .ok_or_else(|| ModelError::Parse("tensor range is outside the file".to_owned()))?;
            let mut output = Vec::with_capacity(columns);
            let column_bytes = expected_bytes
                .checked_div(columns)
                .ok_or_else(|| ModelError::Shape("quantized column width is zero".to_owned()))?;
            for column in 0..columns {
                let column_start = column.checked_mul(column_bytes).ok_or_else(|| {
                    ModelError::Shape("quantized matrix index overflows".to_owned())
                })?;
                let column_end = column_start.checked_add(column_bytes).ok_or_else(|| {
                    ModelError::Shape("quantized matrix range overflows".to_owned())
                })?;
                let weights = decode_values(
                    descriptor.value_type,
                    tensor_bytes.get(column_start..column_end).ok_or_else(|| {
                        ModelError::Parse("quantized column range is outside the tensor".to_owned())
                    })?,
                )?;
                if weights.len() != rows {
                    return Err(ModelError::Shape(format!(
                        "quantized column decoded {} values, expected {rows}",
                        weights.len()
                    )));
                }
                let mut sum = 0.0_f32;
                for (&value, &weight) in input.iter().zip(&weights) {
                    if !weight.is_finite() {
                        return Err(ModelError::Shape(
                            "quantized matmul decoded a non-finite value".to_owned(),
                        ));
                    }
                    sum += value * weight;
                    if !sum.is_finite() {
                        return Err(ModelError::Shape(
                            "quantized matmul produced a non-finite value".to_owned(),
                        ));
                    }
                }
                output.push(sum);
            }
            Ok(output)
        })
    }

    /// Materializes several tensors as F32 values from one validated mapping,
    /// invoking `callback` after each tensor so callers can release host data
    /// before the next tensor is decoded.
    ///
    /// The callback receives tensors in the order of `names`. Duplicate names
    /// are allowed. Its error type must be constructible from [`ModelError`]
    /// so model validation failures can be returned without losing the
    /// caller's error type.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor is missing or uses an unsupported
    /// storage type, the model bytes changed, a tensor range is invalid, a
    /// shape does not match its decoded values, or `callback` fails.
    pub fn for_each_f32<F, E>(&self, names: &[&str], mut callback: F) -> Result<(), E>
    where
        F: FnMut(&str, Tensor) -> Result<(), E>,
        E: From<ModelError>,
    {
        let descriptors = names
            .iter()
            .map(|name| {
                self.tensor(name)
                    .ok_or_else(|| ModelError::TensorNotFound((*name).to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(E::from)?;
        let mapped = map_model(&self.path, self.max_file_bytes).map_err(E::from)?;
        let bytes = mapped.as_bytes();
        if digest_bytes(bytes) != self.digest {
            return Err(E::from(ModelError::ContentChanged));
        }
        for descriptor in descriptors {
            let tensor = Self::materialize_f32(bytes, descriptor).map_err(E::from)?;
            callback(&descriptor.name, tensor)?;
        }
        Ok(())
    }

    fn with_validated_gguf<F, T>(&self, callback: F) -> Result<T, ModelError>
    where
        F: FnOnce(&Gguf<'_>) -> Result<T, ModelError>,
    {
        self.with_validated_bytes(|bytes| {
            let gguf =
                Gguf::from_bytes(bytes).map_err(|error| ModelError::Parse(error.to_string()))?;
            callback(&gguf)
        })
    }

    fn with_validated_bytes<F, T>(&self, callback: F) -> Result<T, ModelError>
    where
        F: FnOnce(&[u8]) -> Result<T, ModelError>,
    {
        let mapped = map_model(&self.path, self.max_file_bytes)?;
        let bytes = mapped.as_bytes();
        if digest_bytes(bytes) != self.digest {
            return Err(ModelError::ContentChanged);
        }
        callback(bytes)
    }

    fn materialize_f32(bytes: &[u8], descriptor: &TensorDescriptor) -> Result<Tensor, ModelError> {
        if !matches!(
            descriptor.value_type.raw(),
            0 | 1 | 2 | 3 | 6 | 7 | 8 | 10 | 11 | 12 | 13 | 14 | 15 | 30
        ) {
            return Err(ModelError::UnsupportedTensorType {
                name: descriptor.name.clone(),
                value_type: descriptor.value_type,
            });
        }
        let start = usize::try_from(descriptor.byte_offset)
            .map_err(|_| ModelError::Shape("tensor offset exceeds usize".to_owned()))?;
        let length = usize::try_from(descriptor.byte_len)
            .map_err(|_| ModelError::Shape("tensor byte length exceeds usize".to_owned()))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| ModelError::Shape("tensor range overflows usize".to_owned()))?;
        let tensor_bytes = bytes
            .get(start..end)
            .ok_or_else(|| ModelError::Parse("tensor range is outside the file".to_owned()))?;
        let values = decode_values(descriptor.value_type, tensor_bytes)?;
        let tensor = Tensor::from_data(descriptor.shape.clone(), values)
            .map_err(|error| ModelError::Shape(error.to_string()))?;
        tensor
            .validate_finite()
            .map_err(|error| ModelError::Shape(error.to_string()))?;
        Ok(tensor)
    }
}

fn decode_values(value_type: TensorType, bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    match value_type.raw() {
        0 => decode_f32(bytes),
        1 => decode_f16(bytes),
        30 => decode_bf16(bytes),
        2 => decode_q4_0(bytes),
        3 => decode_q4_1(bytes),
        6 => decode_q5_0(bytes),
        7 => decode_q5_1(bytes),
        8 => decode_q8_0(bytes),
        10 => decode_q2_k(bytes),
        11 => decode_q3_k(bytes),
        12 => decode_q4_k(bytes),
        13 => decode_q5_k(bytes),
        14 => decode_q6_k(bytes),
        15 => decode_q8_k(bytes),
        _ => Err(ModelError::UnsupportedTensorType {
            name: "<unknown>".to_owned(),
            value_type,
        }),
    }
}

fn decode_f32(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    let (chunks, remainder) = bytes.as_chunks::<4>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "F32 tensor byte length is not aligned".to_owned(),
        ));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f32::from_le_bytes(*chunk))
        .collect())
}

fn decode_f16(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "F16 tensor byte length is not aligned".to_owned(),
        ));
    }
    Ok(chunks
        .iter()
        .map(|chunk| f16_to_f32(u16::from_le_bytes(*chunk)))
        .collect())
}

fn decode_bf16(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    let (chunks, remainder) = bytes.as_chunks::<2>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "BF16 tensor byte length is not aligned".to_owned(),
        ));
    }
    Ok(chunks
        .iter()
        .map(|chunk| u32::from(u16::from_le_bytes(*chunk)) << 16)
        .map(f32::from_bits)
        .collect())
}

fn validate_affine_quantization(group_size: usize, bits: usize) -> Result<(), ModelError> {
    if !matches!(group_size, 32 | 64 | 128) {
        return Err(ModelError::Shape(
            "MLX affine group size must be 32, 64, or 128".to_owned(),
        ));
    }
    if !matches!(bits, 2 | 3 | 4 | 5 | 6 | 8) {
        return Err(ModelError::Shape(
            "MLX affine bit width must be 2, 3, 4, 5, 6, or 8".to_owned(),
        ));
    }
    Ok(())
}

fn affine_quantized_candidate(descriptor: &TensorDescriptor, group_size: usize) -> bool {
    descriptor.shape.len() == 2
        && quantized_block_layout(descriptor.value_type).is_some()
        && descriptor.shape[0].is_multiple_of(group_size)
}

#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]
fn materialize_affine_quantized(
    bytes: &[u8],
    descriptor: &TensorDescriptor,
    group_size: usize,
    bits: usize,
) -> Result<AffineQuantizedMatrix, ModelError> {
    validate_affine_quantization(group_size, bits)?;
    let (input, output) = match descriptor.shape.as_slice() {
        [input, output] => (*input, *output),
        shape => {
            return Err(ModelError::Shape(format!(
                "MLX affine quantization requires a rank-2 matrix, got {shape:?}"
            )));
        }
    };
    if !input.is_multiple_of(group_size) {
        return Err(ModelError::Shape(format!(
            "MLX affine input dimension {input} is not divisible by group size {group_size}"
        )));
    }
    let (block_values, block_bytes) =
        quantized_block_layout(descriptor.value_type).ok_or_else(|| {
            ModelError::UnsupportedTensorType {
                name: descriptor.name.clone(),
                value_type: descriptor.value_type,
            }
        })?;
    if !input.is_multiple_of(block_values) {
        return Err(ModelError::Shape(format!(
            "quantized matrix rows must be a multiple of {block_values}"
        )));
    }
    let expected_bytes = input
        .checked_mul(output)
        .and_then(|elements| elements.checked_div(block_values))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| ModelError::Shape("quantized matrix byte length overflows".to_owned()))?;
    if usize::try_from(descriptor.byte_len).ok() != Some(expected_bytes) {
        return Err(ModelError::Shape(format!(
            "quantized matrix byte length {}, expected {expected_bytes}",
            descriptor.byte_len
        )));
    }
    let start = usize::try_from(descriptor.byte_offset)
        .map_err(|_| ModelError::Shape("tensor offset exceeds usize".to_owned()))?;
    let end = start
        .checked_add(expected_bytes)
        .ok_or_else(|| ModelError::Shape("tensor range overflows usize".to_owned()))?;
    let tensor_bytes = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Parse("tensor range is outside the file".to_owned()))?;
    let packed_columns = input
        .checked_mul(bits)
        .and_then(|values| values.checked_div(32))
        .ok_or_else(|| ModelError::Shape("MLX packed matrix shape overflows".to_owned()))?;
    let groups = input / group_size;
    let packed_len = output
        .checked_mul(packed_columns)
        .ok_or_else(|| ModelError::Shape("MLX packed matrix length overflows".to_owned()))?;
    let affine_len = output
        .checked_mul(groups)
        .ok_or_else(|| ModelError::Shape("MLX affine parameter length overflows".to_owned()))?;
    let mut packed = vec![0_u32; packed_len];
    let mut scales = Vec::with_capacity(affine_len);
    let mut biases = Vec::with_capacity(affine_len);
    let n_bins = f32::from((1_u16 << bits) - 1);
    let mut group_values = vec![0.0_f32; group_size];
    for output_index in 0..output {
        for group_index in 0..groups {
            let group_start = group_index * group_size;
            let mut minimum = f32::INFINITY;
            let mut maximum = f32::NEG_INFINITY;
            for (offset, value_slot) in group_values.iter_mut().enumerate() {
                let index = output_index
                    .checked_mul(input)
                    .and_then(|base| base.checked_add(group_start + offset))
                    .ok_or_else(|| {
                        ModelError::Shape("quantized matrix index overflows".to_owned())
                    })?;
                let value = quantized_value_at(descriptor.value_type, tensor_bytes, index)?;
                if !value.is_finite() {
                    return Err(ModelError::Shape(
                        "quantized matrix contains a non-finite value".to_owned(),
                    ));
                }
                *value_slot = value;
                minimum = minimum.min(value);
                maximum = maximum.max(value);
            }
            let mut scale = ((maximum - minimum) / n_bins).max(1e-7);
            let side = minimum.abs() > maximum.abs();
            let edge = if side { minimum } else { maximum };
            if !side {
                scale = -scale;
            }
            let q0 = (edge / scale).round();
            let bias = if q0 == 0.0 { 0.0 } else { edge };
            if q0 != 0.0 {
                scale = edge / q0;
            }
            if !scale.is_finite() || !bias.is_finite() {
                return Err(ModelError::Shape(
                    "MLX affine parameters are non-finite".to_owned(),
                ));
            }
            scales.push(scale);
            biases.push(bias);
            let packed_start =
                output_index * packed_columns + group_index * (group_size * bits / 32);
            for (offset, &value) in group_values.iter().enumerate() {
                let quantized = ((value - bias) / scale).round().clamp(0.0, n_bins) as u32;
                let bit_offset = offset * bits;
                let word = packed_start + bit_offset / 32;
                let shift = bit_offset % 32;
                packed[word] |= quantized << shift;
                if shift + bits > 32 {
                    packed[word + 1] |= quantized >> (32 - shift);
                }
            }
        }
    }
    Ok(AffineQuantizedMatrix {
        rows: output,
        columns: input,
        group_size,
        bits,
        packed,
        scales,
        biases,
    })
}

fn quantized_block_layout(value_type: TensorType) -> Option<(usize, usize)> {
    match value_type.raw() {
        2 => Some((32, 18)),
        3 => Some((32, 20)),
        6 => Some((32, 22)),
        7 => Some((32, 24)),
        8 => Some((32, 34)),
        10 => Some((256, 84)),
        11 => Some((256, 110)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        15 => Some((256, 292)),
        _ => None,
    }
}

fn validate_quantized_matrix_descriptor(
    descriptor: &TensorDescriptor,
) -> Result<(usize, usize, usize), ModelError> {
    let (rows, columns) = match descriptor.shape.as_slice() {
        [rows, columns] => (*rows, *columns),
        shape => {
            return Err(ModelError::Shape(format!(
                "quantized matrix requires rank 2, got {shape:?}"
            )));
        }
    };
    let (block_values, block_bytes) =
        quantized_block_layout(descriptor.value_type).ok_or_else(|| {
            ModelError::UnsupportedTensorType {
                name: descriptor.name.clone(),
                value_type: descriptor.value_type,
            }
        })?;
    if !rows.is_multiple_of(block_values) {
        return Err(ModelError::Shape(format!(
            "quantized matrix rows must be a multiple of {block_values}"
        )));
    }
    let expected_bytes = rows
        .checked_mul(columns)
        .and_then(|elements| elements.checked_div(block_values))
        .and_then(|blocks| blocks.checked_mul(block_bytes))
        .ok_or_else(|| ModelError::Shape("quantized matrix byte length overflows".to_owned()))?;
    if usize::try_from(descriptor.byte_len).ok() != Some(expected_bytes) {
        return Err(ModelError::Shape(format!(
            "quantized matrix byte length {}, expected {expected_bytes}",
            descriptor.byte_len
        )));
    }
    Ok((rows, columns, expected_bytes))
}

fn materialize_quantized(
    bytes: &[u8],
    descriptor: &TensorDescriptor,
) -> Result<QuantizedMatrix, ModelError> {
    let (rows, columns, expected_bytes) = validate_quantized_matrix_descriptor(descriptor)?;
    let start = usize::try_from(descriptor.byte_offset)
        .map_err(|_| ModelError::Shape("tensor offset exceeds usize".to_owned()))?;
    let end = start
        .checked_add(expected_bytes)
        .ok_or_else(|| ModelError::Shape("tensor range overflows usize".to_owned()))?;
    let tensor_bytes = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Parse("tensor range is outside the file".to_owned()))?;
    Ok(QuantizedMatrix {
        rows,
        columns,
        value_type: descriptor.value_type,
        bytes: tensor_bytes.to_vec(),
    })
}

fn quantized_value_at(
    value_type: TensorType,
    bytes: &[u8],
    index: usize,
) -> Result<f32, ModelError> {
    match value_type.raw() {
        2 => q4_0_value_at(bytes, index),
        3 => q4_1_value_at(bytes, index),
        6 => q5_0_value_at(bytes, index),
        7 => q5_1_value_at(bytes, index),
        8 => q8_0_value_at(bytes, index),
        10 => q2_k_value_at(bytes, index),
        11 => q3_k_value_at(bytes, index),
        12 => q4_k_value_at(bytes, index),
        13 => q5_k_value_at(bytes, index),
        14 => q6_k_value_at(bytes, index),
        15 => q8_k_value_at(bytes, index),
        _ => unreachable!("value type validated above"),
    }
}

fn decode_q4_0(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q4_0 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for index in 0..16 {
            let packed = block[2 + index];
            values.push((f32::from(packed & 0x0f) - 8.0) * scale);
        }
        for index in 0..16 {
            let packed = block[2 + index];
            values.push((f32::from(packed >> 4) - 8.0) * scale);
        }
    }
    Ok(values)
}

fn decode_q4_1(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 20;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q4_1 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        for index in 0..16 {
            let packed = block[4 + index];
            values.push(scale * f32::from(packed & 0x0f) + minimum);
        }
        for index in 0..16 {
            let packed = block[4 + index];
            values.push(scale * f32::from(packed >> 4) + minimum);
        }
    }
    Ok(values)
}

fn decode_q8_0(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 34;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q8_0 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        values.extend(
            block[2..]
                .iter()
                .map(|value| f32::from(i8::from_ne_bytes([*value])) * scale),
        );
    }
    Ok(values)
}

fn q4_0_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 32;
    let offset = index % 32;
    let start = block
        .checked_mul(18)
        .ok_or_else(|| ModelError::Shape("Q4_0 index overflows".to_owned()))?;
    let end = start
        .checked_add(18)
        .ok_or_else(|| ModelError::Shape("Q4_0 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q4_0 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let packed = block[2 + offset % 16];
    let quantized = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok((f32::from(quantized) - 8.0) * scale)
}

fn q4_1_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 32;
    let offset = index % 32;
    let start = block
        .checked_mul(20)
        .ok_or_else(|| ModelError::Shape("Q4_1 index overflows".to_owned()))?;
    let end = start
        .checked_add(20)
        .ok_or_else(|| ModelError::Shape("Q4_1 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q4_1 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let packed = block[4 + offset % 16];
    let quantized = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(quantized) + minimum)
}

fn q5_0_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 32;
    let offset = index % 32;
    let start = block
        .checked_mul(22)
        .ok_or_else(|| ModelError::Shape("Q5_0 index overflows".to_owned()))?;
    let end = start
        .checked_add(22)
        .ok_or_else(|| ModelError::Shape("Q5_0 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q5_0 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let packed = block[6 + offset % 16];
    let low = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let high = ((high_bits >> offset) & 1) as u8;
    Ok((f32::from(low | (high << 4)) - 16.0) * scale)
}

fn q5_1_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 32;
    let offset = index % 32;
    let start = block
        .checked_mul(24)
        .ok_or_else(|| ModelError::Shape("Q5_1 index overflows".to_owned()))?;
    let end = start
        .checked_add(24)
        .ok_or_else(|| ModelError::Shape("Q5_1 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q5_1 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let high_bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let packed = block[8 + offset % 16];
    let low = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    let high = ((high_bits >> offset) & 1) as u8;
    Ok(scale * f32::from(low | (high << 4)) + minimum)
}

fn q8_0_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 32;
    let offset = index % 32;
    let start = block
        .checked_mul(34)
        .ok_or_else(|| ModelError::Shape("Q8_0 index overflows".to_owned()))?;
    let end = start
        .checked_add(34)
        .ok_or_else(|| ModelError::Shape("Q8_0 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q8_0 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    Ok(f32::from(i8::from_ne_bytes([block[2 + offset]])) * scale)
}

fn q4_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block = index / 256;
    let offset = index % 256;
    let start = block
        .checked_mul(144)
        .ok_or_else(|| ModelError::Shape("Q4_K index overflows".to_owned()))?;
    let end = start
        .checked_add(144)
        .ok_or_else(|| ModelError::Shape("Q4_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q4_K block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min_scale = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let quantized = &block[16..];
    let group = offset / 32;
    let (group_scale, group_min) = q4_k_scale_min(group, scales);
    let packed = quantized[(group / 2) * 32 + offset % 32];
    let quantized_value = if group.is_multiple_of(2) {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(group_scale) * f32::from(quantized_value)
        - min_scale * f32::from(group_min))
}

fn q2_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(84)
        .ok_or_else(|| ModelError::Shape("Q2_K index overflows".to_owned()))?;
    let end = start
        .checked_add(84)
        .ok_or_else(|| ModelError::Shape("Q2_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q2_K block is outside the tensor".to_owned()))?;
    let scales = &block[..16];
    let quantized = &block[16..80];
    let scale = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
    let min_scale = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
    let group = offset / 16;
    let index_in_group = offset % 16;
    let group_scale = scale * f32::from(scales[group] & 0x0f);
    let group_min = min_scale * f32::from(scales[group] >> 4);
    let segment = group / 8;
    let half = (group % 2) * 16;
    let shift = ((group % 8) / 2) * 2;
    let quantized_value = (quantized[segment * 32 + half + index_in_group] >> shift) & 0x03;
    Ok(group_scale * f32::from(quantized_value) - group_min)
}

fn q3_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(110)
        .ok_or_else(|| ModelError::Shape("Q3_K index overflows".to_owned()))?;
    let end = start
        .checked_add(110)
        .ok_or_else(|| ModelError::Shape("Q3_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q3_K block is outside the tensor".to_owned()))?;
    let high_bits = &block[..32];
    let quantized = &block[32..96];
    let scales = &block[96..108];
    let scale = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
    let mut group_scales = [0_i8; 16];
    for group in 0..16 {
        let low = if group < 8 {
            scales[group] & 0x0f
        } else {
            scales[group - 8] >> 4
        };
        let high = (scales[8 + group / 4] >> ((group % 4) * 2)) & 0x03;
        group_scales[group] = i8::try_from(low | (high << 4)).unwrap_or_default() - 32;
    }
    let group = offset / 16;
    let index_in_group = offset % 16;
    let group_scale = scale * f32::from(group_scales[group]);
    let segment = group / 8;
    let half = (group % 2) * 16;
    let shift = ((group % 8) / 2) * 2;
    let high_shift = group / 2;
    let low = (quantized[segment * 32 + half + index_in_group] >> shift) & 0x03;
    let high = (high_bits[half + index_in_group] >> high_shift) & 0x01;
    let quantized_value =
        i8::try_from(low).unwrap_or_default() - i8::try_from(high ^ 1).unwrap_or_default() * 4;
    Ok(group_scale * f32::from(quantized_value))
}

fn q5_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(176)
        .ok_or_else(|| ModelError::Shape("Q5_K index overflows".to_owned()))?;
    let end = start
        .checked_add(176)
        .ok_or_else(|| ModelError::Shape("Q5_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q5_K block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let min_scale = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
    let scales = &block[4..16];
    let high_bits = &block[16..48];
    let quantized = &block[48..];
    let group = offset / 32;
    let index_in_group = offset % 32;
    let (group_scale, group_min) = q4_k_scale_min(group, scales);
    let byte_offset = (group / 2) * 32;
    let shift = (group % 2) * 4;
    let low = (quantized[byte_offset + index_in_group] >> shift) & 0x0f;
    let high = (high_bits[index_in_group] >> group) & 1;
    Ok(
        scale * f32::from(group_scale) * f32::from(low | (high << 4))
            - min_scale * f32::from(group_min),
    )
}

fn q6_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(210)
        .ok_or_else(|| ModelError::Shape("Q6_K index overflows".to_owned()))?;
    let end = start
        .checked_add(210)
        .ok_or_else(|| ModelError::Shape("Q6_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q6_K block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let chunk = offset / 128;
    let local = offset % 128;
    let lane = local / 32;
    let l = local % 32;
    let low_offset = chunk * 64;
    let high_offset = chunk * 32;
    let scale_offset = chunk * 8;
    let sub_block = l / 16;
    let quantized = match lane {
        0 => (ql[low_offset + l] & 0x0f) | ((qh[high_offset + l] & 0x03) << 4),
        1 => (ql[low_offset + l + 32] & 0x0f) | (((qh[high_offset + l] >> 2) & 0x03) << 4),
        2 => (ql[low_offset + l] >> 4) | (((qh[high_offset + l] >> 4) & 0x03) << 4),
        3 => (ql[low_offset + l + 32] >> 4) | (((qh[high_offset + l] >> 6) & 0x03) << 4),
        _ => unreachable!("Q6_K lane is bounded by offset modulo 128"),
    };
    let quantized = i8::try_from(quantized).unwrap_or_default() - 32;
    let decoded_scale = scale
        * f32::from(i8::from_ne_bytes([
            scales[scale_offset + sub_block + lane * 2]
        ]));
    Ok(decoded_scale * f32::from(quantized))
}

fn q8_k_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(292)
        .ok_or_else(|| ModelError::Shape("Q8_K index overflows".to_owned()))?;
    let end = start
        .checked_add(292)
        .ok_or_else(|| ModelError::Shape("Q8_K block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("Q8_K block is outside the tensor".to_owned()))?;
    let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let quantized = i8::from_ne_bytes([block[4 + offset]]);
    Ok(scale * f32::from(quantized))
}

fn decode_q5_0(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 22;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q5_0 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let high_bits = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        for index in 0..16 {
            let low = block[6 + index] & 0x0f;
            let high = ((high_bits >> index) & 1) as u8;
            values.push((f32::from(low | (high << 4)) - 16.0) * scale);
        }
        for index in 0..16 {
            let low = block[6 + index] >> 4;
            let high = ((high_bits >> (index + 16)) & 1) as u8;
            values.push((f32::from(low | (high << 4)) - 16.0) * scale);
        }
    }
    Ok(values)
}

fn decode_q5_1(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 24;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q5_1 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let minimum = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let high_bits = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for index in 0..16 {
            let low = block[8 + index] & 0x0f;
            let high = ((high_bits >> index) & 1) as u8;
            values.push(scale * f32::from(low | (high << 4)) + minimum);
        }
        for index in 0..16 {
            let low = block[8 + index] >> 4;
            let high = ((high_bits >> (index + 16)) & 1) as u8;
            values.push(scale * f32::from(low | (high << 4)) + minimum);
        }
    }
    Ok(values)
}

fn decode_q2_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 84;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q2_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scales = &block[..16];
        let quantized = &block[16..80];
        let scale = f16_to_f32(u16::from_le_bytes([block[80], block[81]]));
        let min_scale = f16_to_f32(u16::from_le_bytes([block[82], block[83]]));
        for (group, &scale_bits) in scales.iter().enumerate() {
            let group_scale = scale * f32::from(scale_bits & 0x0f);
            let group_min = min_scale * f32::from(scale_bits >> 4);
            let segment = group / 8;
            let half = (group % 2) * 16;
            let shift = ((group % 8) / 2) * 2;
            for index in 0..16 {
                let quantized_value = (quantized[segment * 32 + half + index] >> shift) & 0x03;
                values.push(group_scale * f32::from(quantized_value) - group_min);
            }
        }
    }
    Ok(values)
}

fn decode_q3_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q3_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let high_bits = &block[..32];
        let quantized = &block[32..96];
        let scales = &block[96..108];
        let scale = f16_to_f32(u16::from_le_bytes([block[108], block[109]]));
        let mut group_scales = [0_i8; 16];
        for index in 0..16 {
            let low = if index < 8 {
                scales[index] & 0x0f
            } else {
                scales[index - 8] >> 4
            };
            let high = (scales[8 + index / 4] >> ((index % 4) * 2)) & 0x03;
            group_scales[index] = i8::try_from(low | (high << 4)).unwrap_or_default() - 32;
        }
        for (group, &group_scale_bits) in group_scales.iter().enumerate() {
            let group_scale = scale * f32::from(group_scale_bits);
            let segment = group / 8;
            let half = (group % 2) * 16;
            let shift = ((group % 8) / 2) * 2;
            let high_shift = group / 2;
            for index in 0..16 {
                let low = (quantized[segment * 32 + half + index] >> shift) & 0x03;
                let high = (high_bits[half + index] >> high_shift) & 0x01;
                let quantized_value = i8::try_from(low).unwrap_or_default()
                    - i8::try_from(high ^ 1).unwrap_or_default() * 4;
                values.push(group_scale * f32::from(quantized_value));
            }
        }
    }
    Ok(values)
}

fn decode_q4_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 144;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q4_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_scale = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let quantized = &block[16..];
        for group in 0..8 {
            let (group_scale, group_min) = q4_k_scale_min(group, scales);
            let byte_offset = (group / 2) * 32;
            let shift = (group % 2) * 4;
            for packed in &quantized[byte_offset..byte_offset + 32] {
                let quantized_value = f32::from((*packed >> shift) & 0x0f);
                values.push(
                    scale * f32::from(group_scale) * quantized_value
                        - min_scale * f32::from(group_min),
                );
            }
        }
    }
    Ok(values)
}

fn decode_q5_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 176;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q5_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let min_scale = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
        let scales = &block[4..16];
        let high_bits = &block[16..48];
        let quantized = &block[48..];
        for group in 0..8 {
            let (group_scale, group_min) = q4_k_scale_min(group, scales);
            let byte_offset = (group / 2) * 32;
            let shift = (group % 2) * 4;
            let high_scale = group;
            let decoded_scale = scale * f32::from(group_scale);
            let decoded_min = min_scale * f32::from(group_min);
            for index in 0..32 {
                let low = (quantized[byte_offset + index] >> shift) & 0x0f;
                let high = (high_bits[index] >> high_scale) & 1;
                values.push(decoded_scale * f32::from(low | (high << 4)) - decoded_min);
            }
        }
    }
    Ok(values)
}

fn decode_q6_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 210;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q6_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = vec![0.0; blocks.len() * BLOCK_VALUES];
    for (block_index, block) in blocks.iter().enumerate() {
        let scale = f16_to_f32(u16::from_le_bytes([block[208], block[209]]));
        let ql = &block[..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let output = &mut values[block_index * BLOCK_VALUES..(block_index + 1) * BLOCK_VALUES];
        for chunk in 0..2 {
            let output_offset = chunk * 128;
            let low_offset = chunk * 64;
            let high_offset = chunk * 32;
            let scale_offset = chunk * 8;
            for l in 0..32 {
                let sub_block = l / 16;
                let q1 =
                    i8::try_from((ql[low_offset + l] & 0x0f) | ((qh[high_offset + l] & 0x03) << 4))
                        .unwrap_or_default()
                        - 32;
                let q2 = i8::try_from(
                    (ql[low_offset + l + 32] & 0x0f) | (((qh[high_offset + l] >> 2) & 0x03) << 4),
                )
                .unwrap_or_default()
                    - 32;
                let q3 = i8::try_from(
                    (ql[low_offset + l] >> 4) | (((qh[high_offset + l] >> 4) & 0x03) << 4),
                )
                .unwrap_or_default()
                    - 32;
                let q4 = i8::try_from(
                    (ql[low_offset + l + 32] >> 4) | (((qh[high_offset + l] >> 6) & 0x03) << 4),
                )
                .unwrap_or_default()
                    - 32;
                let scale_1 =
                    scale * f32::from(i8::from_ne_bytes([scales[scale_offset + sub_block]]));
                let scale_2 =
                    scale * f32::from(i8::from_ne_bytes([scales[scale_offset + sub_block + 2]]));
                let scale_3 =
                    scale * f32::from(i8::from_ne_bytes([scales[scale_offset + sub_block + 4]]));
                let scale_4 =
                    scale * f32::from(i8::from_ne_bytes([scales[scale_offset + sub_block + 6]]));
                output[output_offset + l] = scale_1 * f32::from(q1);
                output[output_offset + l + 32] = scale_2 * f32::from(q2);
                output[output_offset + l + 64] = scale_3 * f32::from(q3);
                output[output_offset + l + 96] = scale_4 * f32::from(q4);
            }
        }
    }
    Ok(values)
}

fn decode_q8_k(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 292;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "Q8_K tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        values.extend(
            block[4..260]
                .iter()
                .map(|value| scale * f32::from(i8::from_ne_bytes([*value]))),
        );
    }
    Ok(values)
}

fn q4_k_scale_min(group: usize, scales: &[u8]) -> (u8, u8) {
    if group < 4 {
        (scales[group] & 0x3f, scales[group + 4] & 0x3f)
    } else {
        (
            (scales[group + 4] & 0x0f) | ((scales[group - 4] >> 6) << 4),
            (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4),
        )
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = (u32::from(bits & 0x8000)) << 16;
    let exponent = u32::from((bits >> 10) & 0x1f);
    let fraction = u32::from(bits & 0x03ff);
    let value = match exponent {
        0 => {
            if fraction == 0 {
                sign
            } else {
                let mut fraction = fraction;
                let mut exponent = 127 - 14;
                while fraction & 0x0400 == 0 {
                    fraction <<= 1;
                    exponent -= 1;
                }
                sign | (exponent << 23) | ((fraction & 0x03ff) << 13)
            }
        }
        0x1f => sign | 0x7f80_0000 | (fraction << 13),
        exponent => sign | ((exponent + 112) << 23) | (fraction << 13),
    };
    f32::from_bits(value)
}

#[allow(unsafe_code)]
fn map_model(path: &Path, max_file_bytes: u64) -> Result<MappedFile, ModelError> {
    // SAFETY: the read-only mapping is held only while bytes are parsed or
    // copied. Callers do not mutate the backing file during that operation.
    unsafe { MappedFile::open(path, max_file_bytes) }
        .map_err(|error| ModelError::Io(error.to_string()))
}

fn metadata_string(gguf: &Gguf<'_>, key: &'static str) -> Result<Option<String>, ModelError> {
    match gguf.metadata_value(key) {
        None => Ok(None),
        Some(ggml_gguf::MetadataValue::Scalar(ggml_gguf::ScalarValue::String(value))) => {
            Ok(Some((*value).to_owned()))
        }
        Some(_) => Err(ModelError::InvalidUtf8Metadata(key)),
    }
}

fn owned_scalar(value: ggml_gguf::ScalarValue<'_>) -> MetadataScalar {
    match value {
        ggml_gguf::ScalarValue::U8(value) => MetadataScalar::U8(value),
        ggml_gguf::ScalarValue::I8(value) => MetadataScalar::I8(value),
        ggml_gguf::ScalarValue::U16(value) => MetadataScalar::U16(value),
        ggml_gguf::ScalarValue::I16(value) => MetadataScalar::I16(value),
        ggml_gguf::ScalarValue::U32(value) => MetadataScalar::U32(value),
        ggml_gguf::ScalarValue::I32(value) => MetadataScalar::I32(value),
        ggml_gguf::ScalarValue::F32(value) => MetadataScalar::F32(value),
        ggml_gguf::ScalarValue::Bool(value) => MetadataScalar::Bool(value),
        ggml_gguf::ScalarValue::String(value) => MetadataScalar::String(value.to_owned()),
        ggml_gguf::ScalarValue::U64(value) => MetadataScalar::U64(value),
        ggml_gguf::ScalarValue::I64(value) => MetadataScalar::I64(value),
        ggml_gguf::ScalarValue::F64(value) => MetadataScalar::F64(value),
    }
}

fn digest_bytes(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn encode_hex(bytes: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(64);
    for byte in bytes {
        encoded.push(HEX[usize::from(byte >> 4)] as char);
        encoded.push(HEX[usize::from(byte & 0x0f)] as char);
    }
    encoded
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    static NEXT_FILE: AtomicU64 = AtomicU64::new(0);

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn fixture(value_type: u32, shape: &[u64], tensor_bytes: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut bytes, "general.architecture");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "llama");
        push_string(&mut bytes, "general.name");
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(&mut bytes, "fixture");
        push_string(&mut bytes, "probe.tensor");
        bytes.extend_from_slice(&u32::try_from(shape.len()).unwrap().to_le_bytes());
        for dimension in shape {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&value_type.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes.extend_from_slice(tensor_bytes);
        while bytes.len() % 32 != 0 {
            bytes.push(0);
        }
        bytes
    }

    fn f32_fixture(value_type: u32, tensor_bytes: &[u8]) -> Vec<u8> {
        fixture(value_type, &[2, 2], tensor_bytes)
    }

    fn write_fixture(bytes: &[u8]) -> PathBuf {
        let id = NEXT_FILE.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("ggml-model-{id}.gguf"));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn assert_quantized_matmul_matches_materialized(
        value_type: u32,
        rows: u64,
        columns: u64,
        tensor_bytes: &[u8],
    ) {
        let rows = usize::try_from(rows).unwrap();
        let columns = usize::try_from(columns).unwrap();
        let input = (0..rows)
            .map(|index| f32::from(u8::try_from(index % 17).unwrap()) - 8.0)
            .collect::<Vec<_>>();
        let path = write_fixture(&fixture(
            value_type,
            &[
                u64::try_from(rows).unwrap(),
                u64::try_from(columns).unwrap(),
            ],
            tensor_bytes,
        ));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let materialized = model.load_f32("probe.tensor").unwrap();
        let expected = materialized
            .data()
            .chunks(rows)
            .map(|column| {
                column
                    .iter()
                    .zip(&input)
                    .map(|(weight, value)| weight * value)
                    .sum::<f32>()
            })
            .collect::<Vec<_>>();
        assert_eq!(expected.len(), columns);
        assert_eq!(
            model.matmul_f32_quantized("probe.tensor", &input).unwrap(),
            expected
        );
        fs::remove_file(path).unwrap();
    }

    fn patterned_bytes(length: usize, seed: u8) -> Vec<u8> {
        (0..length)
            .map(|index| u8::try_from((index * 37 + usize::from(seed) * 13) % 256).unwrap())
            .collect()
    }

    #[test]
    fn indexes_and_loads_f32_tensor() {
        let values = [1.0_f32, -2.0, 3.5, 7.25];
        let bytes = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<_>>();
        let path = write_fixture(&f32_fixture(0, &bytes));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.architecture(), Some("llama"));
        assert_eq!(model.name(), Some("fixture"));
        assert_eq!(
            model.metadata_scalar("general.name").unwrap(),
            Some(MetadataScalar::String("fixture".to_owned()))
        );
        assert_eq!(
            model
                .metadata_scalars(&["general.architecture", "general.name", "missing"])
                .unwrap(),
            vec![
                Some(MetadataScalar::String("llama".to_owned())),
                Some(MetadataScalar::String("fixture".to_owned())),
                None,
            ]
        );
        assert_eq!(model.tensors().len(), 1);
        assert_eq!(model.tensor("probe.tensor").unwrap().shape(), &[2, 2]);
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &values);
        let batched = model
            .load_f32_many(&["probe.tensor", "probe.tensor"])
            .unwrap();
        assert_eq!(batched.len(), 2);
        assert_eq!(batched[0].data(), &values);
        assert_eq!(batched[1].data(), &values);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_f16_tensor() {
        let encoded = [0x00, 0x3c, 0x00, 0xc0, 0x00, 0x40, 0x00, 0x44];
        let path = write_fixture(&f32_fixture(1, &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model.load_f32("probe.tensor").unwrap().data(),
            &[1.0, -2.0, 2.0, 4.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_bf16_tensor() {
        let encoded = [
            0x80, 0x3f, // 1.0
            0x20, 0xc0, // -2.5
            0x00, 0x00, // 0.0
            0x48, 0x41, // 12.5
        ];
        let path = write_fixture(&f32_fixture(30, &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model.load_f32("probe.tensor").unwrap().data(),
            &[1.0, -2.5, 0.0, 12.5]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q4_0_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0x78, 16));
        let path = write_fixture(&fixture(2, &[32], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(&values.data()[..16], &[0.0; 16]);
        assert_eq!(&values.data()[16..], &[-1.0; 16]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q4_1_tensor() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0x00, 16));
        let path = write_fixture(&fixture(3, &[32], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &[1.0; 32]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q8_0_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend((0_u8..32).map(|value| value.wrapping_add(0x80)));
        let path = write_fixture(&fixture(8, &[32], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data()[0].to_bits(), (-128.0_f32).to_bits());
        assert_eq!(values.data()[1].to_bits(), (-127.0_f32).to_bits());
        assert_eq!(values.data()[31].to_bits(), (-97.0_f32).to_bits());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q4_0_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0x99, 16));
        encoded.extend([0x00, 0x3c]);
        encoded.extend(std::iter::repeat_n(0xaa, 16));
        let path = write_fixture(&fixture(2, &[32, 2], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 32])
                .unwrap(),
            &[32.0, 64.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn loads_quantized_matrix_for_columns_and_direct_products() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0x99, 16));
        encoded.extend([0x00, 0x3c]);
        encoded.extend(std::iter::repeat_n(0xaa, 16));
        let path = write_fixture(&fixture(2, &[32, 2], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.rows(), 32);
        assert_eq!(matrix.columns(), 2);
        assert_eq!(matrix.value_type().raw(), 2);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 32]);
        assert_eq!(matrix.column(1).unwrap(), vec![2.0; 32]);
        assert_eq!(matrix.matmul_f32(&[1.0; 32]).unwrap(), &[32.0, 64.0]);
        let batched = model
            .load_quantized_many(&["probe.tensor", "probe.tensor"])
            .unwrap();
        assert_eq!(batched, vec![matrix.clone(), matrix]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn converts_q4_0_matrix_directly_to_mlx_affine_layout() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend((0_u8..16).map(|value| value | (value << 4)));
        let path = write_fixture(&fixture(2, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let quantized = model.load_affine_quantized("probe.tensor", 32, 4).unwrap();
        assert_eq!(quantized.rows(), 1);
        assert_eq!(quantized.columns(), 32);
        assert_eq!(quantized.group_size(), 32);
        assert_eq!(quantized.bits(), 4);
        assert_eq!(quantized.scales(), &[1.0]);
        assert_eq!(quantized.biases(), &[-8.0]);
        assert_eq!(
            quantized.packed(),
            &[0x7654_3210, 0xfedc_ba98, 0x7654_3210, 0xfedc_ba98]
        );

        let names = ["probe.tensor"];
        let mut observed = None;
        model
            .for_each_tensor(&names, 32, 4, |_, tensor| {
                observed = Some(tensor);
                Ok::<(), ModelError>(())
            })
            .unwrap();
        assert!(matches!(observed, Some(LoadedTensor::AffineQuantized(_))));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_unsupported_mlx_affine_parameters() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 16));
        let path = write_fixture(&fixture(2, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.load_affine_quantized("probe.tensor", 16, 4),
            Err(ModelError::Shape(_))
        ));
        assert!(matches!(
            model.load_affine_quantized("probe.tensor", 32, 7),
            Err(ModelError::Shape(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q8_0_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend((-16_i8..16_i8).map(|value| value.to_ne_bytes()[0]));
        encoded.extend([0x00, 0x3c]);
        encoded.extend((16_i8..48_i8).map(|value| value.to_ne_bytes()[0]));
        let path = write_fixture(&fixture(8, &[32, 2], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 32])
                .unwrap(),
            &[-16.0, 1008.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q4_1_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x40];
        encoded.extend(std::iter::repeat_n(0x10, 16));
        let path = write_fixture(&fixture(3, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 32])
                .unwrap(),
            &[80.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q5_0_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c, 0, 0, 0, 0];
        encoded.extend(std::iter::repeat_n(0x10, 16));
        let path = write_fixture(&fixture(6, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 32])
                .unwrap(),
            &[-496.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q5_1_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x40, 0, 0, 0, 0];
        encoded.extend(std::iter::repeat_n(0x10, 16));
        let path = write_fixture(&fixture(7, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 32])
                .unwrap(),
            &[80.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q4_k_matrix_without_materializing_f32_values() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x00];
        encoded.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        encoded.extend(std::iter::repeat_n(0x10, 128));
        let path = write_fixture(&fixture(12, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model
                .matmul_f32_quantized("probe.tensor", &[1.0; 256])
                .unwrap(),
            &[128.0]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn multiplies_q2_k_matrix_with_materialized_parity() {
        let mut block = patterned_bytes(84, 7);
        block[80..82].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[82..84].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut encoded = block.clone();
        encoded.extend(patterned_bytes(84, 19));
        encoded[164..166].copy_from_slice(&0x3c00_u16.to_le_bytes());
        encoded[166..168].copy_from_slice(&0x3c00_u16.to_le_bytes());
        assert_quantized_matmul_matches_materialized(10, 256, 2, &encoded);
    }

    #[test]
    fn multiplies_q3_k_matrix_with_materialized_parity() {
        let mut block = patterned_bytes(110, 11);
        block[108..110].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut encoded = block.clone();
        encoded.extend(patterned_bytes(110, 23));
        encoded[218..220].copy_from_slice(&0x3c00_u16.to_le_bytes());
        assert_quantized_matmul_matches_materialized(11, 256, 2, &encoded);
    }

    #[test]
    fn multiplies_q5_k_matrix_with_materialized_parity() {
        let mut block = patterned_bytes(176, 13);
        block[0..2].copy_from_slice(&0x3c00_u16.to_le_bytes());
        block[2..4].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut encoded = block.clone();
        encoded.extend(patterned_bytes(176, 29));
        encoded[176..178].copy_from_slice(&0x3c00_u16.to_le_bytes());
        encoded[178..180].copy_from_slice(&0x3c00_u16.to_le_bytes());
        assert_quantized_matmul_matches_materialized(13, 256, 2, &encoded);
    }

    #[test]
    fn multiplies_q6_k_matrix_with_materialized_parity() {
        let mut block = patterned_bytes(210, 17);
        block[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let mut encoded = block.clone();
        encoded.extend(patterned_bytes(210, 31));
        encoded[418..420].copy_from_slice(&0x3c00_u16.to_le_bytes());
        assert_quantized_matmul_matches_materialized(14, 256, 2, &encoded);
    }

    #[test]
    fn multiplies_q8_k_matrix_with_materialized_parity() {
        let mut block = patterned_bytes(292, 19);
        block[..4].copy_from_slice(&1.0_f32.to_le_bytes());
        let mut encoded = block.clone();
        encoded.extend(patterned_bytes(292, 37));
        encoded[292..296].copy_from_slice(&1.0_f32.to_le_bytes());
        assert_quantized_matmul_matches_materialized(15, 256, 2, &encoded);
    }

    #[test]
    fn rejects_unsupported_quantized_matmul_inputs() {
        let encoded = vec![0_u8; 32 * 4];
        let path = write_fixture(&fixture(0, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.matmul_f32_quantized("probe.tensor", &[1.0; 32]),
            Err(ModelError::UnsupportedTensorType { .. })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_finite_quantized_matmul_values() {
        let mut encoded = vec![0_u8; 292];
        encoded[..4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        let path = write_fixture(&fixture(15, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.matmul_f32_quantized("probe.tensor", &[1.0; 256]),
            Err(ModelError::Shape(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_finite_quantized_matmul_input() {
        let path = write_fixture(&fixture(10, &[256, 1], &[0; 84]));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.matmul_f32_quantized("probe.tensor", &[f32::NAN; 256]),
            Err(ModelError::Shape(_))
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q8_k_tensor() {
        let mut encoded = 1.0_f32.to_le_bytes().to_vec();
        encoded.extend((0_u8..=255).map(|value| value.wrapping_add(0x80)));
        encoded.extend(std::iter::repeat_n(0_u8, 32));
        let path = write_fixture(&fixture(15, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data()[0].to_bits(), (-128.0_f32).to_bits());
        assert_eq!(values.data()[1].to_bits(), (-127.0_f32).to_bits());
        assert_eq!(values.data()[255].to_bits(), 127.0_f32.to_bits());
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q5_0_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend([0, 0, 0, 0]);
        encoded.extend(std::iter::repeat_n(0x00, 16));
        let path = write_fixture(&fixture(6, &[32], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &[-16.0; 32]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q5_1_tensor() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x3c];
        encoded.extend([0, 0, 0, 0]);
        encoded.extend(std::iter::repeat_n(0x00, 16));
        let path = write_fixture(&fixture(7, &[32], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &[1.0; 32]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q4_k_tensor() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x00];
        encoded.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        encoded.extend(std::iter::repeat_n(0x10, 128));
        let path = write_fixture(&fixture(12, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        for group in 0..8 {
            let start = group * 32;
            let expected: f32 = if group % 2 == 0 { 0.0 } else { 1.0 };
            assert!(
                values.data()[start..start + 32]
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q5_k_tensor() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x3c];
        encoded.extend([1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1]);
        encoded.extend(std::iter::repeat_n(0, 32));
        encoded.extend(std::iter::repeat_n(0x10, 128));
        let path = write_fixture(&fixture(13, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        for group in 0..8 {
            let start = group * 32;
            let expected: f32 = if group % 2 == 0 { 0.0 } else { 1.0 };
            assert!(
                values.data()[start..start + 32]
                    .iter()
                    .all(|value| value.to_bits() == expected.to_bits())
            );
        }
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q2_k_tensor() {
        let mut encoded = vec![1_u8; 16];
        encoded.extend(std::iter::repeat_n(0x55, 64));
        encoded.extend([0x00, 0x3c, 0x00, 0x00]);
        let path = write_fixture(&fixture(10, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &[1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q3_k_tensor() {
        let mut encoded = vec![0_u8; 32];
        encoded.extend(std::iter::repeat_n(0, 64));
        encoded.extend(std::iter::repeat_n(0x11, 8));
        encoded.extend(std::iter::repeat_n(0xaa, 4));
        encoded.extend([0x00, 0x3c]);
        let path = write_fixture(&fixture(11, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &[-4.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_q6_k_tensor() {
        let mut encoded = vec![0_u8; 210];
        encoded[192..208].fill(1);
        encoded[208..210].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let path = write_fixture(&fixture(14, &[256], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            model.load_f32("probe.tensor").unwrap().data(),
            &[-32.0; 256]
        );
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_unsupported_tensor_materialization() {
        let path = write_fixture(&fixture(16, &[256], &[0; 66]));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.load_f32("probe.tensor"),
            Err(ModelError::UnsupportedTensorType { .. })
        ));
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn detects_same_length_model_rewrite() {
        let path = write_fixture(&f32_fixture(0, &[0; 16]));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let mut changed = fs::read(&path).unwrap();
        *changed.last_mut().unwrap() ^= 1;
        fs::write(&path, changed).unwrap();
        assert!(matches!(
            model.load_f32("probe.tensor"),
            Err(ModelError::ContentChanged)
        ));
        fs::remove_file(path).unwrap();
    }
}
