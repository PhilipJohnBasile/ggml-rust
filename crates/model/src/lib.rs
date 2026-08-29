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
        let worker_count = std::thread::available_parallelism()
            .map_or(1, std::num::NonZeroUsize::get)
            .min(16)
            .min(self.columns);
        if worker_count <= 1 || self.columns < 2 {
            return self.matmul_columns(input, 0, self.columns);
        }
        let chunk_width = self.columns.div_ceil(worker_count);
        let mut output = vec![0.0_f32; self.columns];
        std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(worker_count);
            for start in (0..self.columns).step_by(chunk_width) {
                let end = (start + chunk_width).min(self.columns);
                handles.push(scope.spawn(move || {
                    self.matmul_columns(input, start, end)
                        .map(|values| (start, values))
                }));
            }
            for handle in handles {
                let (start, values) = handle.join().map_err(|_| {
                    ModelError::Shape("quantized matmul worker panicked".to_owned())
                })??;
                output[start..start + values.len()].copy_from_slice(&values);
            }
            Ok(output)
        })
    }

    fn matmul_columns(
        &self,
        input: &[f32],
        start: usize,
        end: usize,
    ) -> Result<Vec<f32>, ModelError> {
        let mut output = Vec::with_capacity(end.saturating_sub(start));
        for column in start..end {
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

    /// Reads a bounded boolean metadata array while enforcing the model digest.
    ///
    /// # Errors
    ///
    /// Returns an error when the model changed, the metadata is malformed, the
    /// array type is not boolean, or the caller's element bound is exceeded.
    pub fn metadata_bool_array(
        &self,
        key: &str,
        max_elements: u64,
    ) -> Result<Option<Vec<bool>>, ModelError> {
        self.with_validated_gguf(|gguf| {
            let Some(value) = gguf.metadata_value(key) else {
                return Ok(None);
            };
            let ggml_gguf::MetadataValue::Array(array) = value else {
                return Err(ModelError::MetadataArray(key.to_owned()));
            };
            if array.element_type() != ggml_gguf::MetadataType::Bool {
                return Err(ModelError::MetadataArrayType {
                    key: key.to_owned(),
                    expected: "Bool",
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
                let Some(ggml_gguf::ScalarValue::Bool(value)) = array.get(index) else {
                    return Err(ModelError::Parse(format!(
                        "GGUF metadata {key} boolean array contains an invalid element"
                    )));
                };
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
    /// `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `NVFP4`, `TQ1_0`, and `TQ2_0` storage are supported. Quantized
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
    /// `Q5_0`, `Q5_1`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `Q8_K`,
    /// `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ4_NL`, `IQ4_XS`, `MXFP4`, `NVFP4`, `TQ1_0`, and `TQ2_0` are supported. The operation walks the encoded blocks directly
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
            0 | 1
                | 2
                | 3
                | 6
                | 7
                | 8
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
                | 22
                | 23
                | 21
                | 30
                | 39
                | 40
                | 34
                | 35
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

// The IQ1_S codebook is the fixed 2048-entry lattice used by GGML. Each
// entry packs eight signed 8-bit values in little-endian order.
#[allow(clippy::unreadable_literal)]
const IQ1_S_GRID: [u64; 2048] = [
    0xffffffffffffffff,
    0xffffffffffffff01,
    0xffffffffffff0000,
    0xffffffffffff01ff,
    0xffffffffffff0101,
    0xffffffffff00ff00,
    0xffffffffff000000,
    0xffffffffff01ffff,
    0xffffffffff01ff01,
    0xffffffffff0101ff,
    0xffffffffff010101,
    0xffffffff00ff0000,
    0xffffffff0000ff00,
    0xffffffff000000ff,
    0xffffffff00000001,
    0xffffffff00010000,
    0xffffffff01ffffff,
    0xffffffff01ffff01,
    0xffffffff01ff01ff,
    0xffffffff01ff0101,
    0xffffffff01000000,
    0xffffffff0101ffff,
    0xffffffff0101ff01,
    0xffffffff010101ff,
    0xffffffff01010101,
    0xffffff00ffff00ff,
    0xffffff00ffff0000,
    0xffffff00ff00ff00,
    0xffffff00ff0000ff,
    0xffffff00ff000001,
    0xffffff00ff000100,
    0xffffff00ff000101,
    0xffffff00ff010000,
    0xffffff0000ffff00,
    0xffffff0000ff0001,
    0xffffff0000ff0100,
    0xffffff000000ff01,
    0xffffff0000000000,
    0xffffff0000000101,
    0xffffff000001ff00,
    0xffffff00000100ff,
    0xffffff0000010001,
    0xffffff00000101ff,
    0xffffff0001ff0000,
    0xffffff000100ff00,
    0xffffff00010000ff,
    0xffffff0001000001,
    0xffffff0001010000,
    0xffffff01ffffffff,
    0xffffff01ffffff01,
    0xffffff01ffff01ff,
    0xffffff01ffff0101,
    0xffffff01ff000000,
    0xffffff01ff01ffff,
    0xffffff01ff01ff01,
    0xffffff01ff0101ff,
    0xffffff01ff010101,
    0xffffff0100ff0000,
    0xffffff010000ff00,
    0xffffff0100000100,
    0xffffff01000100ff,
    0xffffff0100010100,
    0xffffff0101ffffff,
    0xffffff0101ffff01,
    0xffffff0101ff01ff,
    0xffffff0101ff0101,
    0xffffff010100ff00,
    0xffffff0101000000,
    0xffffff0101000100,
    0xffffff010101ffff,
    0xffffff010101ff01,
    0xffffff01010101ff,
    0xffffff0101010101,
    0xffff00ffff00ff00,
    0xffff00ffff0000ff,
    0xffff00ffff000001,
    0xffff00ffff010000,
    0xffff00ff00ffff00,
    0xffff00ff00ff0100,
    0xffff00ff00000000,
    0xffff00ff00000101,
    0xffff00ff000100ff,
    0xffff00ff00010000,
    0xffff00ff0100ff00,
    0xffff00ff01000100,
    0xffff00ff01010000,
    0xffff0000ffffff00,
    0xffff0000ffff00ff,
    0xffff0000ffff0000,
    0xffff0000ffff0001,
    0xffff0000ff000000,
    0xffff0000ff0001ff,
    0xffff0000ff000101,
    0xffff0000ff010100,
    0xffff000000ffffff,
    0xffff000000ff0000,
    0xffff000000ff0101,
    0xffff00000000ffff,
    0xffff00000000ff00,
    0xffff0000000000ff,
    0xffff000000000000,
    0xffff000000000001,
    0xffff000000000100,
    0xffff00000001ffff,
    0xffff00000001ff01,
    0xffff000000010000,
    0xffff0000000101ff,
    0xffff000000010101,
    0xffff000001ffff00,
    0xffff00000100ff00,
    0xffff000001000000,
    0xffff0000010001ff,
    0xffff000001000101,
    0xffff00000101ff00,
    0xffff0000010100ff,
    0xffff000001010000,
    0xffff000001010001,
    0xffff000001010100,
    0xffff0001ff0000ff,
    0xffff0001ff000100,
    0xffff000100ffff00,
    0xffff000100ff00ff,
    0xffff00010000ffff,
    0xffff00010000ff01,
    0xffff000100000000,
    0xffff0001000001ff,
    0xffff00010001ffff,
    0xffff00010001ff00,
    0xffff000100010001,
    0xffff000100010100,
    0xffff000101ff0000,
    0xffff00010100ff00,
    0xffff0001010000ff,
    0xffff000101000100,
    0xffff01ffffffffff,
    0xffff01ffffffff01,
    0xffff01ffffff01ff,
    0xffff01ffffff0101,
    0xffff01ffff000000,
    0xffff01ffff01ffff,
    0xffff01ffff01ff01,
    0xffff01ffff0101ff,
    0xffff01ffff010101,
    0xffff01ff00ff0000,
    0xffff01ff0000ff00,
    0xffff01ff00000001,
    0xffff01ff00010000,
    0xffff01ff01ffffff,
    0xffff01ff01ffff01,
    0xffff01ff01ff01ff,
    0xffff01ff01ff0101,
    0xffff01ff01000000,
    0xffff01ff0101ffff,
    0xffff01ff0101ff01,
    0xffff01ff010101ff,
    0xffff01ff01010101,
    0xffff0100ffff0000,
    0xffff0100ff00ff00,
    0xffff0100ff0000ff,
    0xffff0100ff000100,
    0xffff0100ff0100ff,
    0xffff0100ff010000,
    0xffff010000ffff00,
    0xffff01000000ffff,
    0xffff01000000ff00,
    0xffff010000000000,
    0xffff01000001ff00,
    0xffff0100000100ff,
    0xffff010000010100,
    0xffff01000100ff00,
    0xffff0100010000ff,
    0xffff010001000001,
    0xffff010001000100,
    0xffff010001010000,
    0xffff0101ffffffff,
    0xffff0101ffffff01,
    0xffff0101ffff01ff,
    0xffff0101ffff0101,
    0xffff0101ff000000,
    0xffff0101ff01ffff,
    0xffff0101ff01ff01,
    0xffff0101ff0101ff,
    0xffff0101ff010101,
    0xffff010100ff0000,
    0xffff01010000ff00,
    0xffff010100000100,
    0xffff01010001ff00,
    0xffff010100010000,
    0xffff010101ffffff,
    0xffff010101ffff01,
    0xffff010101ff0000,
    0xffff010101ff01ff,
    0xffff010101ff0101,
    0xffff010101000000,
    0xffff01010101ffff,
    0xffff01010101ff01,
    0xffff0101010101ff,
    0xffff010101010101,
    0xff00ffffff00ffff,
    0xff00ffffff00ff00,
    0xff00ffffff0000ff,
    0xff00ffffff000100,
    0xff00ffffff0100ff,
    0xff00ffffff010000,
    0xff00ffff00ffff00,
    0xff00ffff00ff00ff,
    0xff00ffff0000ffff,
    0xff00ffff00000000,
    0xff00ffff000001ff,
    0xff00ffff0001ff00,
    0xff00ffff000100ff,
    0xff00ffff00010000,
    0xff00ffff00010100,
    0xff00ffff0100ff00,
    0xff00ffff010000ff,
    0xff00ffff01000001,
    0xff00ffff0101ff00,
    0xff00ffff01010000,
    0xff00ff00ffffff00,
    0xff00ff00ffff00ff,
    0xff00ff00ffff0001,
    0xff00ff00ffff0100,
    0xff00ff00ff00ffff,
    0xff00ff00ff00ff01,
    0xff00ff00ff000000,
    0xff00ff00ff0001ff,
    0xff00ff00ff01ff00,
    0xff00ff00ff0100ff,
    0xff00ff00ff010100,
    0xff00ff0000ff0000,
    0xff00ff0000ff0101,
    0xff00ff000000ffff,
    0xff00ff000000ff00,
    0xff00ff000000ff01,
    0xff00ff00000000ff,
    0xff00ff0000000000,
    0xff00ff0000000001,
    0xff00ff0000000100,
    0xff00ff000001ffff,
    0xff00ff0000010000,
    0xff00ff0001ff00ff,
    0xff00ff000100ff01,
    0xff00ff0001000000,
    0xff00ff000101ff00,
    0xff00ff00010100ff,
    0xff00ff01ff00ff00,
    0xff00ff01ff0000ff,
    0xff00ff01ff000001,
    0xff00ff01ff010000,
    0xff00ff0100ffffff,
    0xff00ff0100ff0001,
    0xff00ff0100ff0100,
    0xff00ff010000ff01,
    0xff00ff0100000000,
    0xff00ff01000001ff,
    0xff00ff0100000101,
    0xff00ff01000100ff,
    0xff00ff0100010001,
    0xff00ff0101ff0000,
    0xff00ff010100ff00,
    0xff00ff01010000ff,
    0xff00ff0101000001,
    0xff00ff0101010000,
    0xff0000ffffffff00,
    0xff0000ffffff0001,
    0xff0000ffffff0100,
    0xff0000ffff0000ff,
    0xff0000ffff000000,
    0xff0000ffff0001ff,
    0xff0000ffff000100,
    0xff0000ffff01ff00,
    0xff0000ffff010001,
    0xff0000ff00ffff00,
    0xff0000ff00ff0000,
    0xff0000ff00ff0001,
    0xff0000ff00ff01ff,
    0xff0000ff00ff0101,
    0xff0000ff0000ff00,
    0xff0000ff000000ff,
    0xff0000ff00000000,
    0xff0000ff00000001,
    0xff0000ff00000100,
    0xff0000ff0001ff01,
    0xff0000ff00010000,
    0xff0000ff000101ff,
    0xff0000ff01ff00ff,
    0xff0000ff01ff0100,
    0xff0000ff0100ffff,
    0xff0000ff010000ff,
    0xff0000ff01000000,
    0xff0000ff010001ff,
    0xff0000ff01000100,
    0xff0000ff01000101,
    0xff0000ff0101ff00,
    0xff0000ff010100ff,
    0xff0000ff01010000,
    0xff0000ff01010100,
    0xff000000ffffff01,
    0xff000000ffff0000,
    0xff000000ffff0101,
    0xff000000ff00ff00,
    0xff000000ff0000ff,
    0xff000000ff000000,
    0xff000000ff000001,
    0xff000000ff000100,
    0xff000000ff01ffff,
    0xff000000ff01ff01,
    0xff000000ff010000,
    0xff000000ff0101ff,
    0xff000000ff010101,
    0xff00000000ffff00,
    0xff00000000ff00ff,
    0xff00000000ff0000,
    0xff00000000ff0001,
    0xff0000000000ff00,
    0xff0000000000ff01,
    0xff000000000000ff,
    0xff00000000000000,
    0xff00000000000001,
    0xff00000000000100,
    0xff00000000000101,
    0xff0000000001ff00,
    0xff000000000100ff,
    0xff00000000010000,
    0xff00000000010001,
    0xff00000000010100,
    0xff00000001ffffff,
    0xff00000001ffff01,
    0xff00000001ff00ff,
    0xff00000001ff0000,
    0xff00000001ff01ff,
    0xff00000001ff0101,
    0xff0000000100ffff,
    0xff0000000100ff00,
    0xff000000010000ff,
    0xff00000001000000,
    0xff00000001000001,
    0xff00000001000100,
    0xff00000001000101,
    0xff0000000101ffff,
    0xff0000000101ff01,
    0xff00000001010000,
    0xff000001ffffff00,
    0xff000001ffff00ff,
    0xff000001ffff0000,
    0xff000001ffff0001,
    0xff000001ff000000,
    0xff000001ff000001,
    0xff000001ff0001ff,
    0xff000001ff000101,
    0xff000001ff01ff00,
    0xff000001ff010001,
    0xff00000100ffffff,
    0xff00000100ffff01,
    0xff00000100ff00ff,
    0xff00000100ff0000,
    0xff00000100ff01ff,
    0xff00000100ff0101,
    0xff0000010000ff00,
    0xff00000100000000,
    0xff00000100000001,
    0xff000001000001ff,
    0xff00000100000100,
    0xff0000010001ff00,
    0xff000001000100ff,
    0xff00000100010000,
    0xff000001000101ff,
    0xff00000100010100,
    0xff00000100010101,
    0xff00000101ff0001,
    0xff00000101ff0101,
    0xff0000010100ff01,
    0xff00000101000000,
    0xff000001010100ff,
    0xff00000101010100,
    0xff0001ffff00ff00,
    0xff0001ffff000001,
    0xff0001ffff010000,
    0xff0001ff00ffff00,
    0xff0001ff00ff00ff,
    0xff0001ff00ff0001,
    0xff0001ff00ff0100,
    0xff0001ff0000ffff,
    0xff0001ff00000000,
    0xff0001ff000001ff,
    0xff0001ff00000101,
    0xff0001ff0001ffff,
    0xff0001ff0001ff00,
    0xff0001ff000100ff,
    0xff0001ff00010001,
    0xff0001ff00010100,
    0xff0001ff01ff0000,
    0xff0001ff0100ff00,
    0xff0001ff010000ff,
    0xff0001ff01010000,
    0xff000100ff00ffff,
    0xff000100ff00ff01,
    0xff000100ff000000,
    0xff000100ff000101,
    0xff000100ff01ff00,
    0xff000100ff010000,
    0xff00010000ffff01,
    0xff00010000ff00ff,
    0xff00010000ff0000,
    0xff00010000ff01ff,
    0xff0001000000ff00,
    0xff000100000000ff,
    0xff00010000000000,
    0xff00010000000001,
    0xff00010000000100,
    0xff00010000000101,
    0xff0001000001ffff,
    0xff00010000010000,
    0xff00010000010101,
    0xff00010001ff0100,
    0xff0001000100ff00,
    0xff0001000100ff01,
    0xff00010001000000,
    0xff000100010001ff,
    0xff0001000101ff00,
    0xff00010001010001,
    0xff00010001010100,
    0xff000101ffff0100,
    0xff000101ff000001,
    0xff000101ff0100ff,
    0xff000101ff010001,
    0xff00010100ff00ff,
    0xff00010100ff0001,
    0xff00010100ff0100,
    0xff0001010000ffff,
    0xff0001010000ff01,
    0xff00010100000000,
    0xff000101000001ff,
    0xff0001010001ff00,
    0xff00010100010001,
    0xff00010100010100,
    0xff00010101ff0000,
    0xff0001010100ff00,
    0xff00010101000001,
    0xff00010101000101,
    0xff01ffffffffffff,
    0xff01ffffffffff01,
    0xff01ffffffff01ff,
    0xff01ffffffff0101,
    0xff01ffffff000000,
    0xff01ffffff01ffff,
    0xff01ffffff01ff01,
    0xff01ffffff010000,
    0xff01ffffff0101ff,
    0xff01ffffff010101,
    0xff01ffff00ff0000,
    0xff01ffff0000ff00,
    0xff01ffff00000100,
    0xff01ffff0001ff00,
    0xff01ffff00010000,
    0xff01ffff01ffffff,
    0xff01ffff01ffff01,
    0xff01ffff01ff01ff,
    0xff01ffff01ff0101,
    0xff01ffff01000000,
    0xff01ffff0101ffff,
    0xff01ffff0101ff01,
    0xff01ffff01010000,
    0xff01ffff010101ff,
    0xff01ffff01010101,
    0xff01ff00ffff0000,
    0xff01ff00ff00ff00,
    0xff01ff00ff0000ff,
    0xff01ff00ff000100,
    0xff01ff00ff010000,
    0xff01ff0000ffff01,
    0xff01ff0000ff00ff,
    0xff01ff0000ff0100,
    0xff01ff0000000000,
    0xff01ff00000001ff,
    0xff01ff0000000101,
    0xff01ff000001ff00,
    0xff01ff00000100ff,
    0xff01ff0000010000,
    0xff01ff0000010001,
    0xff01ff0001ff0000,
    0xff01ff000100ffff,
    0xff01ff0001000001,
    0xff01ff0001000100,
    0xff01ff0001010000,
    0xff01ff01ffffff00,
    0xff01ff01ffff01ff,
    0xff01ff01ffff0101,
    0xff01ff01ff00ff00,
    0xff01ff01ff000000,
    0xff01ff01ff01ffff,
    0xff01ff01ff01ff01,
    0xff01ff01ff0101ff,
    0xff01ff01ff010101,
    0xff01ff0100ff0000,
    0xff01ff010000ff00,
    0xff01ff0100000001,
    0xff01ff0100000100,
    0xff01ff0100010000,
    0xff01ff0101ffff00,
    0xff01ff0101ff01ff,
    0xff01ff0101ff0101,
    0xff01ff010100ff00,
    0xff01ff0101000000,
    0xff01ff010101ffff,
    0xff01ff010101ff01,
    0xff01ff01010101ff,
    0xff01ff0101010101,
    0xff0100ffffff0000,
    0xff0100ffff0000ff,
    0xff0100ffff000001,
    0xff0100ffff000100,
    0xff0100ffff010000,
    0xff0100ff00ff00ff,
    0xff0100ff00ff0000,
    0xff0100ff00ff0001,
    0xff0100ff00ff0100,
    0xff0100ff0000ff01,
    0xff0100ff00000000,
    0xff0100ff000001ff,
    0xff0100ff00000101,
    0xff0100ff00010001,
    0xff0100ff01ff0000,
    0xff0100ff0100ff00,
    0xff0100ff010000ff,
    0xff0100ff01000100,
    0xff0100ff0101ff00,
    0xff0100ff01010000,
    0xff010000ffff0100,
    0xff010000ff000000,
    0xff010000ff01ff00,
    0xff010000ff010100,
    0xff01000000ffffff,
    0xff01000000ff0000,
    0xff01000000ff01ff,
    0xff0100000000ff00,
    0xff010000000000ff,
    0xff01000000000000,
    0xff01000000000100,
    0xff0100000001ff01,
    0xff01000000010000,
    0xff010000000101ff,
    0xff01000001ff0100,
    0xff0100000100ffff,
    0xff010000010000ff,
    0xff01000001000000,
    0xff010000010001ff,
    0xff01000001000101,
    0xff0100000101ff00,
    0xff010000010100ff,
    0xff01000001010001,
    0xff01000001010100,
    0xff010001ffff0000,
    0xff010001ff00ffff,
    0xff010001ff00ff01,
    0xff010001ff000100,
    0xff010001ff010000,
    0xff01000100ffff00,
    0xff01000100ff0100,
    0xff01000100000000,
    0xff0100010001ffff,
    0xff0100010001ff00,
    0xff01000100010100,
    0xff01000101ff00ff,
    0xff01000101ff0001,
    0xff0100010100ffff,
    0xff01000101000101,
    0xff0101ffffffffff,
    0xff0101ffffffff01,
    0xff0101ffffff01ff,
    0xff0101ffffff0101,
    0xff0101ffff000000,
    0xff0101ffff01ffff,
    0xff0101ffff01ff01,
    0xff0101ffff0101ff,
    0xff0101ffff010101,
    0xff0101ff00ff0000,
    0xff0101ff0000ff00,
    0xff0101ff000000ff,
    0xff0101ff00010000,
    0xff0101ff01ffffff,
    0xff0101ff01ffff01,
    0xff0101ff01ff01ff,
    0xff0101ff01ff0101,
    0xff0101ff0101ffff,
    0xff0101ff0101ff01,
    0xff0101ff010101ff,
    0xff0101ff01010101,
    0xff010100ffff0100,
    0xff010100ff00ff00,
    0xff010100ff0000ff,
    0xff010100ff000100,
    0xff010100ff010000,
    0xff01010000ff0001,
    0xff01010000ff0100,
    0xff0101000000ff01,
    0xff01010000000000,
    0xff0101000001ff00,
    0xff010100000100ff,
    0xff01010000010001,
    0xff01010000010100,
    0xff01010001ff0000,
    0xff0101000100ffff,
    0xff01010001000001,
    0xff01010001000100,
    0xff010100010100ff,
    0xff01010001010000,
    0xff010101ffffffff,
    0xff010101ffffff01,
    0xff010101ffff01ff,
    0xff010101ffff0101,
    0xff010101ff01ffff,
    0xff010101ff01ff01,
    0xff010101ff0101ff,
    0xff010101ff010101,
    0xff01010100ff0000,
    0xff0101010000ff00,
    0xff01010100000001,
    0xff01010100000100,
    0xff01010100010000,
    0xff01010101ffffff,
    0xff01010101ffff01,
    0xff01010101ff01ff,
    0xff01010101ff0101,
    0xff01010101000000,
    0xff0101010101ffff,
    0xff0101010101ff01,
    0xff010101010101ff,
    0xff01010101010101,
    0x00ffffffffff0000,
    0x00ffffffff00ff00,
    0x00ffffffff000001,
    0x00ffffffff010000,
    0x00ffffff00ff0100,
    0x00ffffff0000ff01,
    0x00ffffff00000000,
    0x00ffffff000001ff,
    0x00ffffff00000101,
    0x00ffffff0001ff00,
    0x00ffffff000100ff,
    0x00ffffff00010001,
    0x00ffffff010000ff,
    0x00ffffff01000100,
    0x00ffffff0101ff00,
    0x00ffffff01010001,
    0x00ffff00ffffffff,
    0x00ffff00ffffff00,
    0x00ffff00ffff00ff,
    0x00ffff00ffff0001,
    0x00ffff00ffff0100,
    0x00ffff00ff00ff01,
    0x00ffff00ff000000,
    0x00ffff00ff000001,
    0x00ffff00ff0001ff,
    0x00ffff00ff000101,
    0x00ffff00ff01ff00,
    0x00ffff00ff010001,
    0x00ffff00ff010100,
    0x00ffff0000ff0000,
    0x00ffff0000ff01ff,
    0x00ffff0000ff0101,
    0x00ffff000000ff00,
    0x00ffff00000000ff,
    0x00ffff0000000000,
    0x00ffff0000000001,
    0x00ffff0000000100,
    0x00ffff0000000101,
    0x00ffff0000010000,
    0x00ffff00000101ff,
    0x00ffff0000010101,
    0x00ffff0001ffff00,
    0x00ffff0001ff00ff,
    0x00ffff0001ff0001,
    0x00ffff000100ffff,
    0x00ffff000100ff01,
    0x00ffff0001000000,
    0x00ffff000101ffff,
    0x00ffff000101ff00,
    0x00ffff000101ff01,
    0x00ffff01ffff0000,
    0x00ffff01ff00ff00,
    0x00ffff01ff0000ff,
    0x00ffff01ff000001,
    0x00ffff01ff010000,
    0x00ffff0100ffff00,
    0x00ffff010000ff01,
    0x00ffff0100000000,
    0x00ffff0100000101,
    0x00ffff01000100ff,
    0x00ffff0100010100,
    0x00ffff0101ff0100,
    0x00ffff01010000ff,
    0x00ffff0101010000,
    0x00ff00ffffffff00,
    0x00ff00ffff000000,
    0x00ff00ffff000100,
    0x00ff00ffff010100,
    0x00ff00ff00ff0000,
    0x00ff00ff00ff01ff,
    0x00ff00ff00ff0101,
    0x00ff00ff0000ff00,
    0x00ff00ff000000ff,
    0x00ff00ff00000000,
    0x00ff00ff00000001,
    0x00ff00ff0001ff00,
    0x00ff00ff0001ff01,
    0x00ff00ff00010000,
    0x00ff00ff000101ff,
    0x00ff00ff00010101,
    0x00ff00ff01ffff00,
    0x00ff00ff01ff0001,
    0x00ff00ff01ff0100,
    0x00ff00ff0100ffff,
    0x00ff00ff0100ff01,
    0x00ff00ff01000000,
    0x00ff00ff0101ffff,
    0x00ff00ff0101ff00,
    0x00ff00ff01010100,
    0x00ff0000ffffff00,
    0x00ff0000ffffff01,
    0x00ff0000ffff0000,
    0x00ff0000ffff0101,
    0x00ff0000ff00ff00,
    0x00ff0000ff0000ff,
    0x00ff0000ff000000,
    0x00ff0000ff000001,
    0x00ff0000ff000100,
    0x00ff0000ff01ffff,
    0x00ff0000ff010000,
    0x00ff0000ff010101,
    0x00ff000000ffff00,
    0x00ff000000ff00ff,
    0x00ff000000ff0000,
    0x00ff000000ff0001,
    0x00ff000000ff0100,
    0x00ff00000000ffff,
    0x00ff00000000ff00,
    0x00ff0000000000ff,
    0x00ff000000000000,
    0x00ff000000000001,
    0x00ff0000000001ff,
    0x00ff000000000100,
    0x00ff00000001ff00,
    0x00ff0000000100ff,
    0x00ff000000010000,
    0x00ff000000010001,
    0x00ff000000010100,
    0x00ff000001ffff01,
    0x00ff000001ff00ff,
    0x00ff000001ff0000,
    0x00ff000001ff01ff,
    0x00ff00000100ff00,
    0x00ff0000010000ff,
    0x00ff000001000000,
    0x00ff000001000001,
    0x00ff000001000100,
    0x00ff000001000101,
    0x00ff000001010000,
    0x00ff0000010101ff,
    0x00ff000001010101,
    0x00ff0001ffffff00,
    0x00ff0001ffff0000,
    0x00ff0001ffff0100,
    0x00ff0001ff0000ff,
    0x00ff0001ff000000,
    0x00ff0001ff0001ff,
    0x00ff0001ff000101,
    0x00ff0001ff01ff00,
    0x00ff0001ff0100ff,
    0x00ff0001ff010100,
    0x00ff000100ffffff,
    0x00ff000100ffff01,
    0x00ff000100ff0000,
    0x00ff000100ff01ff,
    0x00ff00010000ffff,
    0x00ff00010000ff00,
    0x00ff00010000ff01,
    0x00ff000100000000,
    0x00ff000100000001,
    0x00ff000100000100,
    0x00ff00010001ff01,
    0x00ff000100010000,
    0x00ff0001000101ff,
    0x00ff000101ffff00,
    0x00ff000101ff0000,
    0x00ff000101ff0101,
    0x00ff0001010000ff,
    0x00ff000101000000,
    0x00ff00010101ff00,
    0x00ff0001010100ff,
    0x00ff000101010001,
    0x00ff01ffffff0000,
    0x00ff01ffff00ff00,
    0x00ff01ffff000000,
    0x00ff01ffff000101,
    0x00ff01ffff010000,
    0x00ff01ff00ffff01,
    0x00ff01ff00ff0100,
    0x00ff01ff0000ffff,
    0x00ff01ff00000000,
    0x00ff01ff000001ff,
    0x00ff01ff0001ff00,
    0x00ff01ff000100ff,
    0x00ff01ff00010001,
    0x00ff01ff00010100,
    0x00ff01ff01ff0000,
    0x00ff01ff0100ff00,
    0x00ff01ff010000ff,
    0x00ff01ff01000001,
    0x00ff01ff01000100,
    0x00ff01ff01010000,
    0x00ff0100ffffff00,
    0x00ff0100ffff0000,
    0x00ff0100ffff0001,
    0x00ff0100ffff0101,
    0x00ff0100ff00ffff,
    0x00ff0100ff0000ff,
    0x00ff0100ff000000,
    0x00ff0100ff0001ff,
    0x00ff0100ff01ff00,
    0x00ff0100ff0100ff,
    0x00ff0100ff010001,
    0x00ff010000ffffff,
    0x00ff010000ff0000,
    0x00ff010000ff0101,
    0x00ff01000000ff00,
    0x00ff01000000ff01,
    0x00ff0100000000ff,
    0x00ff010000000000,
    0x00ff010000000001,
    0x00ff010000000100,
    0x00ff01000001ffff,
    0x00ff01000001ff01,
    0x00ff010000010000,
    0x00ff010000010001,
    0x00ff010000010101,
    0x00ff010001ff0001,
    0x00ff010001ff0100,
    0x00ff01000100ff01,
    0x00ff010001000000,
    0x00ff010001000001,
    0x00ff0100010001ff,
    0x00ff01000101ff00,
    0x00ff0100010100ff,
    0x00ff010001010001,
    0x00ff010001010100,
    0x00ff0101ff000001,
    0x00ff010100ff00ff,
    0x00ff010100ff0001,
    0x00ff010100ff0100,
    0x00ff010100000000,
    0x00ff0101000001ff,
    0x00ff010100000101,
    0x00ff0101000100ff,
    0x00ff010100010100,
    0x00ff0101010000ff,
    0x00ff010101010000,
    0x0000ffffffffff00,
    0x0000ffffffff00ff,
    0x0000ffffffff0000,
    0x0000ffffffff0001,
    0x0000ffffffff0100,
    0x0000ffffff00ff01,
    0x0000ffffff000000,
    0x0000ffffff000101,
    0x0000ffffff01ff00,
    0x0000ffffff0100ff,
    0x0000ffffff010100,
    0x0000ffff00ffffff,
    0x0000ffff00ff0000,
    0x0000ffff00ff01ff,
    0x0000ffff0000ff00,
    0x0000ffff000000ff,
    0x0000ffff00000000,
    0x0000ffff00000001,
    0x0000ffff00000100,
    0x0000ffff00010000,
    0x0000ffff000101ff,
    0x0000ffff01ff0001,
    0x0000ffff01ff0100,
    0x0000ffff01000000,
    0x0000ffff010001ff,
    0x0000ffff0101ffff,
    0x0000ffff0101ff00,
    0x0000ffff01010001,
    0x0000ffff01010100,
    0x0000ff00ffff0000,
    0x0000ff00ffff01ff,
    0x0000ff00ffff0100,
    0x0000ff00ffff0101,
    0x0000ff00ff00ff00,
    0x0000ff00ff0000ff,
    0x0000ff00ff000000,
    0x0000ff00ff000001,
    0x0000ff00ff0001ff,
    0x0000ff00ff000100,
    0x0000ff00ff01ffff,
    0x0000ff00ff010000,
    0x0000ff00ff010001,
    0x0000ff00ff0101ff,
    0x0000ff00ff010101,
    0x0000ff0000ffff00,
    0x0000ff0000ff00ff,
    0x0000ff0000ff0000,
    0x0000ff0000ff0001,
    0x0000ff0000ff0100,
    0x0000ff000000ffff,
    0x0000ff000000ff00,
    0x0000ff000000ff01,
    0x0000ff00000000ff,
    0x0000ff0000000000,
    0x0000ff0000000001,
    0x0000ff00000001ff,
    0x0000ff0000000100,
    0x0000ff0000000101,
    0x0000ff000001ff00,
    0x0000ff00000100ff,
    0x0000ff0000010000,
    0x0000ff0000010001,
    0x0000ff0000010100,
    0x0000ff0001ffff01,
    0x0000ff0001ff0000,
    0x0000ff000100ff00,
    0x0000ff00010000ff,
    0x0000ff0001000000,
    0x0000ff0001000001,
    0x0000ff0001000100,
    0x0000ff000101ffff,
    0x0000ff0001010000,
    0x0000ff0001010101,
    0x0000ff01ffffff00,
    0x0000ff01ffff0001,
    0x0000ff01ff00ff01,
    0x0000ff01ff000000,
    0x0000ff01ff000101,
    0x0000ff01ff01ff00,
    0x0000ff01ff0100ff,
    0x0000ff0100ffff01,
    0x0000ff0100ff0000,
    0x0000ff0100ff0101,
    0x0000ff010000ff00,
    0x0000ff01000000ff,
    0x0000ff0100000000,
    0x0000ff0100000001,
    0x0000ff0100000100,
    0x0000ff010001ff01,
    0x0000ff0100010000,
    0x0000ff0101ff0000,
    0x0000ff010100ffff,
    0x0000ff010100ff01,
    0x0000ff0101000000,
    0x0000ff0101000100,
    0x0000ff0101000101,
    0x0000ff01010100ff,
    0x000000ffffff00ff,
    0x000000ffffff0000,
    0x000000ffff00ff00,
    0x000000ffff0000ff,
    0x000000ffff000000,
    0x000000ffff000001,
    0x000000ffff0001ff,
    0x000000ffff000100,
    0x000000ffff01ff00,
    0x000000ffff010000,
    0x000000ffff0101ff,
    0x000000ffff010101,
    0x000000ff00ffff00,
    0x000000ff00ff00ff,
    0x000000ff00ff0000,
    0x000000ff00ff0001,
    0x000000ff00ff0100,
    0x000000ff00ff0101,
    0x000000ff0000ffff,
    0x000000ff0000ff00,
    0x000000ff000000ff,
    0x000000ff00000000,
    0x000000ff00000001,
    0x000000ff000001ff,
    0x000000ff00000100,
    0x000000ff00000101,
    0x000000ff0001ff00,
    0x000000ff0001ff01,
    0x000000ff000100ff,
    0x000000ff00010000,
    0x000000ff00010001,
    0x000000ff00010100,
    0x000000ff01ffffff,
    0x000000ff01ff01ff,
    0x000000ff01ff0101,
    0x000000ff0100ff00,
    0x000000ff010000ff,
    0x000000ff01000000,
    0x000000ff01000001,
    0x000000ff01000100,
    0x000000ff0101ff00,
    0x000000ff010100ff,
    0x000000ff01010000,
    0x000000ff01010101,
    0x00000000ffffff00,
    0x00000000ffffff01,
    0x00000000ffff00ff,
    0x00000000ffff0000,
    0x00000000ffff0001,
    0x00000000ffff0100,
    0x00000000ff00ffff,
    0x00000000ff00ff00,
    0x00000000ff00ff01,
    0x00000000ff0000ff,
    0x00000000ff000000,
    0x00000000ff000001,
    0x00000000ff000100,
    0x00000000ff000101,
    0x00000000ff01ff00,
    0x00000000ff0100ff,
    0x00000000ff010000,
    0x00000000ff010001,
    0x00000000ff010100,
    0x0000000000ffffff,
    0x0000000000ffff00,
    0x0000000000ffff01,
    0x0000000000ff00ff,
    0x0000000000ff0000,
    0x0000000000ff0001,
    0x0000000000ff01ff,
    0x0000000000ff0100,
    0x000000000000ffff,
    0x000000000000ff00,
    0x000000000000ff01,
    0x00000000000000ff,
    0x0000000000000000,
    0x0000000000000001,
    0x00000000000001ff,
    0x0000000000000100,
    0x0000000000000101,
    0x000000000001ffff,
    0x000000000001ff00,
    0x00000000000100ff,
    0x0000000000010000,
    0x0000000000010001,
    0x00000000000101ff,
    0x0000000000010100,
    0x0000000000010101,
    0x0000000001ffff00,
    0x0000000001ff00ff,
    0x0000000001ff0000,
    0x0000000001ff0100,
    0x0000000001ff0101,
    0x000000000100ffff,
    0x000000000100ff00,
    0x00000000010000ff,
    0x0000000001000000,
    0x0000000001000001,
    0x00000000010001ff,
    0x0000000001000100,
    0x000000000101ff00,
    0x00000000010100ff,
    0x0000000001010000,
    0x0000000001010001,
    0x0000000001010100,
    0x00000001ffffffff,
    0x00000001ffffff00,
    0x00000001ffffff01,
    0x00000001ffff00ff,
    0x00000001ffff0001,
    0x00000001ffff01ff,
    0x00000001ffff0100,
    0x00000001ff00ff00,
    0x00000001ff0000ff,
    0x00000001ff000000,
    0x00000001ff0001ff,
    0x00000001ff000100,
    0x00000001ff01ffff,
    0x00000001ff01ff00,
    0x00000001ff01ff01,
    0x00000001ff0100ff,
    0x00000001ff010000,
    0x00000001ff010001,
    0x00000001ff0101ff,
    0x00000001ff010100,
    0x0000000100ffff00,
    0x0000000100ff0000,
    0x0000000100ff0001,
    0x0000000100ff01ff,
    0x0000000100ff0100,
    0x0000000100ff0101,
    0x000000010000ffff,
    0x000000010000ff00,
    0x000000010000ff01,
    0x00000001000000ff,
    0x0000000100000000,
    0x0000000100000001,
    0x00000001000001ff,
    0x0000000100000100,
    0x0000000100000101,
    0x000000010001ff00,
    0x00000001000100ff,
    0x0000000100010000,
    0x0000000100010100,
    0x0000000101ffff01,
    0x0000000101ff0000,
    0x0000000101ff0001,
    0x0000000101ff01ff,
    0x0000000101ff0100,
    0x0000000101ff0101,
    0x000000010100ff00,
    0x0000000101000000,
    0x0000000101000101,
    0x000000010101ff01,
    0x0000000101010000,
    0x0000000101010001,
    0x00000001010101ff,
    0x0000000101010100,
    0x000001ffffff00ff,
    0x000001ffffff0000,
    0x000001ffffff0001,
    0x000001ffffff0100,
    0x000001ffff00ffff,
    0x000001ffff000000,
    0x000001ffff0001ff,
    0x000001ffff01ff00,
    0x000001ffff010101,
    0x000001ff00ff0000,
    0x000001ff00ff01ff,
    0x000001ff00ff0101,
    0x000001ff0000ff00,
    0x000001ff000000ff,
    0x000001ff00000000,
    0x000001ff00000001,
    0x000001ff000001ff,
    0x000001ff00000100,
    0x000001ff0001ffff,
    0x000001ff0001ff01,
    0x000001ff000100ff,
    0x000001ff00010000,
    0x000001ff01ffff01,
    0x000001ff01ff0100,
    0x000001ff0100ffff,
    0x000001ff0100ff01,
    0x000001ff01000000,
    0x000001ff010001ff,
    0x000001ff0101ff00,
    0x000001ff01010100,
    0x00000100ffffff00,
    0x00000100ffffff01,
    0x00000100ffff0000,
    0x00000100ffff0101,
    0x00000100ff00ff00,
    0x00000100ff0000ff,
    0x00000100ff000000,
    0x00000100ff000001,
    0x00000100ff000100,
    0x00000100ff010000,
    0x0000010000ffff00,
    0x0000010000ff00ff,
    0x0000010000ff0000,
    0x0000010000ff0001,
    0x0000010000ff0100,
    0x000001000000ffff,
    0x000001000000ff00,
    0x000001000000ff01,
    0x00000100000000ff,
    0x0000010000000000,
    0x0000010000000001,
    0x00000100000001ff,
    0x0000010000000100,
    0x0000010000000101,
    0x000001000001ff00,
    0x00000100000100ff,
    0x0000010000010000,
    0x0000010000010001,
    0x0000010000010100,
    0x0000010001ffff00,
    0x0000010001ff0000,
    0x0000010001ff0100,
    0x000001000100ff00,
    0x00000100010000ff,
    0x0000010001000000,
    0x0000010001000001,
    0x00000100010001ff,
    0x0000010001000100,
    0x0000010001010000,
    0x00000101ffff00ff,
    0x00000101ffff01ff,
    0x00000101ff000000,
    0x00000101ff000101,
    0x00000101ff01ffff,
    0x00000101ff010000,
    0x00000101ff010001,
    0x00000101ff010100,
    0x0000010100ff0000,
    0x0000010100ff01ff,
    0x0000010100ff0100,
    0x000001010000ff00,
    0x0000010100000000,
    0x0000010100000001,
    0x00000101000001ff,
    0x0000010100000100,
    0x000001010001ff01,
    0x0000010100010000,
    0x00000101000101ff,
    0x0000010100010101,
    0x0000010101ffff00,
    0x0000010101ff0101,
    0x000001010100ff01,
    0x0000010101000000,
    0x0000010101000001,
    0x00000101010001ff,
    0x0000010101000101,
    0x000001010101ff00,
    0x0001ffffffff0000,
    0x0001ffffff0000ff,
    0x0001ffffff000001,
    0x0001ffffff000100,
    0x0001ffffff010000,
    0x0001ffff00ff00ff,
    0x0001ffff0000ffff,
    0x0001ffff00000000,
    0x0001ffff00000001,
    0x0001ffff000001ff,
    0x0001ffff00000101,
    0x0001ffff0001ff00,
    0x0001ffff000100ff,
    0x0001ffff00010001,
    0x0001ffff00010100,
    0x0001ffff01ffff00,
    0x0001ffff01000001,
    0x0001ffff01010000,
    0x0001ff00ffffff00,
    0x0001ff00ffff00ff,
    0x0001ff00ffff0001,
    0x0001ff00ffff0100,
    0x0001ff00ff00ff01,
    0x0001ff00ff000000,
    0x0001ff00ff01ff00,
    0x0001ff00ff01ff01,
    0x0001ff00ff010001,
    0x0001ff00ff010100,
    0x0001ff0000ff0000,
    0x0001ff0000ff0100,
    0x0001ff000000ff00,
    0x0001ff0000000000,
    0x0001ff0000000001,
    0x0001ff0000000100,
    0x0001ff0000010000,
    0x0001ff0000010001,
    0x0001ff0000010101,
    0x0001ff0001ff00ff,
    0x0001ff0001ff0101,
    0x0001ff000100ff01,
    0x0001ff0001000000,
    0x0001ff000101ff00,
    0x0001ff0001010001,
    0x0001ff0001010100,
    0x0001ff01ff00ff00,
    0x0001ff01ff000001,
    0x0001ff01ff000100,
    0x0001ff0100ffffff,
    0x0001ff0100ffff00,
    0x0001ff0100ff0001,
    0x0001ff0100000000,
    0x0001ff0100000001,
    0x0001ff01000001ff,
    0x0001ff010001ffff,
    0x0001ff0101ff0000,
    0x0001ff010100ff00,
    0x0001ff0101000001,
    0x0001ff0101010000,
    0x000100ffff00ff00,
    0x000100ffff00ff01,
    0x000100ffff000000,
    0x000100ffff000001,
    0x000100ffff000101,
    0x000100ffff01ff00,
    0x000100ffff010001,
    0x000100ffff010100,
    0x000100ff00ffffff,
    0x000100ff00ffff01,
    0x000100ff00ff0000,
    0x000100ff00ff01ff,
    0x000100ff00ff0101,
    0x000100ff0000ff00,
    0x000100ff000000ff,
    0x000100ff00000000,
    0x000100ff00000001,
    0x000100ff00000100,
    0x000100ff00000101,
    0x000100ff0001ffff,
    0x000100ff0001ff01,
    0x000100ff00010000,
    0x000100ff01ff00ff,
    0x000100ff01ff0000,
    0x000100ff01ff0100,
    0x000100ff0100ffff,
    0x000100ff0100ff01,
    0x000100ff010000ff,
    0x000100ff01000000,
    0x000100ff01000001,
    0x000100ff010001ff,
    0x000100ff01000101,
    0x000100ff0101ff00,
    0x000100ff010100ff,
    0x000100ff01010100,
    0x00010000ffff0000,
    0x00010000ffff01ff,
    0x00010000ffff0101,
    0x00010000ff00ff00,
    0x00010000ff000000,
    0x00010000ff000001,
    0x00010000ff000100,
    0x0001000000ff00ff,
    0x0001000000ff0000,
    0x0001000000ff0001,
    0x0001000000ff0100,
    0x000100000000ffff,
    0x000100000000ff00,
    0x00010000000000ff,
    0x0001000000000000,
    0x0001000000000001,
    0x0001000000000100,
    0x000100000001ff00,
    0x00010000000100ff,
    0x0001000000010000,
    0x0001000000010001,
    0x0001000000010100,
    0x0001000001ff0001,
    0x0001000001ff0100,
    0x0001000001ff0101,
    0x000100000100ff00,
    0x0001000001000000,
    0x0001000001000001,
    0x0001000001000100,
    0x0001000001000101,
    0x000100000101ff01,
    0x0001000001010000,
    0x0001000001010001,
    0x00010000010101ff,
    0x00010001ffffff01,
    0x00010001ffff0100,
    0x00010001ff000000,
    0x00010001ff01ffff,
    0x00010001ff010001,
    0x00010001ff0101ff,
    0x00010001ff010100,
    0x0001000100ffffff,
    0x0001000100ff0000,
    0x0001000100ff01ff,
    0x0001000100ff0101,
    0x000100010000ff00,
    0x00010001000000ff,
    0x0001000100000000,
    0x0001000100000001,
    0x00010001000001ff,
    0x0001000100000101,
    0x000100010001ffff,
    0x0001000100010000,
    0x00010001000101ff,
    0x0001000101ffffff,
    0x0001000101ffff01,
    0x0001000101ff0000,
    0x0001000101ff0101,
    0x00010001010000ff,
    0x0001000101000001,
    0x00010001010001ff,
    0x0001000101000100,
    0x000100010101ffff,
    0x00010001010100ff,
    0x0001000101010001,
    0x0001000101010101,
    0x000101ffff000001,
    0x000101ffff000100,
    0x000101ffff010000,
    0x000101ff00ffff00,
    0x000101ff0000ff01,
    0x000101ff00000000,
    0x000101ff00000101,
    0x000101ff0001ff00,
    0x000101ff00010100,
    0x000101ff01ff0000,
    0x000101ff0100ff00,
    0x000101ff010001ff,
    0x000101ff01010001,
    0x00010100ffffff00,
    0x00010100ffff00ff,
    0x00010100ff00ffff,
    0x00010100ff000000,
    0x00010100ff01ff00,
    0x00010100ff0100ff,
    0x00010100ff010001,
    0x00010100ff010100,
    0x0001010000ffffff,
    0x0001010000ffff00,
    0x0001010000ff0000,
    0x0001010000ff0001,
    0x0001010000ff01ff,
    0x000101000000ff00,
    0x00010100000000ff,
    0x0001010000000000,
    0x0001010000000001,
    0x0001010000000100,
    0x000101000001ffff,
    0x0001010000010000,
    0x0001010000010101,
    0x0001010001ffff01,
    0x0001010001ff00ff,
    0x0001010001ff0101,
    0x0001010001000000,
    0x000101000101ff00,
    0x00010100010100ff,
    0x0001010001010000,
    0x0001010001010100,
    0x00010101ff00ff00,
    0x00010101ff000001,
    0x00010101ff0001ff,
    0x0001010100ffff00,
    0x0001010100ff00ff,
    0x0001010100ff0100,
    0x000101010000ffff,
    0x0001010100000000,
    0x00010101000001ff,
    0x0001010100000101,
    0x00010101000100ff,
    0x0001010100010000,
    0x0001010100010100,
    0x0001010101ff0001,
    0x00010101010000ff,
    0x00010101010001ff,
    0x0001010101000101,
    0x0001010101010001,
    0x01ffffffffffffff,
    0x01ffffffffffff01,
    0x01ffffffffff01ff,
    0x01ffffffffff0101,
    0x01ffffffff01ffff,
    0x01ffffffff01ff01,
    0x01ffffffff0101ff,
    0x01ffffffff010101,
    0x01ffffff00ff0000,
    0x01ffffff0000ffff,
    0x01ffffff0000ff00,
    0x01ffffff000000ff,
    0x01ffffff00000001,
    0x01ffffff00000100,
    0x01ffffff00010000,
    0x01ffffff01ffffff,
    0x01ffffff01ffff01,
    0x01ffffff01ff01ff,
    0x01ffffff01ff0101,
    0x01ffffff01000000,
    0x01ffffff0101ffff,
    0x01ffffff0101ff01,
    0x01ffffff010101ff,
    0x01ffffff01010101,
    0x01ffff00ffff0000,
    0x01ffff00ff00ff00,
    0x01ffff00ff0000ff,
    0x01ffff00ff000001,
    0x01ffff00ff000100,
    0x01ffff00ff010000,
    0x01ffff0000ffff00,
    0x01ffff0000ff00ff,
    0x01ffff0000ff0100,
    0x01ffff000000ffff,
    0x01ffff000000ff01,
    0x01ffff0000000000,
    0x01ffff0000000001,
    0x01ffff00000001ff,
    0x01ffff0000000100,
    0x01ffff00000100ff,
    0x01ffff0000010001,
    0x01ffff0000010100,
    0x01ffff0001ff0000,
    0x01ffff0001ff0100,
    0x01ffff00010000ff,
    0x01ffff0001000001,
    0x01ffff0001000100,
    0x01ffff0001010000,
    0x01ffff01ffffffff,
    0x01ffff01ffffff01,
    0x01ffff01ffff01ff,
    0x01ffff01ffff0101,
    0x01ffff01ff000000,
    0x01ffff01ff01ffff,
    0x01ffff01ff01ff01,
    0x01ffff01ff0101ff,
    0x01ffff01ff010101,
    0x01ffff010000ff00,
    0x01ffff01000000ff,
    0x01ffff0100000100,
    0x01ffff0100010000,
    0x01ffff0101ffffff,
    0x01ffff0101ffff01,
    0x01ffff0101ff01ff,
    0x01ffff0101ff0101,
    0x01ffff0101000000,
    0x01ffff010101ffff,
    0x01ffff010101ff01,
    0x01ffff01010101ff,
    0x01ffff0101010101,
    0x01ff00ffff0000ff,
    0x01ff00ffff000100,
    0x01ff00ff00ffff00,
    0x01ff00ff00ff00ff,
    0x01ff00ff0000ff00,
    0x01ff00ff00000000,
    0x01ff00ff00000101,
    0x01ff00ff0001ff00,
    0x01ff00ff000100ff,
    0x01ff00ff00010100,
    0x01ff00ff010000ff,
    0x01ff00ff01000100,
    0x01ff0000ffffff00,
    0x01ff0000ffff0100,
    0x01ff0000ff00ff01,
    0x01ff0000ff000000,
    0x01ff0000ff000101,
    0x01ff0000ff010001,
    0x01ff0000ff010100,
    0x01ff000000ffffff,
    0x01ff000000ffff00,
    0x01ff000000ff0000,
    0x01ff000000ff01ff,
    0x01ff00000000ff00,
    0x01ff0000000000ff,
    0x01ff000000000000,
    0x01ff000000000001,
    0x01ff000000000100,
    0x01ff000000000101,
    0x01ff000000010000,
    0x01ff000000010001,
    0x01ff0000000101ff,
    0x01ff000000010101,
    0x01ff000001ffff00,
    0x01ff000001ff00ff,
    0x01ff000001ff0001,
    0x01ff000001ff0100,
    0x01ff00000100ffff,
    0x01ff00000100ff01,
    0x01ff000001000000,
    0x01ff0000010001ff,
    0x01ff000001010001,
    0x01ff0001ff00ff00,
    0x01ff0001ff000001,
    0x01ff0001ff000100,
    0x01ff0001ff010000,
    0x01ff000100ffff00,
    0x01ff000100ff00ff,
    0x01ff000100ff0100,
    0x01ff000100ff0101,
    0x01ff00010000ffff,
    0x01ff000100000000,
    0x01ff000100000100,
    0x01ff000100000101,
    0x01ff00010001ff00,
    0x01ff000100010001,
    0x01ff000100010101,
    0x01ff000101ff0000,
    0x01ff00010100ff00,
    0x01ff000101000101,
    0x01ff0001010100ff,
    0x01ff01ffffffffff,
    0x01ff01ffffffff01,
    0x01ff01ffffff01ff,
    0x01ff01ffffff0101,
    0x01ff01ffff000000,
    0x01ff01ffff01ffff,
    0x01ff01ffff01ff01,
    0x01ff01ffff0101ff,
    0x01ff01ffff010101,
    0x01ff01ff00ffff00,
    0x01ff01ff00ff0000,
    0x01ff01ff0000ff00,
    0x01ff01ff000000ff,
    0x01ff01ff00000100,
    0x01ff01ff00010000,
    0x01ff01ff00010100,
    0x01ff01ff01ffffff,
    0x01ff01ff01ffff01,
    0x01ff01ff01ff01ff,
    0x01ff01ff01ff0101,
    0x01ff01ff01000000,
    0x01ff01ff0101ffff,
    0x01ff01ff0101ff01,
    0x01ff01ff010101ff,
    0x01ff01ff01010101,
    0x01ff0100ffff0000,
    0x01ff0100ffff0001,
    0x01ff0100ff00ff00,
    0x01ff0100ff0000ff,
    0x01ff0100ff000001,
    0x01ff0100ff010000,
    0x01ff010000ffff00,
    0x01ff010000ff00ff,
    0x01ff010000ff0001,
    0x01ff010000ff0100,
    0x01ff01000000ffff,
    0x01ff01000000ff01,
    0x01ff010000000000,
    0x01ff010000000101,
    0x01ff01000001ff00,
    0x01ff0100000100ff,
    0x01ff010001ff0000,
    0x01ff010001000001,
    0x01ff010001000100,
    0x01ff010001010000,
    0x01ff0101ffffffff,
    0x01ff0101ffffff01,
    0x01ff0101ffff01ff,
    0x01ff0101ffff0101,
    0x01ff0101ff000000,
    0x01ff0101ff01ffff,
    0x01ff0101ff01ff01,
    0x01ff0101ff0101ff,
    0x01ff0101ff010101,
    0x01ff010100ff0000,
    0x01ff01010000ff00,
    0x01ff0101000000ff,
    0x01ff010100000001,
    0x01ff010101ffffff,
    0x01ff010101ffff01,
    0x01ff010101ff01ff,
    0x01ff010101ff0101,
    0x01ff010101000000,
    0x01ff01010101ffff,
    0x01ff01010101ff01,
    0x01ff0101010101ff,
    0x01ff010101010101,
    0x0100ffffffff0000,
    0x0100ffffff00ff00,
    0x0100ffffff000001,
    0x0100ffffff0001ff,
    0x0100ffffff000100,
    0x0100ffffff010000,
    0x0100ffff00ffff00,
    0x0100ffff00ff0001,
    0x0100ffff00ff0100,
    0x0100ffff00000000,
    0x0100ffff000001ff,
    0x0100ffff00000101,
    0x0100ffff00010100,
    0x0100ffff00010101,
    0x0100ffff01ff0000,
    0x0100ffff0100ff00,
    0x0100ffff010000ff,
    0x0100ffff01000001,
    0x0100ffff01000100,
    0x0100ffff01010000,
    0x0100ff00ffffff00,
    0x0100ff00ffff00ff,
    0x0100ff00ffff0001,
    0x0100ff00ffff0100,
    0x0100ff00ff00ffff,
    0x0100ff00ff000000,
    0x0100ff00ff0001ff,
    0x0100ff00ff000101,
    0x0100ff00ff01ff00,
    0x0100ff00ff0100ff,
    0x0100ff00ff010001,
    0x0100ff00ff010100,
    0x0100ff0000ffffff,
    0x0100ff0000ff0000,
    0x0100ff000000ffff,
    0x0100ff000000ff00,
    0x0100ff00000000ff,
    0x0100ff0000000000,
    0x0100ff0000000001,
    0x0100ff0000000100,
    0x0100ff000001ff01,
    0x0100ff0000010000,
    0x0100ff0001ff00ff,
    0x0100ff0001ff0001,
    0x0100ff000100ff01,
    0x0100ff0001000000,
    0x0100ff00010001ff,
    0x0100ff000101ff00,
    0x0100ff00010100ff,
    0x0100ff0001010001,
    0x0100ff0001010100,
    0x0100ff01ffff0000,
    0x0100ff01ff00ff00,
    0x0100ff01ff0000ff,
    0x0100ff01ff000100,
    0x0100ff01ff010000,
    0x0100ff0100ff00ff,
    0x0100ff0100ff0001,
    0x0100ff0100ff0100,
    0x0100ff010000ffff,
    0x0100ff010000ff01,
    0x0100ff0100000000,
    0x0100ff01000001ff,
    0x0100ff0100010001,
    0x0100ff0100010100,
    0x0100ff0101ff0000,
    0x0100ff01010000ff,
    0x0100ff0101000001,
    0x0100ff0101010100,
    0x010000ffffffff00,
    0x010000ffffff00ff,
    0x010000ffffff0001,
    0x010000ffff00ffff,
    0x010000ffff000000,
    0x010000ffff0001ff,
    0x010000ffff010001,
    0x010000ff00ffffff,
    0x010000ff00ff0101,
    0x010000ff0000ff00,
    0x010000ff000000ff,
    0x010000ff00000000,
    0x010000ff00000001,
    0x010000ff000001ff,
    0x010000ff00000100,
    0x010000ff0001ffff,
    0x010000ff0001ff00,
    0x010000ff0001ff01,
    0x010000ff00010000,
    0x010000ff01ff00ff,
    0x010000ff01ff0001,
    0x010000ff0100ff01,
    0x010000ff010000ff,
    0x010000ff01000000,
    0x010000ff010001ff,
    0x010000ff0101ff00,
    0x010000ff01010100,
    0x01000000ffffffff,
    0x01000000ffff0000,
    0x01000000ffff01ff,
    0x01000000ffff0101,
    0x01000000ff00ffff,
    0x01000000ff00ff00,
    0x01000000ff0000ff,
    0x01000000ff000000,
    0x01000000ff000001,
    0x01000000ff000100,
    0x01000000ff01ff00,
    0x01000000ff010000,
    0x01000000ff010100,
    0x01000000ff010101,
    0x0100000000ffff00,
    0x0100000000ff00ff,
    0x0100000000ff0000,
    0x0100000000ff0001,
    0x0100000000ff0100,
    0x010000000000ffff,
    0x010000000000ff00,
    0x010000000000ff01,
    0x01000000000000ff,
    0x0100000000000000,
    0x0100000000000001,
    0x01000000000001ff,
    0x0100000000000100,
    0x0100000000000101,
    0x010000000001ff00,
    0x01000000000100ff,
    0x0100000000010000,
    0x0100000000010001,
    0x0100000000010100,
    0x0100000001ffff00,
    0x0100000001ff0000,
    0x0100000001ff01ff,
    0x010000000100ff00,
    0x010000000100ff01,
    0x01000000010000ff,
    0x0100000001000000,
    0x0100000001000001,
    0x0100000001000100,
    0x0100000001000101,
    0x010000000101ffff,
    0x010000000101ff01,
    0x0100000001010000,
    0x01000000010101ff,
    0x0100000001010101,
    0x01000001ffffff00,
    0x01000001ffff00ff,
    0x01000001ff00ffff,
    0x01000001ff000000,
    0x01000001ff000100,
    0x01000001ff01ffff,
    0x01000001ff010001,
    0x01000001ff010100,
    0x0100000100ff0000,
    0x0100000100ff01ff,
    0x0100000100ff0100,
    0x010000010000ff00,
    0x010000010000ff01,
    0x0100000100000000,
    0x0100000100000001,
    0x0100000100000100,
    0x0100000100010000,
    0x01000001000101ff,
    0x0100000101ffff01,
    0x0100000101ff00ff,
    0x0100000101ff0100,
    0x0100000101ff0101,
    0x010000010100ff01,
    0x01000001010000ff,
    0x0100000101000000,
    0x01000001010100ff,
    0x0100000101010001,
    0x0100000101010100,
    0x010001ffffff0000,
    0x010001ffff000001,
    0x010001ffff000100,
    0x010001ffff010000,
    0x010001ff00ffff00,
    0x010001ff00ff0001,
    0x010001ff0000ffff,
    0x010001ff0000ff01,
    0x010001ff00000000,
    0x010001ff00000001,
    0x010001ff00000101,
    0x010001ff000100ff,
    0x010001ff00010000,
    0x010001ff01ff0000,
    0x010001ff0100ff00,
    0x010001ff01000001,
    0x010001ff01000100,
    0x010001ff01010000,
    0x01000100ffff00ff,
    0x01000100ffff0001,
    0x01000100ffff0100,
    0x01000100ff00ffff,
    0x01000100ff00ff01,
    0x01000100ff000000,
    0x01000100ff0001ff,
    0x01000100ff000101,
    0x01000100ff01ffff,
    0x01000100ff01ff00,
    0x01000100ff0100ff,
    0x01000100ff010001,
    0x0100010000ffffff,
    0x0100010000ffff01,
    0x0100010000ff0000,
    0x0100010000ff01ff,
    0x0100010000ff0101,
    0x010001000000ff00,
    0x01000100000000ff,
    0x0100010000000000,
    0x0100010000000001,
    0x0100010000000100,
    0x010001000001ff01,
    0x0100010000010000,
    0x0100010000010001,
    0x0100010000010101,
    0x0100010001ffff00,
    0x0100010001ff00ff,
    0x010001000100ffff,
    0x010001000100ff01,
    0x0100010001000000,
    0x0100010001000101,
    0x010001000101ff00,
    0x0100010001010001,
    0x01000101ffff0000,
    0x01000101ff000000,
    0x01000101ff010000,
    0x0100010100ff00ff,
    0x0100010100ff0001,
    0x0100010100ff0100,
    0x010001010000ffff,
    0x0100010100000000,
    0x01000101000001ff,
    0x010001010001ff00,
    0x0100010101ff0000,
    0x010001010100ff00,
    0x01000101010000ff,
    0x0100010101000000,
    0x0100010101000001,
    0x0101ffffffffffff,
    0x0101ffffffffff01,
    0x0101ffffffff01ff,
    0x0101ffffffff0101,
    0x0101ffffff000000,
    0x0101ffffff01ffff,
    0x0101ffffff01ff01,
    0x0101ffffff0101ff,
    0x0101ffffff010101,
    0x0101ffff00ff0000,
    0x0101ffff0000ff00,
    0x0101ffff000000ff,
    0x0101ffff00000001,
    0x0101ffff00000100,
    0x0101ffff01ffffff,
    0x0101ffff01ffff01,
    0x0101ffff01ff01ff,
    0x0101ffff01ff0101,
    0x0101ffff01000000,
    0x0101ffff0101ffff,
    0x0101ffff0101ff01,
    0x0101ffff010101ff,
    0x0101ffff01010101,
    0x0101ff00ffff0000,
    0x0101ff00ffff0100,
    0x0101ff00ff00ff00,
    0x0101ff00ff0000ff,
    0x0101ff00ff000001,
    0x0101ff00ff000100,
    0x0101ff00ff000101,
    0x0101ff0000ff0001,
    0x0101ff0000ff0100,
    0x0101ff000000ff00,
    0x0101ff0000000000,
    0x0101ff00000001ff,
    0x0101ff0000000101,
    0x0101ff000001ff00,
    0x0101ff00000100ff,
    0x0101ff0001ff0000,
    0x0101ff000100ffff,
    0x0101ff000100ff01,
    0x0101ff0001000001,
    0x0101ff0001000100,
    0x0101ff01ffffff01,
    0x0101ff01ffff01ff,
    0x0101ff01ffff0101,
    0x0101ff01ff00ffff,
    0x0101ff01ff000100,
    0x0101ff01ff01ff01,
    0x0101ff01ff0101ff,
    0x0101ff01ff010101,
    0x0101ff0100ff0000,
    0x0101ff010000ff00,
    0x0101ff0100000001,
    0x0101ff0100000100,
    0x0101ff0100010000,
    0x0101ff0101ffffff,
    0x0101ff0101ffff01,
    0x0101ff0101ff01ff,
    0x0101ff0101ff0101,
    0x0101ff0101000000,
    0x0101ff010101ffff,
    0x0101ff010101ff01,
    0x0101ff01010101ff,
    0x0101ff0101010101,
    0x010100ffff000100,
    0x010100ffff010000,
    0x010100ff00ffff00,
    0x010100ff00ff00ff,
    0x010100ff0000ffff,
    0x010100ff000000ff,
    0x010100ff00000000,
    0x010100ff000001ff,
    0x010100ff00000101,
    0x010100ff0001ff00,
    0x010100ff00010000,
    0x010100ff00010001,
    0x010100ff000101ff,
    0x010100ff00010100,
    0x010100ff01ff0000,
    0x01010000ffff0001,
    0x01010000ffff0100,
    0x01010000ff00ffff,
    0x01010000ff00ff01,
    0x01010000ff000000,
    0x01010000ff0001ff,
    0x01010000ff010001,
    0x01010000ff010100,
    0x0101000000ffff01,
    0x0101000000ff0000,
    0x010100000000ff00,
    0x01010000000000ff,
    0x0101000000000000,
    0x0101000000000001,
    0x0101000000000100,
    0x0101000000010000,
    0x0101000000010101,
    0x0101000001ffff00,
    0x0101000001ff00ff,
    0x0101000001ff0000,
    0x0101000001ff0001,
    0x0101000001ff0100,
    0x010100000100ff01,
    0x0101000001000000,
    0x01010000010001ff,
    0x01010001ffff0000,
    0x01010001ff00ff00,
    0x01010001ff000001,
    0x01010001ff000101,
    0x01010001ff01ff00,
    0x01010001ff010000,
    0x0101000100ff00ff,
    0x0101000100ff0001,
    0x0101000100ff0101,
    0x010100010000ff01,
    0x0101000100000000,
    0x0101000100000001,
    0x01010001000001ff,
    0x010100010001ffff,
    0x010100010001ff01,
    0x0101000101ff0001,
    0x010100010100ffff,
    0x0101000101000000,
    0x0101000101000001,
    0x0101000101000100,
    0x010100010101ff00,
    0x01010001010100ff,
    0x0101000101010001,
    0x010101ffffffffff,
    0x010101ffffffff01,
    0x010101ffffff01ff,
    0x010101ffffff0101,
    0x010101ffff01ffff,
    0x010101ffff01ff01,
    0x010101ffff0101ff,
    0x010101ffff010101,
    0x010101ff0000ff00,
    0x010101ff000000ff,
    0x010101ff00000001,
    0x010101ff00000100,
    0x010101ff01ffffff,
    0x010101ff01ffff01,
    0x010101ff01ff01ff,
    0x010101ff01ff0101,
    0x010101ff01000000,
    0x010101ff0101ffff,
    0x010101ff0101ff01,
    0x010101ff010101ff,
    0x010101ff01010101,
    0x01010100ffff0000,
    0x01010100ff0000ff,
    0x01010100ff000100,
    0x01010100ff01ff00,
    0x01010100ff010000,
    0x0101010000ffff00,
    0x010101000000ffff,
    0x0101010000000000,
    0x0101010000000101,
    0x010101000001ff00,
    0x0101010000010001,
    0x0101010000010100,
    0x010101000100ffff,
    0x0101010001000001,
    0x01010101ffffffff,
    0x01010101ffffff01,
    0x01010101ffff01ff,
    0x01010101ffff0101,
    0x01010101ff01ffff,
    0x01010101ff01ff01,
    0x01010101ff0101ff,
    0x01010101ff010101,
    0x010101010000ff00,
    0x01010101000000ff,
    0x0101010100000001,
    0x0101010101ffffff,
    0x0101010101ffff01,
    0x0101010101ff01ff,
    0x0101010101ff0101,
    0x0101010101000000,
    0x010101010101ffff,
    0x010101010101ff01,
    0x01010101010101ff,
    0x0101010101010101,
];

// The IQ2_XXS codebook is the fixed 256-entry lattice used by GGML. Each
// entry packs eight unsigned 8-bit magnitudes in little-endian order.
#[allow(clippy::unreadable_literal)]
const IQ2_XXS_GRID: [u64; 256] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b2b08,
    0x08080808082b2b2b,
    0x0808080819080819,
    0x0808080819081908,
    0x0808080819190808,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b082b2b,
    0x080808082b2b082b,
    0x0808081908080819,
    0x0808081908081908,
    0x0808081908190808,
    0x0808081908191919,
    0x0808081919080808,
    0x080808192b081908,
    0x080808192b192b08,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b082b082b,
    0x0808082b2b08082b,
    0x0808190808080819,
    0x0808190808081908,
    0x0808190808190808,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819082b08,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908082b08,
    0x08081919082b0808,
    0x080819191908192b,
    0x08081919192b2b19,
    0x080819192b080808,
    0x080819192b190819,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b19080808,
    0x0808192b2b081908,
    0x0808192b2b2b1908,
    0x08082b0808080808,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808191908,
    0x08082b08082b2b08,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b082b082b08,
    0x08082b1908081908,
    0x08082b1919080808,
    0x08082b2b0808082b,
    0x08082b2b08191908,
    0x0819080808080819,
    0x0819080808081908,
    0x0819080808190808,
    0x08190808082b0819,
    0x0819080819080808,
    0x08190808192b0808,
    0x081908082b081908,
    0x081908082b190808,
    0x081908082b191919,
    0x0819081908080808,
    0x0819081908082b08,
    0x08190819082b0808,
    0x0819081919190808,
    0x0819081919192b2b,
    0x081908192b080808,
    0x0819082b082b1908,
    0x0819082b19081919,
    0x0819190808080808,
    0x0819190808082b08,
    0x08191908082b0808,
    0x08191908082b1919,
    0x0819190819082b19,
    0x081919082b080808,
    0x0819191908192b08,
    0x08191919192b082b,
    0x0819192b08080808,
    0x0819192b0819192b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b0819080808,
    0x08192b082b080819,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b192b2b0808,
    0x08192b2b19190819,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808082b2b,
    0x082b080819081908,
    0x082b0808192b0819,
    0x082b08082b080808,
    0x082b08082b08082b,
    0x082b0819082b2b19,
    0x082b081919082b08,
    0x082b082b08080808,
    0x082b082b0808082b,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b190819080808,
    0x082b19081919192b,
    0x082b191908080808,
    0x082b191919080819,
    0x082b1919192b1908,
    0x082b192b2b190808,
    0x082b2b0808082b08,
    0x082b2b08082b0808,
    0x082b2b082b191908,
    0x082b2b2b19081908,
    0x1908080808080819,
    0x1908080808081908,
    0x1908080808190808,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x1908080819082b08,
    0x190808081919192b,
    0x19080808192b0808,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x19080819082b0808,
    0x19080819192b0819,
    0x190808192b080808,
    0x190808192b081919,
    0x1908082b08080819,
    0x1908082b08190808,
    0x1908082b19082b08,
    0x1908082b1919192b,
    0x1908082b192b2b08,
    0x1908190808080808,
    0x1908190808082b08,
    0x19081908082b0808,
    0x190819082b080808,
    0x190819082b192b19,
    0x190819190819082b,
    0x19081919082b1908,
    0x1908192b08080808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b1908080808,
    0x19082b1919192b08,
    0x19082b19192b0819,
    0x19082b192b08082b,
    0x19082b2b19081919,
    0x19082b2b2b190808,
    0x1919080808080808,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808192b19,
    0x19190808082b0808,
    0x191908082b080808,
    0x191908082b082b08,
    0x1919081908081908,
    0x191908191908082b,
    0x191908192b2b1908,
    0x1919082b2b190819,
    0x191919082b190808,
    0x191919082b19082b,
    0x1919191908082b2b,
    0x1919192b08080819,
    0x1919192b19191908,
    0x19192b0808080808,
    0x19192b0808190819,
    0x19192b0808192b19,
    0x19192b08192b1908,
    0x19192b1919080808,
    0x19192b2b08082b08,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b0808192b2b08,
    0x192b081908080808,
    0x192b081919191919,
    0x192b082b08192b08,
    0x192b082b192b0808,
    0x192b190808080808,
    0x192b190808081919,
    0x192b191908190808,
    0x192b19190819082b,
    0x192b19192b081908,
    0x192b2b081908082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808082b2b,
    0x2b08080819080819,
    0x2b0808082b08082b,
    0x2b08081908081908,
    0x2b08081908192b08,
    0x2b08081919080808,
    0x2b08082b08190819,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b08191908080808,
    0x2b0819191908192b,
    0x2b0819192b191908,
    0x2b08192b08082b19,
    0x2b08192b19080808,
    0x2b08192b192b0808,
    0x2b082b080808082b,
    0x2b082b1908081908,
    0x2b082b2b08190819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908082b2b0819,
    0x2b1908190819192b,
    0x2b1908192b080808,
    0x2b19082b19081919,
    0x2b19190808080808,
    0x2b191908082b082b,
    0x2b19190819081908,
    0x2b19191919190819,
    0x2b192b082b080819,
    0x2b192b19082b0808,
    0x2b2b08080808082b,
    0x2b2b080819190808,
    0x2b2b08082b081919,
    0x2b2b081908082b19,
    0x2b2b082b08080808,
    0x2b2b190808192b08,
    0x2b2b2b0819190808,
    0x2b2b2b1908081908,
];

#[allow(clippy::unreadable_literal)]
const IQ2_XS_GRID: [u64; 512] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b192b19,
    0x080808082b2b0808,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x0808081908192b2b,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x08080819192b0808,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b082b2b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x08081908192b0808,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b190808,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908190819,
    0x0808191908191908,
    0x08081919082b0808,
    0x0808191919080819,
    0x0808191919081908,
    0x0808191919190808,
    0x08081919192b0819,
    0x080819192b080808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b08190808,
    0x0808192b082b192b,
    0x0808192b19080808,
    0x0808192b1908082b,
    0x0808192b2b081908,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808082b2b,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b0819190808,
    0x08082b0819192b08,
    0x08082b082b080808,
    0x08082b082b2b0808,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b1919080808,
    0x08082b192b080819,
    0x08082b192b082b19,
    0x08082b2b08080808,
    0x08082b2b082b0808,
    0x08082b2b082b2b08,
    0x08082b2b2b19192b,
    0x08082b2b2b2b0808,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x08190808192b0808,
    0x08190808192b2b2b,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b190808,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908190819,
    0x0819081908191908,
    0x08190819082b0808,
    0x0819081919080819,
    0x0819081919081908,
    0x0819081919190808,
    0x081908192b080808,
    0x081908192b191908,
    0x081908192b19192b,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b0808192b,
    0x0819082b08190808,
    0x0819082b19080808,
    0x0819082b192b0808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x08191908082b0808,
    0x0819190819080819,
    0x0819190819081908,
    0x0819190819082b19,
    0x0819190819190808,
    0x08191908192b1908,
    0x081919082b080808,
    0x0819191908080819,
    0x0819191908081908,
    0x0819191908190808,
    0x0819191919080808,
    0x0819192b08080808,
    0x0819192b08191908,
    0x0819192b19082b19,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b0808190808,
    0x08192b080819082b,
    0x08192b0819080808,
    0x08192b0819191908,
    0x08192b082b08192b,
    0x08192b1908080808,
    0x08192b1908081919,
    0x08192b19192b192b,
    0x08192b2b19190819,
    0x08192b2b2b2b2b19,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808082b2b,
    0x082b080808190819,
    0x082b080808191908,
    0x082b0808082b0808,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08082b080808,
    0x082b08082b2b0808,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b081919080808,
    0x082b081919082b08,
    0x082b0819192b1919,
    0x082b082b08080808,
    0x082b082b082b082b,
    0x082b082b2b080808,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b190808190808,
    0x082b1908082b2b19,
    0x082b190819080808,
    0x082b191908080808,
    0x082b191919080819,
    0x082b19191919082b,
    0x082b19192b192b19,
    0x082b192b08080819,
    0x082b192b08192b2b,
    0x082b192b2b2b192b,
    0x082b2b0808080808,
    0x082b2b0808082b08,
    0x082b2b0808082b2b,
    0x082b2b08082b0808,
    0x082b2b0819191919,
    0x082b2b082b082b08,
    0x082b2b082b2b082b,
    0x082b2b19192b2b08,
    0x082b2b192b190808,
    0x082b2b2b08082b08,
    0x082b2b2b082b0808,
    0x082b2b2b2b08082b,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x19080808082b0819,
    0x19080808082b1908,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x19080808192b0808,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x19080819082b0808,
    0x1908081919080819,
    0x1908081919081908,
    0x1908081919190808,
    0x190808192b080808,
    0x190808192b081919,
    0x190808192b2b082b,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b082b2b19,
    0x1908082b19080808,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808190819,
    0x1908190808191908,
    0x1908190808192b19,
    0x19081908082b0808,
    0x1908190819080819,
    0x1908190819081908,
    0x1908190819190808,
    0x190819082b080808,
    0x190819082b191908,
    0x1908191908080819,
    0x1908191908081908,
    0x1908191908190808,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819192b192b2b,
    0x1908192b08080808,
    0x1908192b08082b2b,
    0x1908192b19081908,
    0x1908192b19190808,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808190808,
    0x19082b0819080808,
    0x19082b0819081919,
    0x19082b0819191908,
    0x19082b08192b082b,
    0x19082b1908080808,
    0x19082b1908190819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b19192b2b19,
    0x19082b2b08081908,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x19190808082b0808,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x1919080819190808,
    0x191908082b080808,
    0x1919081908080819,
    0x1919081908081908,
    0x1919081908190808,
    0x1919081908191919,
    0x1919081919080808,
    0x191908191908082b,
    0x1919082b08080808,
    0x1919082b19081908,
    0x1919082b2b2b2b2b,
    0x1919190808080819,
    0x1919190808081908,
    0x1919190808190808,
    0x19191908082b0819,
    0x1919190819080808,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b2b0819,
    0x1919191908080808,
    0x1919191908082b08,
    0x191919192b080808,
    0x191919192b082b08,
    0x1919192b082b0819,
    0x1919192b192b2b08,
    0x1919192b2b2b0819,
    0x19192b0808080808,
    0x19192b0808191908,
    0x19192b0819080819,
    0x19192b0819190808,
    0x19192b082b192b19,
    0x19192b1908192b2b,
    0x19192b1919080808,
    0x19192b191908082b,
    0x19192b2b2b081919,
    0x192b080808080819,
    0x192b080808081908,
    0x192b080808190808,
    0x192b080819080808,
    0x192b080819191908,
    0x192b0808192b082b,
    0x192b08082b08192b,
    0x192b08082b2b2b19,
    0x192b081908080808,
    0x192b082b082b1908,
    0x192b082b19082b2b,
    0x192b082b2b19082b,
    0x192b190808080808,
    0x192b19080819192b,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919081919,
    0x192b19192b2b1908,
    0x192b2b0808080819,
    0x192b2b08192b2b2b,
    0x192b2b19082b1919,
    0x192b2b2b0808192b,
    0x192b2b2b19191908,
    0x192b2b2b192b082b,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b080808082b0808,
    0x2b080808082b2b2b,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808082b080808,
    0x2b0808082b08082b,
    0x2b0808082b2b2b08,
    0x2b0808082b2b2b2b,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b0808190808192b,
    0x2b08081908190808,
    0x2b08081919080808,
    0x2b08081919190819,
    0x2b08081919192b19,
    0x2b08082b08080808,
    0x2b08082b082b0808,
    0x2b08082b2b080808,
    0x2b08082b2b08082b,
    0x2b08082b2b2b0808,
    0x2b08082b2b2b2b08,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190819080808,
    0x2b081908192b0808,
    0x2b0819082b082b19,
    0x2b08191908080808,
    0x2b08191919081908,
    0x2b0819192b2b1919,
    0x2b08192b08192b08,
    0x2b08192b192b2b2b,
    0x2b082b0808080808,
    0x2b082b0808082b08,
    0x2b082b08082b1919,
    0x2b082b0819192b2b,
    0x2b082b082b080808,
    0x2b082b082b08082b,
    0x2b082b082b2b2b08,
    0x2b082b190808192b,
    0x2b082b2b082b082b,
    0x2b082b2b2b080808,
    0x2b082b2b2b082b08,
    0x2b082b2b2b19192b,
    0x2b082b2b2b2b2b08,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808190808,
    0x2b19080819080808,
    0x2b1908081919192b,
    0x2b1908082b081908,
    0x2b19081908080808,
    0x2b190819082b082b,
    0x2b190819192b1908,
    0x2b19082b1919192b,
    0x2b19082b2b082b19,
    0x2b19190808080808,
    0x2b19190808081919,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b19190819192b08,
    0x2b191919082b2b19,
    0x2b1919192b190808,
    0x2b1919192b19082b,
    0x2b19192b19080819,
    0x2b192b0819190819,
    0x2b192b082b2b192b,
    0x2b192b1919082b19,
    0x2b192b2b08191919,
    0x2b192b2b192b0808,
    0x2b2b080808080808,
    0x2b2b08080808082b,
    0x2b2b080808082b08,
    0x2b2b080808082b2b,
    0x2b2b0808082b0808,
    0x2b2b0808082b2b2b,
    0x2b2b08082b2b0808,
    0x2b2b081919190819,
    0x2b2b081919192b19,
    0x2b2b08192b2b192b,
    0x2b2b082b08080808,
    0x2b2b082b0808082b,
    0x2b2b082b08082b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b080808,
    0x2b2b082b2b2b0808,
    0x2b2b190819080808,
    0x2b2b19082b191919,
    0x2b2b192b192b1919,
    0x2b2b192b2b192b08,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b0808,
    0x2b2b2b08082b082b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b0808,
    0x2b2b2b082b2b2b08,
    0x2b2b2b1908081908,
    0x2b2b2b192b081908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b082b2b2b,
    0x2b2b2b2b2b190819,
    0x2b2b2b2b2b2b2b2b,
];

// The IQ2_S codebook is the fixed 1024-entry lattice used by GGML. Each
// entry packs eight unsigned 8-bit magnitudes in little-endian order.
#[allow(clippy::unreadable_literal)]
const IQ2_S_GRID: [u64; 1024] = [
    0x0808080808080808,
    0x080808080808082b,
    0x0808080808081919,
    0x0808080808082b08,
    0x0808080808082b2b,
    0x0808080808190819,
    0x0808080808191908,
    0x080808080819192b,
    0x0808080808192b19,
    0x08080808082b0808,
    0x08080808082b082b,
    0x08080808082b1919,
    0x08080808082b2b08,
    0x0808080819080819,
    0x0808080819081908,
    0x080808081908192b,
    0x0808080819082b19,
    0x0808080819190808,
    0x080808081919082b,
    0x0808080819191919,
    0x0808080819192b08,
    0x08080808192b0819,
    0x08080808192b1908,
    0x08080808192b192b,
    0x08080808192b2b19,
    0x080808082b080808,
    0x080808082b08082b,
    0x080808082b081919,
    0x080808082b082b08,
    0x080808082b190819,
    0x080808082b191908,
    0x080808082b2b0808,
    0x080808082b2b1919,
    0x080808082b2b2b2b,
    0x0808081908080819,
    0x0808081908081908,
    0x080808190808192b,
    0x0808081908082b19,
    0x0808081908190808,
    0x080808190819082b,
    0x0808081908191919,
    0x0808081908192b08,
    0x08080819082b0819,
    0x08080819082b1908,
    0x0808081919080808,
    0x080808191908082b,
    0x0808081919081919,
    0x0808081919082b08,
    0x0808081919190819,
    0x0808081919191908,
    0x080808191919192b,
    0x0808081919192b19,
    0x08080819192b0808,
    0x08080819192b1919,
    0x08080819192b2b08,
    0x080808192b080819,
    0x080808192b081908,
    0x080808192b190808,
    0x080808192b19082b,
    0x080808192b191919,
    0x080808192b2b0819,
    0x080808192b2b1908,
    0x0808082b08080808,
    0x0808082b0808082b,
    0x0808082b08081919,
    0x0808082b08082b08,
    0x0808082b08190819,
    0x0808082b08191908,
    0x0808082b082b0808,
    0x0808082b082b2b2b,
    0x0808082b19080819,
    0x0808082b19081908,
    0x0808082b1908192b,
    0x0808082b19082b19,
    0x0808082b19190808,
    0x0808082b19191919,
    0x0808082b2b080808,
    0x0808082b2b081919,
    0x0808082b2b082b2b,
    0x0808082b2b191908,
    0x0808082b2b2b082b,
    0x0808190808080819,
    0x0808190808081908,
    0x080819080808192b,
    0x0808190808082b19,
    0x0808190808190808,
    0x080819080819082b,
    0x0808190808191919,
    0x0808190808192b08,
    0x08081908082b0819,
    0x08081908082b1908,
    0x08081908082b192b,
    0x08081908082b2b19,
    0x0808190819080808,
    0x080819081908082b,
    0x0808190819081919,
    0x0808190819082b08,
    0x0808190819082b2b,
    0x0808190819190819,
    0x0808190819191908,
    0x080819081919192b,
    0x0808190819192b19,
    0x08081908192b0808,
    0x08081908192b082b,
    0x08081908192b1919,
    0x080819082b080819,
    0x080819082b081908,
    0x080819082b08192b,
    0x080819082b082b19,
    0x080819082b190808,
    0x080819082b191919,
    0x080819082b192b08,
    0x080819082b2b0819,
    0x080819082b2b1908,
    0x0808191908080808,
    0x080819190808082b,
    0x0808191908081919,
    0x0808191908082b08,
    0x0808191908082b2b,
    0x0808191908190819,
    0x0808191908191908,
    0x080819190819192b,
    0x0808191908192b19,
    0x08081919082b0808,
    0x08081919082b1919,
    0x08081919082b2b08,
    0x0808191919080819,
    0x0808191919081908,
    0x080819191908192b,
    0x0808191919082b19,
    0x0808191919190808,
    0x080819191919082b,
    0x0808191919191919,
    0x0808191919192b08,
    0x08081919192b0819,
    0x08081919192b1908,
    0x080819192b080808,
    0x080819192b08082b,
    0x080819192b081919,
    0x080819192b082b08,
    0x080819192b190819,
    0x080819192b191908,
    0x080819192b2b0808,
    0x0808192b08080819,
    0x0808192b08081908,
    0x0808192b0808192b,
    0x0808192b08082b19,
    0x0808192b08190808,
    0x0808192b08191919,
    0x0808192b19080808,
    0x0808192b19081919,
    0x0808192b19082b08,
    0x0808192b19190819,
    0x0808192b19191908,
    0x0808192b192b0808,
    0x0808192b2b080819,
    0x0808192b2b081908,
    0x0808192b2b190808,
    0x08082b0808080808,
    0x08082b080808082b,
    0x08082b0808081919,
    0x08082b0808082b08,
    0x08082b0808190819,
    0x08082b0808191908,
    0x08082b080819192b,
    0x08082b0808192b19,
    0x08082b08082b0808,
    0x08082b08082b1919,
    0x08082b08082b2b2b,
    0x08082b0819080819,
    0x08082b0819081908,
    0x08082b081908192b,
    0x08082b0819082b19,
    0x08082b0819190808,
    0x08082b081919082b,
    0x08082b0819191919,
    0x08082b0819192b08,
    0x08082b08192b0819,
    0x08082b08192b1908,
    0x08082b082b080808,
    0x08082b082b081919,
    0x08082b082b191908,
    0x08082b082b2b2b2b,
    0x08082b1908080819,
    0x08082b1908081908,
    0x08082b1908190808,
    0x08082b190819082b,
    0x08082b1908191919,
    0x08082b1908192b08,
    0x08082b19082b0819,
    0x08082b1919080808,
    0x08082b1919081919,
    0x08082b1919082b08,
    0x08082b1919190819,
    0x08082b1919191908,
    0x08082b19192b0808,
    0x08082b192b080819,
    0x08082b192b190808,
    0x08082b2b08080808,
    0x08082b2b08190819,
    0x08082b2b08191908,
    0x08082b2b082b082b,
    0x08082b2b082b2b08,
    0x08082b2b082b2b2b,
    0x08082b2b19190808,
    0x08082b2b2b192b19,
    0x0819080808080819,
    0x0819080808081908,
    0x081908080808192b,
    0x0819080808082b19,
    0x0819080808190808,
    0x081908080819082b,
    0x0819080808191919,
    0x0819080808192b08,
    0x08190808082b0819,
    0x08190808082b1908,
    0x08190808082b192b,
    0x0819080819080808,
    0x081908081908082b,
    0x0819080819081919,
    0x0819080819082b08,
    0x0819080819190819,
    0x0819080819191908,
    0x081908081919192b,
    0x0819080819192b19,
    0x08190808192b0808,
    0x08190808192b082b,
    0x08190808192b1919,
    0x08190808192b2b08,
    0x081908082b080819,
    0x081908082b081908,
    0x081908082b08192b,
    0x081908082b190808,
    0x081908082b191919,
    0x081908082b192b08,
    0x081908082b2b0819,
    0x081908082b2b1908,
    0x0819081908080808,
    0x081908190808082b,
    0x0819081908081919,
    0x0819081908082b08,
    0x0819081908082b2b,
    0x0819081908190819,
    0x0819081908191908,
    0x081908190819192b,
    0x0819081908192b19,
    0x08190819082b0808,
    0x08190819082b082b,
    0x08190819082b1919,
    0x08190819082b2b08,
    0x0819081919080819,
    0x0819081919081908,
    0x081908191908192b,
    0x0819081919082b19,
    0x0819081919190808,
    0x081908191919082b,
    0x0819081919191919,
    0x0819081919192b08,
    0x08190819192b0819,
    0x08190819192b1908,
    0x081908192b080808,
    0x081908192b08082b,
    0x081908192b081919,
    0x081908192b082b08,
    0x081908192b190819,
    0x081908192b191908,
    0x0819082b08080819,
    0x0819082b08081908,
    0x0819082b08082b19,
    0x0819082b08190808,
    0x0819082b08191919,
    0x0819082b082b0819,
    0x0819082b082b1908,
    0x0819082b19080808,
    0x0819082b19081919,
    0x0819082b19190819,
    0x0819082b19191908,
    0x0819082b2b080819,
    0x0819082b2b081908,
    0x0819082b2b190808,
    0x0819190808080808,
    0x081919080808082b,
    0x0819190808081919,
    0x0819190808082b08,
    0x0819190808190819,
    0x0819190808191908,
    0x081919080819192b,
    0x0819190808192b19,
    0x08191908082b0808,
    0x08191908082b1919,
    0x08191908082b2b08,
    0x0819190819080819,
    0x0819190819081908,
    0x081919081908192b,
    0x0819190819082b19,
    0x0819190819190808,
    0x081919081919082b,
    0x0819190819191919,
    0x0819190819192b08,
    0x08191908192b0819,
    0x08191908192b1908,
    0x081919082b080808,
    0x081919082b08082b,
    0x081919082b081919,
    0x081919082b082b08,
    0x081919082b190819,
    0x081919082b191908,
    0x081919082b2b0808,
    0x0819191908080819,
    0x0819191908081908,
    0x081919190808192b,
    0x0819191908082b19,
    0x0819191908190808,
    0x081919190819082b,
    0x0819191908191919,
    0x0819191908192b08,
    0x08191919082b0819,
    0x08191919082b1908,
    0x0819191919080808,
    0x081919191908082b,
    0x0819191919081919,
    0x0819191919082b08,
    0x0819191919190819,
    0x0819191919191908,
    0x08191919192b0808,
    0x081919192b080819,
    0x081919192b081908,
    0x081919192b190808,
    0x0819192b08080808,
    0x0819192b08081919,
    0x0819192b08082b08,
    0x0819192b08190819,
    0x0819192b08191908,
    0x0819192b082b0808,
    0x0819192b19080819,
    0x0819192b19081908,
    0x0819192b19190808,
    0x0819192b2b080808,
    0x0819192b2b2b2b2b,
    0x08192b0808080819,
    0x08192b0808081908,
    0x08192b080808192b,
    0x08192b0808082b19,
    0x08192b0808190808,
    0x08192b0808191919,
    0x08192b0808192b08,
    0x08192b08082b0819,
    0x08192b0819080808,
    0x08192b081908082b,
    0x08192b0819081919,
    0x08192b0819082b08,
    0x08192b0819190819,
    0x08192b0819191908,
    0x08192b08192b0808,
    0x08192b082b080819,
    0x08192b082b081908,
    0x08192b1908080808,
    0x08192b190808082b,
    0x08192b1908081919,
    0x08192b1908082b08,
    0x08192b1908190819,
    0x08192b1908191908,
    0x08192b19082b0808,
    0x08192b1919080819,
    0x08192b1919081908,
    0x08192b1919190808,
    0x08192b19192b2b19,
    0x08192b192b2b082b,
    0x08192b2b08081908,
    0x08192b2b08190808,
    0x08192b2b19080808,
    0x08192b2b1919192b,
    0x082b080808080808,
    0x082b08080808082b,
    0x082b080808081919,
    0x082b080808082b08,
    0x082b080808190819,
    0x082b080808191908,
    0x082b08080819192b,
    0x082b080808192b19,
    0x082b0808082b0808,
    0x082b0808082b1919,
    0x082b0808082b2b2b,
    0x082b080819080819,
    0x082b080819081908,
    0x082b080819190808,
    0x082b08081919082b,
    0x082b080819191919,
    0x082b0808192b1908,
    0x082b08082b080808,
    0x082b08082b082b2b,
    0x082b08082b191908,
    0x082b08082b2b2b2b,
    0x082b081908080819,
    0x082b081908081908,
    0x082b081908190808,
    0x082b08190819082b,
    0x082b081908191919,
    0x082b0819082b0819,
    0x082b081919080808,
    0x082b08191908082b,
    0x082b081919081919,
    0x082b081919190819,
    0x082b081919191908,
    0x082b0819192b0808,
    0x082b08192b080819,
    0x082b08192b081908,
    0x082b08192b190808,
    0x082b082b08080808,
    0x082b082b08082b2b,
    0x082b082b082b082b,
    0x082b082b082b2b08,
    0x082b082b082b2b2b,
    0x082b082b19081908,
    0x082b082b19190808,
    0x082b082b2b082b08,
    0x082b082b2b082b2b,
    0x082b082b2b2b2b08,
    0x082b190808080819,
    0x082b190808081908,
    0x082b19080808192b,
    0x082b190808082b19,
    0x082b190808190808,
    0x082b190808191919,
    0x082b190808192b08,
    0x082b1908082b0819,
    0x082b1908082b1908,
    0x082b190819080808,
    0x082b19081908082b,
    0x082b190819081919,
    0x082b190819082b08,
    0x082b190819190819,
    0x082b190819191908,
    0x082b1908192b0808,
    0x082b19082b080819,
    0x082b19082b081908,
    0x082b19082b190808,
    0x082b191908080808,
    0x082b191908081919,
    0x082b191908082b08,
    0x082b191908190819,
    0x082b191908191908,
    0x082b1919082b0808,
    0x082b191919080819,
    0x082b191919081908,
    0x082b191919190808,
    0x082b1919192b192b,
    0x082b19192b080808,
    0x082b192b08080819,
    0x082b192b08081908,
    0x082b192b08190808,
    0x082b192b19080808,
    0x082b192b19192b19,
    0x082b2b0808080808,
    0x082b2b0808081919,
    0x082b2b0808190819,
    0x082b2b0808191908,
    0x082b2b0819080819,
    0x082b2b0819081908,
    0x082b2b0819190808,
    0x082b2b082b082b2b,
    0x082b2b082b2b2b2b,
    0x082b2b1908080819,
    0x082b2b1908081908,
    0x082b2b1908190808,
    0x082b2b192b191919,
    0x082b2b2b08082b2b,
    0x082b2b2b082b082b,
    0x082b2b2b192b1908,
    0x082b2b2b2b082b08,
    0x082b2b2b2b082b2b,
    0x1908080808080819,
    0x1908080808081908,
    0x190808080808192b,
    0x1908080808082b19,
    0x1908080808190808,
    0x190808080819082b,
    0x1908080808191919,
    0x1908080808192b08,
    0x1908080808192b2b,
    0x19080808082b0819,
    0x19080808082b1908,
    0x19080808082b192b,
    0x1908080819080808,
    0x190808081908082b,
    0x1908080819081919,
    0x1908080819082b08,
    0x1908080819082b2b,
    0x1908080819190819,
    0x1908080819191908,
    0x190808081919192b,
    0x1908080819192b19,
    0x19080808192b0808,
    0x19080808192b082b,
    0x19080808192b1919,
    0x190808082b080819,
    0x190808082b081908,
    0x190808082b190808,
    0x190808082b191919,
    0x190808082b192b08,
    0x190808082b2b0819,
    0x190808082b2b1908,
    0x1908081908080808,
    0x190808190808082b,
    0x1908081908081919,
    0x1908081908082b08,
    0x1908081908190819,
    0x1908081908191908,
    0x190808190819192b,
    0x1908081908192b19,
    0x19080819082b0808,
    0x19080819082b082b,
    0x19080819082b1919,
    0x1908081919080819,
    0x1908081919081908,
    0x190808191908192b,
    0x1908081919082b19,
    0x1908081919190808,
    0x190808191919082b,
    0x1908081919191919,
    0x1908081919192b08,
    0x19080819192b0819,
    0x19080819192b1908,
    0x190808192b080808,
    0x190808192b08082b,
    0x190808192b081919,
    0x190808192b082b08,
    0x190808192b190819,
    0x190808192b191908,
    0x190808192b2b0808,
    0x1908082b08080819,
    0x1908082b08081908,
    0x1908082b08190808,
    0x1908082b0819082b,
    0x1908082b08191919,
    0x1908082b08192b08,
    0x1908082b082b1908,
    0x1908082b19080808,
    0x1908082b19081919,
    0x1908082b19082b08,
    0x1908082b19190819,
    0x1908082b19191908,
    0x1908082b192b0808,
    0x1908082b2b080819,
    0x1908082b2b081908,
    0x1908190808080808,
    0x190819080808082b,
    0x1908190808081919,
    0x1908190808082b08,
    0x1908190808082b2b,
    0x1908190808190819,
    0x1908190808191908,
    0x190819080819192b,
    0x1908190808192b19,
    0x19081908082b0808,
    0x19081908082b082b,
    0x19081908082b1919,
    0x19081908082b2b08,
    0x1908190819080819,
    0x1908190819081908,
    0x190819081908192b,
    0x1908190819082b19,
    0x1908190819190808,
    0x190819081919082b,
    0x1908190819191919,
    0x1908190819192b08,
    0x19081908192b0819,
    0x19081908192b1908,
    0x190819082b080808,
    0x190819082b08082b,
    0x190819082b081919,
    0x190819082b082b08,
    0x190819082b190819,
    0x190819082b191908,
    0x190819082b2b0808,
    0x1908191908080819,
    0x1908191908081908,
    0x190819190808192b,
    0x1908191908082b19,
    0x1908191908190808,
    0x190819190819082b,
    0x1908191908191919,
    0x1908191908192b08,
    0x19081919082b0819,
    0x19081919082b1908,
    0x1908191919080808,
    0x190819191908082b,
    0x1908191919081919,
    0x1908191919082b08,
    0x1908191919190819,
    0x1908191919191908,
    0x19081919192b0808,
    0x19081919192b2b2b,
    0x190819192b080819,
    0x190819192b081908,
    0x190819192b190808,
    0x1908192b08080808,
    0x1908192b0808082b,
    0x1908192b08081919,
    0x1908192b08082b08,
    0x1908192b08190819,
    0x1908192b08191908,
    0x1908192b082b0808,
    0x1908192b19080819,
    0x1908192b19081908,
    0x1908192b19190808,
    0x1908192b2b080808,
    0x1908192b2b2b1919,
    0x19082b0808080819,
    0x19082b0808081908,
    0x19082b0808082b19,
    0x19082b0808190808,
    0x19082b080819082b,
    0x19082b0808191919,
    0x19082b0808192b08,
    0x19082b08082b0819,
    0x19082b08082b1908,
    0x19082b0819080808,
    0x19082b081908082b,
    0x19082b0819081919,
    0x19082b0819082b08,
    0x19082b0819190819,
    0x19082b0819191908,
    0x19082b08192b0808,
    0x19082b082b081908,
    0x19082b082b190808,
    0x19082b1908080808,
    0x19082b190808082b,
    0x19082b1908081919,
    0x19082b1908082b08,
    0x19082b1908190819,
    0x19082b1908191908,
    0x19082b19082b0808,
    0x19082b1919080819,
    0x19082b1919081908,
    0x19082b1919190808,
    0x19082b192b080808,
    0x19082b192b19192b,
    0x19082b2b08080819,
    0x19082b2b08081908,
    0x19082b2b08190808,
    0x19082b2b19080808,
    0x1919080808080808,
    0x191908080808082b,
    0x1919080808081919,
    0x1919080808082b08,
    0x1919080808190819,
    0x1919080808191908,
    0x191908080819192b,
    0x1919080808192b19,
    0x19190808082b0808,
    0x19190808082b082b,
    0x19190808082b1919,
    0x19190808082b2b08,
    0x1919080819080819,
    0x1919080819081908,
    0x191908081908192b,
    0x1919080819082b19,
    0x1919080819190808,
    0x191908081919082b,
    0x1919080819191919,
    0x1919080819192b08,
    0x19190808192b0819,
    0x19190808192b1908,
    0x191908082b080808,
    0x191908082b08082b,
    0x191908082b081919,
    0x191908082b082b08,
    0x191908082b190819,
    0x191908082b191908,
    0x1919081908080819,
    0x1919081908081908,
    0x191908190808192b,
    0x1919081908082b19,
    0x1919081908190808,
    0x191908190819082b,
    0x1919081908191919,
    0x1919081908192b08,
    0x19190819082b0819,
    0x19190819082b1908,
    0x1919081919080808,
    0x191908191908082b,
    0x1919081919081919,
    0x1919081919082b08,
    0x1919081919190819,
    0x1919081919191908,
    0x19190819192b0808,
    0x191908192b080819,
    0x191908192b081908,
    0x191908192b190808,
    0x1919082b08080808,
    0x1919082b08081919,
    0x1919082b08082b08,
    0x1919082b08190819,
    0x1919082b08191908,
    0x1919082b082b0808,
    0x1919082b19080819,
    0x1919082b19081908,
    0x1919082b19190808,
    0x1919082b192b2b19,
    0x1919082b2b080808,
    0x1919190808080819,
    0x1919190808081908,
    0x191919080808192b,
    0x1919190808082b19,
    0x1919190808190808,
    0x191919080819082b,
    0x1919190808191919,
    0x1919190808192b08,
    0x19191908082b0819,
    0x19191908082b1908,
    0x1919190819080808,
    0x191919081908082b,
    0x1919190819081919,
    0x1919190819082b08,
    0x1919190819190819,
    0x1919190819191908,
    0x19191908192b0808,
    0x191919082b080819,
    0x191919082b081908,
    0x191919082b190808,
    0x1919191908080808,
    0x191919190808082b,
    0x1919191908081919,
    0x1919191908082b08,
    0x1919191908190819,
    0x1919191908191908,
    0x19191919082b0808,
    0x1919191919080819,
    0x1919191919081908,
    0x1919191919190808,
    0x191919192b080808,
    0x1919192b08080819,
    0x1919192b08081908,
    0x1919192b08190808,
    0x1919192b082b192b,
    0x1919192b19080808,
    0x19192b0808080808,
    0x19192b080808082b,
    0x19192b0808081919,
    0x19192b0808082b08,
    0x19192b0808190819,
    0x19192b0808191908,
    0x19192b08082b0808,
    0x19192b0819080819,
    0x19192b0819081908,
    0x19192b0819190808,
    0x19192b0819192b2b,
    0x19192b082b080808,
    0x19192b1908080819,
    0x19192b1908081908,
    0x19192b1908190808,
    0x19192b1919080808,
    0x19192b2b08080808,
    0x19192b2b08192b19,
    0x19192b2b2b081919,
    0x19192b2b2b2b2b08,
    0x192b080808080819,
    0x192b080808081908,
    0x192b08080808192b,
    0x192b080808190808,
    0x192b08080819082b,
    0x192b080808191919,
    0x192b080808192b08,
    0x192b0808082b0819,
    0x192b0808082b1908,
    0x192b080819080808,
    0x192b080819081919,
    0x192b080819082b08,
    0x192b080819190819,
    0x192b080819191908,
    0x192b0808192b0808,
    0x192b08082b081908,
    0x192b08082b190808,
    0x192b081908080808,
    0x192b08190808082b,
    0x192b081908081919,
    0x192b081908082b08,
    0x192b081908190819,
    0x192b081908191908,
    0x192b0819082b0808,
    0x192b081919080819,
    0x192b081919081908,
    0x192b081919190808,
    0x192b08192b080808,
    0x192b08192b192b19,
    0x192b082b08081908,
    0x192b082b08190808,
    0x192b082b19080808,
    0x192b082b1919192b,
    0x192b082b2b2b0819,
    0x192b190808080808,
    0x192b190808081919,
    0x192b190808082b08,
    0x192b190808190819,
    0x192b190808191908,
    0x192b1908082b0808,
    0x192b190819080819,
    0x192b190819081908,
    0x192b190819190808,
    0x192b19082b080808,
    0x192b191908080819,
    0x192b191908081908,
    0x192b191908190808,
    0x192b191919080808,
    0x192b191919082b2b,
    0x192b1919192b2b08,
    0x192b19192b19082b,
    0x192b192b08080808,
    0x192b192b2b191908,
    0x192b2b0808080819,
    0x192b2b0808081908,
    0x192b2b0808190808,
    0x192b2b08192b1919,
    0x192b2b082b192b08,
    0x192b2b1908080808,
    0x192b2b19082b2b2b,
    0x192b2b2b1908082b,
    0x192b2b2b2b2b0819,
    0x2b08080808080808,
    0x2b0808080808082b,
    0x2b08080808081919,
    0x2b08080808082b08,
    0x2b08080808190819,
    0x2b08080808191908,
    0x2b08080808192b19,
    0x2b080808082b0808,
    0x2b080808082b1919,
    0x2b08080819080819,
    0x2b08080819081908,
    0x2b08080819190808,
    0x2b0808081919082b,
    0x2b08080819191919,
    0x2b08080819192b08,
    0x2b080808192b0819,
    0x2b0808082b080808,
    0x2b0808082b081919,
    0x2b0808082b190819,
    0x2b0808082b191908,
    0x2b08081908080819,
    0x2b08081908081908,
    0x2b08081908082b19,
    0x2b08081908190808,
    0x2b0808190819082b,
    0x2b08081908191919,
    0x2b08081908192b08,
    0x2b080819082b0819,
    0x2b080819082b1908,
    0x2b08081919080808,
    0x2b0808191908082b,
    0x2b08081919081919,
    0x2b08081919082b08,
    0x2b08081919190819,
    0x2b08081919191908,
    0x2b0808192b080819,
    0x2b0808192b081908,
    0x2b0808192b190808,
    0x2b0808192b2b2b19,
    0x2b08082b08080808,
    0x2b08082b08081919,
    0x2b08082b08082b2b,
    0x2b08082b08190819,
    0x2b08082b08191908,
    0x2b08082b19080819,
    0x2b08082b19081908,
    0x2b08082b19190808,
    0x2b08190808080819,
    0x2b08190808081908,
    0x2b0819080808192b,
    0x2b08190808082b19,
    0x2b08190808190808,
    0x2b0819080819082b,
    0x2b08190808191919,
    0x2b08190808192b08,
    0x2b081908082b0819,
    0x2b08190819080808,
    0x2b0819081908082b,
    0x2b08190819081919,
    0x2b08190819082b08,
    0x2b08190819190819,
    0x2b08190819191908,
    0x2b081908192b0808,
    0x2b0819082b080819,
    0x2b0819082b081908,
    0x2b0819082b190808,
    0x2b08191908080808,
    0x2b0819190808082b,
    0x2b08191908081919,
    0x2b08191908082b08,
    0x2b08191908190819,
    0x2b08191908191908,
    0x2b081919082b0808,
    0x2b08191919080819,
    0x2b08191919081908,
    0x2b08191919190808,
    0x2b0819192b080808,
    0x2b0819192b082b2b,
    0x2b08192b08080819,
    0x2b08192b08081908,
    0x2b08192b08190808,
    0x2b08192b082b2b19,
    0x2b08192b19080808,
    0x2b082b0808080808,
    0x2b082b0808081919,
    0x2b082b0808190819,
    0x2b082b0808191908,
    0x2b082b0819080819,
    0x2b082b0819081908,
    0x2b082b0819190808,
    0x2b082b082b2b082b,
    0x2b082b1908080819,
    0x2b082b1908081908,
    0x2b082b1919080808,
    0x2b082b19192b1919,
    0x2b082b2b082b082b,
    0x2b082b2b19192b08,
    0x2b082b2b19192b2b,
    0x2b082b2b2b08082b,
    0x2b082b2b2b2b082b,
    0x2b19080808080819,
    0x2b19080808081908,
    0x2b19080808082b19,
    0x2b19080808190808,
    0x2b1908080819082b,
    0x2b19080808191919,
    0x2b19080808192b08,
    0x2b190808082b1908,
    0x2b19080819080808,
    0x2b1908081908082b,
    0x2b19080819081919,
    0x2b19080819082b08,
    0x2b19080819190819,
    0x2b19080819191908,
    0x2b190808192b0808,
    0x2b1908082b080819,
    0x2b1908082b081908,
    0x2b1908082b190808,
    0x2b19081908080808,
    0x2b19081908081919,
    0x2b19081908190819,
    0x2b19081908191908,
    0x2b19081919080819,
    0x2b19081919081908,
    0x2b19081919190808,
    0x2b19081919192b2b,
    0x2b19082b08080819,
    0x2b19082b08081908,
    0x2b19082b08190808,
    0x2b19082b19080808,
    0x2b19082b2b2b192b,
    0x2b19190808080808,
    0x2b1919080808082b,
    0x2b19190808081919,
    0x2b19190808082b08,
    0x2b19190808190819,
    0x2b19190808191908,
    0x2b191908082b0808,
    0x2b19190819080819,
    0x2b19190819081908,
    0x2b19190819190808,
    0x2b1919082b080808,
    0x2b1919082b19192b,
    0x2b19191908080819,
    0x2b19191908081908,
    0x2b19191908190808,
    0x2b19191919080808,
    0x2b1919192b192b08,
    0x2b1919192b2b0819,
    0x2b19192b08080808,
    0x2b19192b1908192b,
    0x2b19192b192b1908,
    0x2b192b0808080819,
    0x2b192b0808081908,
    0x2b192b0808190808,
    0x2b192b08082b192b,
    0x2b192b0819080808,
    0x2b192b082b2b2b19,
    0x2b192b1908080808,
    0x2b192b1919082b19,
    0x2b192b191919082b,
    0x2b192b2b2b190808,
    0x2b2b080808080808,
    0x2b2b080808081919,
    0x2b2b080808082b2b,
    0x2b2b080808191908,
    0x2b2b0808082b082b,
    0x2b2b0808082b2b2b,
    0x2b2b080819080819,
    0x2b2b080819081908,
    0x2b2b080819190808,
    0x2b2b08082b2b082b,
    0x2b2b08082b2b2b2b,
    0x2b2b081919080808,
    0x2b2b0819192b1919,
    0x2b2b082b0808082b,
    0x2b2b082b08082b2b,
    0x2b2b082b082b082b,
    0x2b2b082b082b2b08,
    0x2b2b082b082b2b2b,
    0x2b2b082b2b08082b,
    0x2b2b082b2b082b08,
    0x2b2b082b2b082b2b,
    0x2b2b082b2b2b2b08,
    0x2b2b190808080819,
    0x2b2b190808081908,
    0x2b2b190808190808,
    0x2b2b190819080808,
    0x2b2b19082b082b19,
    0x2b2b19082b2b1908,
    0x2b2b191908080808,
    0x2b2b191908192b19,
    0x2b2b192b19190819,
    0x2b2b2b0808082b2b,
    0x2b2b2b08082b2b08,
    0x2b2b2b082b2b082b,
    0x2b2b2b1919191908,
    0x2b2b2b192b08192b,
    0x2b2b2b2b08082b08,
    0x2b2b2b2b08082b2b,
    0x2b2b2b2b082b0808,
    0x2b2b2b2b082b082b,
    0x2b2b2b2b082b2b08,
    0x2b2b2b2b2b082b08,
    0x2b2b2b2b2b2b2b2b,
];

// The IQ3_XXS codebook stores two four-value grids in each little-endian
// 32-bit entry. The table is shared by the scalar decoder and direct lookup.
#[allow(clippy::unreadable_literal)]
const IQ3_XXS_GRID: [u32; 256] = [
    0x04040404, 0x04040414, 0x04040424, 0x04040c0c, 0x04040c1c, 0x04040c3e, 0x04041404, 0x04041414,
    0x04041c0c, 0x04042414, 0x04043e1c, 0x04043e2c, 0x040c040c, 0x040c041c, 0x040c0c04, 0x040c0c14,
    0x040c140c, 0x040c142c, 0x040c1c04, 0x040c1c14, 0x040c240c, 0x040c2c24, 0x040c3e04, 0x04140404,
    0x04140414, 0x04140424, 0x04140c0c, 0x04141404, 0x04141414, 0x04141c0c, 0x04141c1c, 0x04141c3e,
    0x04142c0c, 0x04142c3e, 0x04143e2c, 0x041c040c, 0x041c043e, 0x041c0c04, 0x041c0c14, 0x041c142c,
    0x041c3e04, 0x04240c1c, 0x04241c3e, 0x04242424, 0x04242c3e, 0x04243e1c, 0x04243e2c, 0x042c040c,
    0x042c043e, 0x042c1c14, 0x042c2c14, 0x04341c2c, 0x04343424, 0x043e0c04, 0x043e0c24, 0x043e0c34,
    0x043e241c, 0x043e340c, 0x0c04040c, 0x0c04041c, 0x0c040c04, 0x0c040c14, 0x0c04140c, 0x0c04141c,
    0x0c041c04, 0x0c041c14, 0x0c041c24, 0x0c04243e, 0x0c042c04, 0x0c0c0404, 0x0c0c0414, 0x0c0c0c0c,
    0x0c0c1404, 0x0c0c1414, 0x0c14040c, 0x0c14041c, 0x0c140c04, 0x0c140c14, 0x0c14140c, 0x0c141c04,
    0x0c143e14, 0x0c1c0404, 0x0c1c0414, 0x0c1c1404, 0x0c1c1c0c, 0x0c1c2434, 0x0c1c3434, 0x0c24040c,
    0x0c24042c, 0x0c242c04, 0x0c2c1404, 0x0c2c1424, 0x0c2c2434, 0x0c2c3e0c, 0x0c34042c, 0x0c3e1414,
    0x0c3e2404, 0x14040404, 0x14040414, 0x14040c0c, 0x14040c1c, 0x14041404, 0x14041414, 0x14041434,
    0x14041c0c, 0x14042414, 0x140c040c, 0x140c041c, 0x140c042c, 0x140c0c04, 0x140c0c14, 0x140c140c,
    0x140c1c04, 0x140c341c, 0x140c343e, 0x140c3e04, 0x14140404, 0x14140414, 0x14140c0c, 0x14140c3e,
    0x14141404, 0x14141414, 0x14141c3e, 0x14142404, 0x14142c2c, 0x141c040c, 0x141c0c04, 0x141c0c24,
    0x141c3e04, 0x141c3e24, 0x14241c2c, 0x14242c1c, 0x142c041c, 0x142c143e, 0x142c240c, 0x142c3e24,
    0x143e040c, 0x143e041c, 0x143e0c34, 0x143e242c, 0x1c04040c, 0x1c040c04, 0x1c040c14, 0x1c04140c,
    0x1c04141c, 0x1c042c04, 0x1c04342c, 0x1c043e14, 0x1c0c0404, 0x1c0c0414, 0x1c0c1404, 0x1c0c1c0c,
    0x1c0c2424, 0x1c0c2434, 0x1c14040c, 0x1c14041c, 0x1c140c04, 0x1c14142c, 0x1c142c14, 0x1c143e14,
    0x1c1c0c0c, 0x1c1c1c1c, 0x1c241c04, 0x1c24243e, 0x1c243e14, 0x1c2c0404, 0x1c2c0434, 0x1c2c1414,
    0x1c2c2c2c, 0x1c340c24, 0x1c341c34, 0x1c34341c, 0x1c3e1c1c, 0x1c3e3404, 0x24040424, 0x24040c3e,
    0x24041c2c, 0x24041c3e, 0x24042c1c, 0x24042c3e, 0x240c3e24, 0x24141404, 0x24141c3e, 0x24142404,
    0x24143404, 0x24143434, 0x241c043e, 0x241c242c, 0x24240424, 0x24242c0c, 0x24243424, 0x242c142c,
    0x242c241c, 0x242c3e04, 0x243e042c, 0x243e0c04, 0x243e0c14, 0x243e1c04, 0x2c040c14, 0x2c04240c,
    0x2c043e04, 0x2c0c0404, 0x2c0c0434, 0x2c0c1434, 0x2c0c2c2c, 0x2c140c24, 0x2c141c14, 0x2c143e14,
    0x2c1c0414, 0x2c1c2c1c, 0x2c240c04, 0x2c24141c, 0x2c24143e, 0x2c243e14, 0x2c2c0414, 0x2c2c1c0c,
    0x2c342c04, 0x2c3e1424, 0x2c3e2414, 0x34041424, 0x34042424, 0x34042434, 0x34043424, 0x340c140c,
    0x340c340c, 0x34140c3e, 0x34143424, 0x341c1c04, 0x341c1c34, 0x34242424, 0x342c042c, 0x342c2c14,
    0x34341c1c, 0x343e041c, 0x343e140c, 0x3e04041c, 0x3e04042c, 0x3e04043e, 0x3e040c04, 0x3e041c14,
    0x3e042c14, 0x3e0c1434, 0x3e0c2404, 0x3e140c14, 0x3e14242c, 0x3e142c14, 0x3e1c0404, 0x3e1c0c2c,
    0x3e1c1c1c, 0x3e1c3404, 0x3e24140c, 0x3e24240c, 0x3e2c0404, 0x3e2c0414, 0x3e2c1424, 0x3e341c04,
];

// The IQ3_S codebook is the fixed 512-entry lattice used by GGML. Each
// entry packs four unsigned 8-bit magnitudes in little-endian order.
#[allow(clippy::unreadable_literal)]
const IQ3_S_GRID: [u32; 512] = [
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
];

const IQ4_NL_VALUES: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

const MXFP4_VALUES: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

const TQ1_POW3: [u16; 5] = [1, 3, 9, 27, 81];

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
        16 => decode_iq2_xxs(bytes),
        17 => decode_iq2_xs(bytes),
        22 => decode_iq2_s(bytes),
        18 => decode_iq3_xxs(bytes),
        19 => decode_iq1_s(bytes),
        21 => decode_iq3_s(bytes),
        20 => decode_iq4_nl(bytes),
        23 => decode_iq4_xs(bytes),
        39 => decode_mxfp4(bytes),
        40 => decode_nvfp4(bytes),
        34 => decode_tq1_0(bytes),
        35 => decode_tq2_0(bytes),
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
        && matches!(
            descriptor.value_type.raw(),
            2 | 3 | 6 | 7 | 8 | 16 | 17 | 18 | 20 | 22 | 23 | 34 | 35 | 39 | 40
        )
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
        let mut iq2_block = None;
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
                let value = if matches!(descriptor.value_type.raw(), 16..=18) {
                    let block_index = index / 256;
                    let block_offset = index % 256;
                    let needs_reload = iq2_block
                        .as_ref()
                        .is_none_or(|(cached_index, _)| *cached_index != block_index);
                    if needs_reload {
                        let block_start =
                            block_index.checked_mul(block_bytes).ok_or_else(|| {
                                ModelError::Shape("IQ2 block offset overflows".to_owned())
                            })?;
                        let block_end = block_start.checked_add(block_bytes).ok_or_else(|| {
                            ModelError::Shape("IQ2 block range overflows".to_owned())
                        })?;
                        let values = match descriptor.value_type.raw() {
                            16 => {
                                let block = tensor_bytes
                                    .get(block_start..block_end)
                                    .and_then(|slice| <&[u8; 66]>::try_from(slice).ok())
                                    .ok_or_else(|| {
                                        ModelError::Shape(
                                            "IQ2_XXS block is outside the tensor".to_owned(),
                                        )
                                    })?;
                                decode_iq2_xxs_block(block)
                            }
                            17 => {
                                let block = tensor_bytes
                                    .get(block_start..block_end)
                                    .and_then(|slice| <&[u8; 74]>::try_from(slice).ok())
                                    .ok_or_else(|| {
                                        ModelError::Shape(
                                            "IQ2_XS block is outside the tensor".to_owned(),
                                        )
                                    })?;
                                decode_iq2_xs_block(block)
                            }
                            18 => {
                                let block = tensor_bytes
                                    .get(block_start..block_end)
                                    .and_then(|slice| <&[u8; 98]>::try_from(slice).ok())
                                    .ok_or_else(|| {
                                        ModelError::Shape(
                                            "IQ3_XXS block is outside the tensor".to_owned(),
                                        )
                                    })?;
                                decode_iq3_xxs_block(block)
                            }
                            _ => unreachable!("IQ block type validated above"),
                        };
                        iq2_block = Some((block_index, values));
                    }
                    iq2_block
                        .as_ref()
                        .map(|(_, values)| values[block_offset])
                        .ok_or_else(|| {
                            ModelError::Shape("IQ2_XXS block cache is empty".to_owned())
                        })?
                } else {
                    quantized_value_at(descriptor.value_type, tensor_bytes, index)?
                };
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
        2 | 20 => Some((32, 18)),
        3 => Some((32, 20)),
        6 => Some((32, 22)),
        7 => Some((32, 24)),
        8 => Some((32, 34)),
        10 => Some((256, 84)),
        11 | 21 => Some((256, 110)),
        12 => Some((256, 144)),
        13 => Some((256, 176)),
        14 => Some((256, 210)),
        15 => Some((256, 292)),
        16 | 35 => Some((256, 66)),
        17 => Some((256, 74)),
        22 => Some((256, 82)),
        18 => Some((256, 98)),
        19 => Some((256, 50)),
        23 => Some((256, 136)),
        39 => Some((32, 17)),
        40 => Some((64, 36)),
        34 => Some((256, 54)),
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
        16 => iq2_xxs_value_at(bytes, index),
        17 => iq2_xs_value_at(bytes, index),
        22 => iq2_s_value_at(bytes, index),
        18 => iq3_xxs_value_at(bytes, index),
        19 => iq1_s_value_at(bytes, index),
        21 => iq3_s_value_at(bytes, index),
        20 => iq4_nl_value_at(bytes, index),
        23 => iq4_xs_value_at(bytes, index),
        39 => mxfp4_value_at(bytes, index),
        40 => nvfp4_value_at(bytes, index),
        34 => tq1_0_value_at(bytes, index),
        35 => tq2_0_value_at(bytes, index),
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

fn decode_iq4_nl(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 18;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ4_NL tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        for index in 0..16 {
            let packed = block[2 + index];
            values.push(scale * f32::from(IQ4_NL_VALUES[usize::from(packed & 0x0f)]));
        }
        for index in 0..16 {
            let packed = block[2 + index];
            values.push(scale * f32::from(IQ4_NL_VALUES[usize::from(packed >> 4)]));
        }
    }
    Ok(values)
}

fn decode_iq4_xs(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 136;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ4_XS tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
        let scales_high = u16::from_le_bytes([block[2], block[3]]);
        let scales_low = &block[4..8];
        let quantized = &block[8..];
        for sub_block in 0..8 {
            let scale_index = sub_block / 2;
            let scale_shift = 4 * (sub_block % 2);
            let quantized_scale = ((scales_low[scale_index] >> scale_shift) & 0x0f)
                | ((((scales_high >> (2 * sub_block)) & 0x03) as u8) << 4);
            let sub_scale = scale * (f32::from(quantized_scale) - 32.0);
            for index in 0..16 {
                let packed = quantized[sub_block * 16 + index];
                values.push(sub_scale * f32::from(IQ4_NL_VALUES[usize::from(packed & 0x0f)]));
            }
            for index in 0..16 {
                let packed = quantized[sub_block * 16 + index];
                values.push(sub_scale * f32::from(IQ4_NL_VALUES[usize::from(packed >> 4)]));
            }
        }
    }
    Ok(values)
}

fn e8m0_to_f32_half(exponent: u8) -> f32 {
    let bits = if exponent < 2 {
        0x0020_0000_u32 << exponent
    } else {
        u32::from(exponent - 1) << 23
    };
    f32::from_bits(bits)
}

fn decode_mxfp4(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 17;
    const BLOCK_VALUES: usize = 32;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "MXFP4 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = e8m0_to_f32_half(block[0]);
        for index in 0..16 {
            let packed = block[1 + index];
            values.push(scale * f32::from(MXFP4_VALUES[usize::from(packed & 0x0f)]));
        }
        for index in 0..16 {
            let packed = block[1 + index];
            values.push(scale * f32::from(MXFP4_VALUES[usize::from(packed >> 4)]));
        }
    }
    Ok(values)
}

fn ue4m3_to_f32_half(value: u8) -> f32 {
    if value == 0 || value == 0x7f {
        return 0.0;
    }
    let exponent = (value >> 3) & 0x0f;
    let mantissa = value & 0x07;
    if exponent == 0 {
        f32::from(mantissa) * 0.000_976_562_5
    } else {
        let bits = (u32::from(exponent) + 120) << 23 | u32::from(mantissa) << 20;
        f32::from_bits(bits) * 0.5
    }
}

fn decode_nvfp4(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 36;
    const BLOCK_VALUES: usize = 64;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "NVFP4 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        for sub_block in 0..4 {
            let scale = ue4m3_to_f32_half(block[sub_block]);
            for index in 0..8 {
                let packed = block[4 + sub_block * 8 + index];
                values.push(scale * f32::from(MXFP4_VALUES[usize::from(packed & 0x0f)]));
            }
            for index in 0..8 {
                let packed = block[4 + sub_block * 8 + index];
                values.push(scale * f32::from(MXFP4_VALUES[usize::from(packed >> 4)]));
            }
        }
    }
    Ok(values)
}

fn tq1_digit(packed: u8, power: usize) -> f32 {
    let value = (u16::from(packed) * TQ1_POW3[power] * 3) >> 8;
    f32::from(value.cast_signed() - 1)
}

fn decode_tq1_0(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 54;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "TQ1_0 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
        for power in 0..5 {
            for &packed in &block[..32] {
                values.push(scale * tq1_digit(packed, power));
            }
        }
        for power in 0..5 {
            for &packed in &block[32..48] {
                values.push(scale * tq1_digit(packed, power));
            }
        }
        for power in 0..4 {
            for &packed in &block[48..52] {
                values.push(scale * tq1_digit(packed, power));
            }
        }
    }
    Ok(values)
}

fn decode_tq2_0(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 66;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "TQ2_0 tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        let scale = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
        for chunk in 0..2 {
            for shift in (0..4).map(|index| index * 2) {
                for index in 0..32 {
                    let quantized = (block[chunk * 32 + index] >> shift) & 0x03;
                    values.push(scale * (f32::from(quantized) - 1.0));
                }
            }
        }
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_xxs(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 66;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ2_XXS tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq2_xxs_block(block));
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_xxs_block(block: &[u8; 66]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for ib32 in 0..8 {
        let offset = ib32 * 8;
        let aux32_g = u32::from(u16::from_le_bytes([qs[offset], qs[offset + 1]]))
            | (u32::from(u16::from_le_bytes([qs[offset + 2], qs[offset + 3]])) << 16);
        let aux32_s = u32::from(u16::from_le_bytes([qs[offset + 4], qs[offset + 5]]))
            | (u32::from(u16::from_le_bytes([qs[offset + 6], qs[offset + 7]])) << 16);
        let block_scale = scale * (0.5 + (aux32_s >> 28) as f32) * 0.25;
        for group in 0..4 {
            let grid = IQ2_XXS_GRID[((aux32_g >> (8 * group)) & 0xff) as usize].to_le_bytes();
            let sign_index = ((aux32_s >> (7 * group)) & 0x7f) as u8;
            let signs = sign_index | (sign_index.count_ones() as u8 % 2) << 7;
            for (index, magnitude) in grid.iter().enumerate() {
                let sign = if signs & (1 << index) == 0 { 1.0 } else { -1.0 };
                values[value_index] = block_scale * f32::from(*magnitude) * sign;
                value_index += 1;
            }
        }
    }
    values
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_xs(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 74;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ2_XS tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq2_xs_block(block));
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_xs_block(block: &[u8; 74]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let scales = &block[66..];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for (block_index, &scale_byte) in scales.iter().enumerate() {
        let block_scales = [
            scale * (0.5 + f32::from(scale_byte & 0x0f)) * 0.25,
            scale * (0.5 + f32::from(scale_byte >> 4)) * 0.25,
        ];
        for group in 0..4 {
            let q_offset = (block_index * 4 + group) * 2;
            let quantized = u16::from_le_bytes([qs[q_offset], qs[q_offset + 1]]);
            let grid = IQ2_XS_GRID[usize::from(quantized & 0x01ff)].to_le_bytes();
            let sign_index = ((quantized >> 9) & 0x7f) as u8;
            let signs = sign_index | (sign_index.count_ones() as u8 % 2) << 7;
            let group_scale = block_scales[group / 2];
            for (index, magnitude) in grid.iter().enumerate() {
                let sign = if signs & (1 << index) == 0 { 1.0 } else { -1.0 };
                values[value_index] = group_scale * f32::from(*magnitude) * sign;
                value_index += 1;
            }
        }
    }
    values
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_s(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 82;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ2_S tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq2_s_block(block));
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq2_s_block(block: &[u8; 82]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let scales = &block[74..82];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for ib32 in 0..8 {
        let block_scales = [
            scale * (0.5 + f32::from(scales[ib32] & 0x0f)) * 0.25,
            scale * (0.5 + f32::from(scales[ib32] >> 4)) * 0.25,
        ];
        for group in 0..4 {
            let q_offset = ib32 * 4 + group;
            let grid_index =
                usize::from(qs[q_offset]) | usize::from((qh[ib32] >> (2 * group)) & 0x03) << 8;
            let grid = IQ2_S_GRID[grid_index].to_le_bytes();
            let signs = qs[32 + q_offset];
            let group_scale = block_scales[group / 2];
            for (index, magnitude) in grid.iter().enumerate() {
                let sign = if signs & (1 << index) == 0 { 1.0 } else { -1.0 };
                values[value_index] = group_scale * f32::from(*magnitude) * sign;
                value_index += 1;
            }
        }
    }
    values
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq1_s(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 50;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ1_S tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq1_s_block(block));
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq1_s_block(block: &[u8; 50]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..34];
    let qh = &block[34..50];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for ib32 in 0..8 {
        let high = u16::from_le_bytes([qh[2 * ib32], qh[2 * ib32 + 1]]);
        let block_scale = scale * (2.0 * f32::from((high >> 12) & 0x07) + 1.0);
        let delta = if high & 0x8000 != 0 { -0.125 } else { 0.125 };
        for group in 0..4 {
            let grid_index =
                usize::from(qs[ib32 * 4 + group]) | usize::from((high >> (3 * group)) & 0x07) << 8;
            let grid = IQ1_S_GRID[grid_index].to_le_bytes();
            for &magnitude in &grid {
                values[value_index] = block_scale * (f32::from(magnitude.cast_signed()) + delta);
                value_index += 1;
            }
        }
    }
    values
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq3_s(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 110;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ3_S tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq3_s_block(block));
    }
    Ok(values)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq3_s_block(block: &[u8; 110]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..66];
    let qh = &block[66..74];
    let signs = &block[74..106];
    let scales = &block[106..110];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for ib32 in 0..8 {
        let scale_byte = scales[ib32 / 2];
        let block_scale = scale
            * (1.0
                + 2.0
                    * f32::from(if ib32.is_multiple_of(2) {
                        scale_byte & 0x0f
                    } else {
                        scale_byte >> 4
                    }));
        let q_offset = ib32 * 8;
        let sign_offset = ib32 * 4;
        for group in 0..4 {
            let grid1_index = usize::from(qs[q_offset + 2 * group])
                | usize::from((qh[ib32] >> (2 * group)) & 0x01) << 8;
            let grid2_index = usize::from(qs[q_offset + 2 * group + 1])
                | usize::from((qh[ib32] >> (2 * group + 1)) & 0x01) << 8;
            let grid1 = IQ3_S_GRID[grid1_index].to_le_bytes();
            let grid2 = IQ3_S_GRID[grid2_index].to_le_bytes();
            let sign_mask = signs[sign_offset + group];
            for (index, &magnitude) in grid1.iter().enumerate() {
                let sign = if sign_mask & (1 << index) == 0 {
                    1.0
                } else {
                    -1.0
                };
                values[value_index] = block_scale * f32::from(magnitude) * sign;
                value_index += 1;
            }
            for (index, &magnitude) in grid2.iter().enumerate() {
                let sign = if sign_mask & (1 << (index + 4)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                values[value_index] = block_scale * f32::from(magnitude) * sign;
                value_index += 1;
            }
        }
    }
    values
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn decode_iq3_xxs(bytes: &[u8]) -> Result<Vec<f32>, ModelError> {
    const BLOCK_BYTES: usize = 98;
    const BLOCK_VALUES: usize = 256;
    let (blocks, remainder) = bytes.as_chunks::<BLOCK_BYTES>();
    if !remainder.is_empty() {
        return Err(ModelError::Shape(
            "IQ3_XXS tensor byte length is not block aligned".to_owned(),
        ));
    }
    let mut values = Vec::with_capacity(blocks.len() * BLOCK_VALUES);
    for block in blocks {
        values.extend(decode_iq3_xxs_block(block));
    }
    Ok(values)
}

fn decode_iq3_xxs_block(block: &[u8; 98]) -> [f32; 256] {
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..];
    let mut values = [0.0_f32; 256];
    let mut value_index = 0;
    for ib32 in 0..8 {
        let aux_offset = 64 + ib32 * 4;
        let aux32 = u32::from_le_bytes([
            qs[aux_offset],
            qs[aux_offset + 1],
            qs[aux_offset + 2],
            qs[aux_offset + 3],
        ]);
        let block_scale = scale * (0.5 + f32::from((aux32 >> 28) as u8)) * 0.5;
        let q_offset = ib32 * 8;
        for group in 0..4 {
            let signs = sign_mask((aux32 >> (7 * group)) & 0x7f);
            let grid1 = IQ3_XXS_GRID[usize::from(qs[q_offset + 2 * group])].to_le_bytes();
            let grid2 = IQ3_XXS_GRID[usize::from(qs[q_offset + 2 * group + 1])].to_le_bytes();
            for index in 0..4 {
                let sign = if signs & (1 << index) == 0 { 1.0 } else { -1.0 };
                values[value_index] = block_scale * f32::from(grid1[index]) * sign;
                let sign = if signs & (1 << (index + 4)) == 0 {
                    1.0
                } else {
                    -1.0
                };
                values[value_index + 4] = block_scale * f32::from(grid2[index]) * sign;
                value_index += 1;
            }
            value_index += 4;
        }
    }
    values
}

fn sign_mask(index: u32) -> u8 {
    let index = u8::try_from(index).expect("IQ3_XXS sign index is 7-bit");
    if index.count_ones().is_multiple_of(2) {
        index
    } else {
        index | 0x80
    }
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

fn iq4_nl_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 18;
    let block_index = index / 32;
    let offset = index % 32;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ4_NL index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ4_NL block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("IQ4_NL block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let packed = block[2 + offset % 16];
    let quantized = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(IQ4_NL_VALUES[usize::from(quantized)]))
}

fn iq4_xs_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 136;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ4_XS index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ4_XS block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 136]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ4_XS block is outside the tensor".to_owned()))?;
    let sub_block = offset / 32;
    let scale_index = sub_block / 2;
    let scale_shift = 4 * (sub_block % 2);
    let scales_high = u16::from_le_bytes([block[2], block[3]]);
    let quantized_scale = ((block[4 + scale_index] >> scale_shift) & 0x0f)
        | ((((scales_high >> (2 * sub_block)) & 0x03) as u8) << 4);
    let scale =
        f16_to_f32(u16::from_le_bytes([block[0], block[1]])) * (f32::from(quantized_scale) - 32.0);
    let packed = block[8 + sub_block * 16 + offset % 16];
    let quantized = if offset % 32 < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(IQ4_NL_VALUES[usize::from(quantized)]))
}

fn mxfp4_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 17;
    let block_index = index / 32;
    let offset = index % 32;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("MXFP4 index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("MXFP4 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("MXFP4 block is outside the tensor".to_owned()))?;
    let scale = e8m0_to_f32_half(block[0]);
    let packed = block[1 + offset % 16];
    let quantized = if offset < 16 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(MXFP4_VALUES[usize::from(quantized)]))
}

fn nvfp4_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 36;
    let block_index = index / 64;
    let offset = index % 64;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("NVFP4 index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("NVFP4 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("NVFP4 block is outside the tensor".to_owned()))?;
    let sub_block = offset / 16;
    let scale = ue4m3_to_f32_half(block[sub_block]);
    let packed = block[4 + sub_block * 8 + offset % 8];
    let quantized = if offset % 16 < 8 {
        packed & 0x0f
    } else {
        packed >> 4
    };
    Ok(scale * f32::from(MXFP4_VALUES[usize::from(quantized)]))
}

fn tq1_0_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 54;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("TQ1_0 index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("TQ1_0 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("TQ1_0 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[52], block[53]]));
    let (packed, power) = if offset < 160 {
        (block[offset % 32], offset / 32)
    } else if offset < 240 {
        let local = offset - 160;
        (block[32 + local % 16], local / 16)
    } else {
        let local = offset - 240;
        (block[48 + local % 4], local / 4)
    };
    Ok(scale * tq1_digit(packed, power))
}

fn tq2_0_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 66;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("TQ2_0 index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("TQ2_0 block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("TQ2_0 block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[64], block[65]]));
    let chunk = offset / 128;
    let local = offset % 128;
    let shift = (local / 32) * 2;
    let packed = block[chunk * 32 + local % 32];
    let quantized = (packed >> shift) & 0x03;
    Ok(scale * (f32::from(quantized) - 1.0))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq2_xxs_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 66;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_XXS index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_XXS block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .ok_or_else(|| ModelError::Shape("IQ2_XXS block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let qs = &block[2..];
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let q_offset = ib32 * 8;
    let aux32_g = u32::from(u16::from_le_bytes([qs[q_offset], qs[q_offset + 1]]))
        | (u32::from(u16::from_le_bytes([qs[q_offset + 2], qs[q_offset + 3]])) << 16);
    let aux32_s = u32::from(u16::from_le_bytes([qs[q_offset + 4], qs[q_offset + 5]]))
        | (u32::from(u16::from_le_bytes([qs[q_offset + 6], qs[q_offset + 7]])) << 16);
    let block_scale = scale * (0.5 + (aux32_s >> 28) as f32) * 0.25;
    let grid = IQ2_XXS_GRID[((aux32_g >> (8 * group)) & 0xff) as usize].to_le_bytes();
    let sign_index = ((aux32_s >> (7 * group)) & 0x7f) as u8;
    let signs = sign_index | (sign_index.count_ones() as u8 % 2) << 7;
    let sign = if signs & (1 << index_in_group) == 0 {
        1.0
    } else {
        -1.0
    };
    Ok(block_scale * f32::from(grid[index_in_group]) * sign)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq2_xs_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 74;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_XS index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_XS block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 74]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ2_XS block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let scale_byte = block[66 + ib32];
    let group_scale = scale
        * (0.5
            + f32::from(if group.is_multiple_of(2) {
                scale_byte & 0x0f
            } else {
                scale_byte >> 4
            }))
        * 0.25;
    let q_offset = (ib32 * 4 + group) * 2;
    let quantized = u16::from_le_bytes([block[2 + q_offset], block[3 + q_offset]]);
    let grid = IQ2_XS_GRID[usize::from(quantized & 0x01ff)].to_le_bytes();
    let sign_index = ((quantized >> 9) & 0x7f) as u8;
    let signs = sign_index | (sign_index.count_ones() as u8 % 2) << 7;
    let sign = if signs & (1 << index_in_group) == 0 {
        1.0
    } else {
        -1.0
    };
    Ok(group_scale * f32::from(grid[index_in_group]) * sign)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq2_s_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 82;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_S index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ2_S block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 82]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ2_S block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let scale_byte = block[74 + ib32];
    let group_scale = scale
        * (0.5
            + f32::from(if group.is_multiple_of(2) {
                scale_byte & 0x0f
            } else {
                scale_byte >> 4
            }))
        * 0.25;
    let q_offset = ib32 * 4 + group;
    let grid_index = usize::from(block[2 + q_offset])
        | usize::from((block[66 + ib32] >> (2 * group)) & 0x03) << 8;
    let grid = IQ2_S_GRID[grid_index].to_le_bytes();
    let signs = block[2 + 32 + q_offset];
    let sign = if signs & (1 << index_in_group) == 0 {
        1.0
    } else {
        -1.0
    };
    Ok(group_scale * f32::from(grid[index_in_group]) * sign)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq1_s_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 50;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ1_S index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ1_S block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 50]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ1_S block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let high = u16::from_le_bytes([block[34 + 2 * ib32], block[35 + 2 * ib32]]);
    let block_scale = scale * (2.0 * f32::from((high >> 12) & 0x07) + 1.0);
    let delta = if high & 0x8000 != 0 { -0.125 } else { 0.125 };
    let grid_index =
        usize::from(block[2 + ib32 * 4 + group]) | usize::from((high >> (3 * group)) & 0x07) << 8;
    let grid = IQ1_S_GRID[grid_index].to_le_bytes();
    Ok(block_scale * (f32::from(grid[index_in_group].cast_signed()) + delta))
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq3_s_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 110;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ3_S index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ3_S block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 110]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ3_S block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let scale_byte = block[106 + ib32 / 2];
    let block_scale = scale
        * (1.0
            + 2.0
                * f32::from(if ib32.is_multiple_of(2) {
                    scale_byte & 0x0f
                } else {
                    scale_byte >> 4
                }));
    let q_offset = 2 + ib32 * 8 + group * 2;
    let qh = block[66 + ib32];
    let grid_index = usize::from(block[q_offset]) | usize::from((qh >> (2 * group)) & 0x01) << 8;
    let grid = if index_in_group < 4 {
        IQ3_S_GRID[grid_index].to_le_bytes()
    } else {
        let grid_index =
            usize::from(block[q_offset + 1]) | usize::from((qh >> (2 * group + 1)) & 0x01) << 8;
        IQ3_S_GRID[grid_index].to_le_bytes()
    };
    let sign_mask = block[74 + ib32 * 4 + group];
    let sign = if sign_mask & (1 << index_in_group) == 0 {
        1.0
    } else {
        -1.0
    };
    let magnitude_index = index_in_group % 4;
    Ok(block_scale * f32::from(grid[magnitude_index]) * sign)
}

#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
fn iq3_xxs_value_at(bytes: &[u8], index: usize) -> Result<f32, ModelError> {
    const BLOCK_BYTES: usize = 98;
    let block_index = index / 256;
    let offset = index % 256;
    let start = block_index
        .checked_mul(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ3_XXS index overflows".to_owned()))?;
    let end = start
        .checked_add(BLOCK_BYTES)
        .ok_or_else(|| ModelError::Shape("IQ3_XXS block range overflows".to_owned()))?;
    let block = bytes
        .get(start..end)
        .and_then(|slice| <&[u8; 98]>::try_from(slice).ok())
        .ok_or_else(|| ModelError::Shape("IQ3_XXS block is outside the tensor".to_owned()))?;
    let scale = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
    let ib32 = offset / 32;
    let group = (offset % 32) / 8;
    let index_in_group = offset % 8;
    let aux_offset = 66 + ib32 * 4;
    let aux32 = u32::from_le_bytes([
        block[aux_offset],
        block[aux_offset + 1],
        block[aux_offset + 2],
        block[aux_offset + 3],
    ]);
    let block_scale = scale * (0.5 + (aux32 >> 28) as f32) * 0.5;
    let signs = sign_mask((aux32 >> (7 * group)) & 0x7f);
    let q_offset = 2 + ib32 * 8 + group * 2 + index_in_group / 4;
    let grid = IQ3_XXS_GRID[usize::from(block[q_offset])].to_le_bytes();
    let magnitude = grid[index_in_group % 4];
    let sign = if signs & (1 << index_in_group) == 0 {
        1.0
    } else {
        -1.0
    };
    Ok(block_scale * f32::from(magnitude) * sign)
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
    fn materializes_iq2_xxs_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 64));
        let path = write_fixture(&fixture(16, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 16);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq2_xs_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 72));
        let path = write_fixture(&fixture(17, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 17);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq2_s_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 80));
        let path = write_fixture(&fixture(22, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 22);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq1_s_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 48));
        let path = write_fixture(&fixture(19, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[-0.875; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 19);
        assert_eq!(matrix.column(0).unwrap(), vec![-0.875; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq3_xxs_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 96));
        let path = write_fixture(&fixture(18, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 18);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq3_s_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 108));
        let path = write_fixture(&fixture(21, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 21);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq4_nl_tensor() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0x88, 16));
        let path = write_fixture(&fixture(20, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(&values.data()[..16], &[1.0; 16]);
        assert_eq!(&values.data()[16..], &[1.0; 16]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 20);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 32]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_iq4_xs_tensor() {
        let mut encoded = vec![0x00, 0x3c, 0xaa, 0xaa, 0x11, 0x11, 0x11, 0x11];
        encoded.extend(std::iter::repeat_n(0x88, 128));
        let path = write_fixture(&fixture(23, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 256]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 23);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 256]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_mxfp4_tensor() {
        let mut encoded = vec![128_u8];
        encoded.extend(std::iter::repeat_n(0x11, 16));
        let path = write_fixture(&fixture(39, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 32]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 39);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 32]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_nvfp4_tensor() {
        let mut encoded = vec![0x40; 4];
        encoded.extend(std::iter::repeat_n(0x11, 32));
        let path = write_fixture(&fixture(40, &[64, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let values = model.load_f32("probe.tensor").unwrap();
        assert_eq!(values.data(), &[1.0; 64]);
        let matrix = model.load_quantized("probe.tensor").unwrap();
        assert_eq!(matrix.value_type().raw(), 40);
        assert_eq!(matrix.column(0).unwrap(), vec![1.0; 64]);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn materializes_tq1_and_tq2_tensors() {
        let mut tq1 = vec![0_u8; 54];
        tq1[52..54].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let tq1_path = write_fixture(&fixture(34, &[256, 1], &tq1));
        let tq1_model = GgufModel::open(&tq1_path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            tq1_model.load_f32("probe.tensor").unwrap().data(),
            &[-1.0; 256]
        );
        assert_eq!(
            tq1_model
                .load_quantized("probe.tensor")
                .unwrap()
                .column(0)
                .unwrap(),
            vec![-1.0; 256]
        );
        fs::remove_file(tq1_path).unwrap();

        let mut tq2 = vec![0xaa_u8; 66];
        tq2[64..66].copy_from_slice(&0x3c00_u16.to_le_bytes());
        let tq2_path = write_fixture(&fixture(35, &[256, 1], &tq2));
        let tq2_model = GgufModel::open(&tq2_path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert_eq!(
            tq2_model.load_f32("probe.tensor").unwrap().data(),
            &[1.0; 256]
        );
        assert_eq!(
            tq2_model
                .load_quantized("probe.tensor")
                .unwrap()
                .column(0)
                .unwrap(),
            vec![1.0; 256]
        );
        fs::remove_file(tq2_path).unwrap();
    }

    #[test]
    fn converts_iq2_xxs_matrix_directly_to_mlx_affine_layout() {
        let mut encoded = vec![0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0, 64));
        let path = write_fixture(&fixture(16, &[256, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        let quantized = model.load_affine_quantized("probe.tensor", 64, 4).unwrap();
        assert_eq!(quantized.rows(), 1);
        assert_eq!(quantized.columns(), 256);
        assert_eq!(quantized.group_size(), 64);
        assert_eq!(quantized.bits(), 4);
        assert_eq!(quantized.scales(), &[-1e-7; 4]);
        assert_eq!(quantized.biases(), &[1.0; 4]);
        let mut observed = None;
        model
            .for_each_tensor(&["probe.tensor"], 64, 4, |_, tensor| {
                observed = Some(tensor);
                Ok::<(), ModelError>(())
            })
            .unwrap();
        assert!(matches!(observed, Some(LoadedTensor::AffineQuantized(_))));
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
        let path = write_fixture(&fixture(29, &[256], &[0; 128]));
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
