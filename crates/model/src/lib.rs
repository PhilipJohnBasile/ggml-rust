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

    /// Computes a row-vector matrix product directly from a quantized GGUF
    /// matrix without materializing the matrix as F32 values.
    ///
    /// The input must have the length of the descriptor's first dimension.
    /// GGML stores matrix values in column-major tensor order, so each output
    /// column is decoded and dotted against the input row. `Q4_0` and
    /// `Q8_0` are supported initially because their fixed 32-value blocks can
    /// be addressed without allocating a temporary tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not a rank-2 `Q4_0` or `Q8_0`
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
        if !rows.is_multiple_of(32) {
            return Err(ModelError::Shape(
                "quantized matmul rows must be a multiple of 32".to_owned(),
            ));
        }
        let block_bytes = match descriptor.value_type.raw() {
            2 => 18,
            8 => 34,
            _ => {
                return Err(ModelError::UnsupportedTensorType {
                    name: descriptor.name.clone(),
                    value_type: descriptor.value_type,
                });
            }
        };
        let expected_bytes = rows
            .checked_mul(columns)
            .and_then(|elements| elements.checked_div(32))
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
            for column in 0..columns {
                let mut sum = 0.0_f32;
                for (row, &value) in input.iter().enumerate() {
                    let flat_index = column
                        .checked_mul(rows)
                        .and_then(|base| base.checked_add(row))
                        .ok_or_else(|| {
                            ModelError::Shape("quantized matrix index overflows".to_owned())
                        })?;
                    let weight = match descriptor.value_type.raw() {
                        2 => q4_0_value_at(tensor_bytes, flat_index),
                        8 => q8_0_value_at(tensor_bytes, flat_index),
                        _ => unreachable!("value type validated above"),
                    }?;
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
    fn rejects_unsupported_quantized_matmul_inputs() {
        let mut encoded = vec![0x00, 0x3c, 0x00, 0x3c];
        encoded.extend(std::iter::repeat_n(0_u8, 16));
        let path = write_fixture(&fixture(3, &[32, 1], &encoded));
        let model = GgufModel::open(&path, DEFAULT_MODEL_BYTE_LIMIT).unwrap();
        assert!(matches!(
            model.matmul_f32_quantized("probe.tensor", &[1.0; 32]),
            Err(ModelError::UnsupportedTensorType { .. })
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
