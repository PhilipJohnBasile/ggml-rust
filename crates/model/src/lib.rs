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

    /// Materializes one F32 tensor into the checked CPU tensor engine.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is missing or non-F32, the model bytes
    /// changed, the tensor range is invalid, or the shape does not match the
    /// decoded F32 values.
    pub fn load_f32(&self, name: &str) -> Result<Tensor, ModelError> {
        let descriptor = self
            .tensor(name)
            .ok_or_else(|| ModelError::TensorNotFound(name.to_owned()))?;
        if descriptor.value_type.raw() != 0 {
            return Err(ModelError::UnsupportedTensorType {
                name: name.to_owned(),
                value_type: descriptor.value_type,
            });
        }
        let mapped = map_model(&self.path, self.max_file_bytes)?;
        let bytes = mapped.as_bytes();
        if digest_bytes(bytes) != self.digest {
            return Err(ModelError::ContentChanged);
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
        let (chunks, remainder) = tensor_bytes.as_chunks::<4>();
        if !remainder.is_empty() {
            return Err(ModelError::Shape(
                "F32 tensor byte length is not aligned".to_owned(),
            ));
        }
        let values = chunks
            .iter()
            .map(|chunk| f32::from_le_bytes(*chunk))
            .collect::<Vec<_>>();
        Tensor::from_data(descriptor.shape.clone(), values)
            .map_err(|error| ModelError::Shape(error.to_string()))
    }
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

    fn f32_fixture(value_type: u32, tensor_bytes: &[u8]) -> Vec<u8> {
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
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
        bytes.extend_from_slice(&2_u64.to_le_bytes());
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
        assert_eq!(model.tensors().len(), 1);
        assert_eq!(model.tensor("probe.tensor").unwrap().shape(), &[2, 2]);
        assert_eq!(model.load_f32("probe.tensor").unwrap().data(), &values);
        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_f32_tensor_materialization() {
        let path = write_fixture(&f32_fixture(1, &[0; 8]));
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
