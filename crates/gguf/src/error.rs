use std::fmt;
use std::io;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub offset: usize,
    pub kind: ErrorKind,
}

impl Error {
    pub(crate) const fn new(offset: usize, kind: ErrorKind) -> Self {
        Self { offset, kind }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ErrorKind {
    UnexpectedEof {
        needed: usize,
        remaining: usize,
    },
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    EndiannessMismatch(u32),
    LimitExceeded {
        field: &'static str,
        value: u64,
        limit: u64,
    },
    IntegerOverflow(&'static str),
    AllocationFailed(&'static str),
    InvalidUtf8(&'static str),
    EmptyMetadataKey,
    MetadataKeyTooLong(usize),
    InvalidMetadataKey,
    DuplicateMetadataKey(String),
    InvalidMetadataType(u32),
    NestedMetadataArray,
    InvalidBoolean(u8),
    InvalidAlignment(u32),
    DuplicateTensorName(String),
    TensorNameTooLong(usize),
    TensorNameContainsNul,
    TooManyDimensions(u32),
    InvalidDimension(u64),
    InvalidTensorType(u32),
    MisalignedTensorOffset {
        offset: u64,
        alignment: u32,
    },
    TensorRowNotDivisible {
        elements_per_row: u64,
        block_size: u64,
    },
    UnexpectedTensorOffset {
        expected: u64,
        actual: u64,
    },
    TensorDataTruncated {
        expected: u64,
        actual: u64,
    },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GGUF parse error at byte {}: ", self.offset)?;
        match &self.kind {
            ErrorKind::UnexpectedEof { needed, remaining } => {
                write!(f, "needed {needed} bytes but only {remaining} remain")
            }
            ErrorKind::InvalidMagic(magic) => write!(f, "invalid magic {magic:?}"),
            ErrorKind::UnsupportedVersion(version) => write!(f, "unsupported version {version}"),
            ErrorKind::EndiannessMismatch(version) => {
                write!(
                    f,
                    "version {version:#010x} indicates an endianness mismatch"
                )
            }
            ErrorKind::LimitExceeded {
                field,
                value,
                limit,
            } => write!(f, "{field} value {value} exceeds limit {limit}"),
            ErrorKind::IntegerOverflow(field) => {
                write!(f, "{field} overflows the host integer range")
            }
            ErrorKind::AllocationFailed(field) => {
                write!(f, "could not allocate storage for {field}")
            }
            ErrorKind::InvalidUtf8(field) => write!(f, "{field} is not valid UTF-8"),
            ErrorKind::EmptyMetadataKey => f.write_str("metadata key is empty"),
            ErrorKind::MetadataKeyTooLong(length) => {
                write!(
                    f,
                    "metadata key length {length} exceeds the 65535-byte limit"
                )
            }
            ErrorKind::InvalidMetadataKey => f.write_str(
                "metadata key must contain non-empty lower-case ASCII, digit, or underscore segments separated by periods",
            ),
            ErrorKind::DuplicateMetadataKey(key) => write!(f, "duplicate metadata key {key:?}"),
            ErrorKind::InvalidMetadataType(value_type) => {
                write!(f, "invalid metadata type {value_type}")
            }
            ErrorKind::NestedMetadataArray => {
                f.write_str("nested metadata arrays are not supported by ggml")
            }
            ErrorKind::InvalidBoolean(value) => write!(f, "invalid boolean byte {value}"),
            ErrorKind::InvalidAlignment(alignment) => {
                write!(f, "alignment {alignment} is not a nonzero power of two")
            }
            ErrorKind::DuplicateTensorName(name) => write!(f, "duplicate tensor name {name:?}"),
            ErrorKind::TensorNameTooLong(length) => {
                write!(f, "tensor name length {length} exceeds the 63-byte limit")
            }
            ErrorKind::TensorNameContainsNul => {
                f.write_str("tensor name contains an interior NUL byte")
            }
            ErrorKind::TooManyDimensions(count) => {
                write!(f, "tensor has {count} dimensions, maximum is 4")
            }
            ErrorKind::InvalidDimension(value) => {
                write!(f, "tensor dimension {value} exceeds i64::MAX")
            }
            ErrorKind::InvalidTensorType(value_type) => {
                write!(f, "invalid tensor type {value_type}")
            }
            ErrorKind::MisalignedTensorOffset { offset, alignment } => {
                write!(
                    f,
                    "tensor offset {offset} is not aligned to {alignment} bytes"
                )
            }
            ErrorKind::TensorRowNotDivisible {
                elements_per_row,
                block_size,
            } => write!(
                f,
                "tensor row has {elements_per_row} elements, not a multiple of block size {block_size}"
            ),
            ErrorKind::UnexpectedTensorOffset { expected, actual } => {
                write!(
                    f,
                    "tensor offset {actual} does not match expected offset {expected}"
                )
            }
            ErrorKind::TensorDataTruncated { expected, actual } => write!(
                f,
                "tensor data section needs {expected} bytes but only {actual} are present"
            ),
        }
    }
}

impl std::error::Error for Error {}

#[derive(Debug)]
pub enum ReadError {
    /// The underlying reader failed during a named operation.
    Io {
        operation: &'static str,
        offset: Option<u64>,
        source: io::Error,
    },
    /// The GGUF structure or declared ranges are invalid.
    Parse(Error),
    /// The reader's reported file length changed during validation.
    FileLengthChanged { expected: u64, actual: u64 },
}

impl fmt::Display for ReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                operation,
                offset: Some(offset),
                source,
            } => write!(
                f,
                "GGUF I/O error while {operation} at byte {offset}: {source}"
            ),
            Self::Io {
                operation,
                offset: None,
                source,
            } => write!(f, "GGUF I/O error while {operation}: {source}"),
            Self::Parse(error) => error.fmt(f),
            Self::FileLengthChanged { expected, actual } => write!(
                f,
                "GGUF file length changed while reading: expected {expected} bytes, found {actual}"
            ),
        }
    }
}

impl std::error::Error for ReadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Parse(error) => Some(error),
            Self::FileLengthChanged { .. } => None,
        }
    }
}

impl ReadError {
    pub(crate) fn io(operation: &'static str, offset: Option<u64>, source: io::Error) -> Self {
        Self::Io {
            operation,
            offset,
            source,
        }
    }
}

impl From<Error> for ReadError {
    fn from(value: Error) -> Self {
        Self::Parse(value)
    }
}
