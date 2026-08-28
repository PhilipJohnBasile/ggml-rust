#![forbid(unsafe_code)]

mod error;
mod parser;
mod types;

pub use error::{Error, ErrorKind, ReadError};
pub use parser::{DEFAULT_MAX_HEADER_BYTES, Gguf, GgufHeader, HeaderLimits, Limits};
pub use types::{
    MetadataArray, MetadataEntry, MetadataType, MetadataValue, ScalarValue, TensorInfo, TensorType,
};
