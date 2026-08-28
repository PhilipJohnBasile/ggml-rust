use std::borrow::Cow;
use std::collections::HashSet;
use std::io::{Read, Seek, SeekFrom};
use std::marker::PhantomData;
use std::ops::Range;

use crate::error::{Error, ErrorKind, ReadError};
use crate::types::{
    MetadataArray, MetadataEntry, MetadataType, MetadataValue, ScalarValue, TensorInfo, TensorType,
};

const MAGIC: [u8; 4] = *b"GGUF";
const CURRENT_VERSION: u32 = 3;
const DEFAULT_ALIGNMENT: u32 = 32;
const MAX_METADATA_KEY_BYTES: usize = u16::MAX as usize;
const MAX_TENSOR_NAME_BYTES: usize = 63;
const MIN_METADATA_ENTRY_BYTES: usize = size_of::<u64>() + size_of::<u32>();
const MIN_TENSOR_INFO_BYTES: usize =
    size_of::<u64>() + size_of::<u32>() + size_of::<u32>() + size_of::<u64>();
/// Default seekable-reader limit for GGUF metadata and tensor-table bytes.
pub const DEFAULT_MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_string_bytes: u64,
    pub max_array_elements: u64,
    pub max_metadata_entries: u64,
    pub max_tensors: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_string_bytes: 1024 * 1024 * 1024,
            max_array_elements: 16 * 1024 * 1024,
            max_metadata_entries: 1024 * 1024,
            max_tensors: 1024 * 1024,
        }
    }
}

impl Limits {
    #[must_use]
    pub const fn upstream_compatible() -> Self {
        Self {
            max_string_bytes: 1024 * 1024 * 1024,
            max_array_elements: 1024 * 1024 * 1024,
            max_metadata_entries: u64::MAX,
            max_tensors: u64::MAX,
        }
    }
}

/// Limits used when validating a GGUF from a seekable reader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeaderLimits {
    pub format: Limits,
    pub max_header_bytes: u64,
}

impl Default for HeaderLimits {
    fn default() -> Self {
        Self {
            format: Limits::default(),
            max_header_bytes: DEFAULT_MAX_HEADER_BYTES,
        }
    }
}

/// Validated GGUF layout returned by the seekable-reader parser.
///
/// The parser retains no metadata values. It does not read alignment padding
/// or tensor data.
///
/// File length is checked before and after validation. A same-length rewrite
/// cannot be detected, so callers that need a stable artifact snapshot must
/// prevent concurrent writers with external synchronization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GgufHeader {
    version: u32,
    alignment: u32,
    header_size: u64,
    data_offset: u64,
    data_size: u64,
    file_size: u64,
    metadata_count: u64,
    tensor_count: u64,
}

impl GgufHeader {
    /// Validates a complete GGUF v2 or v3 from a seekable reader.
    ///
    /// The reader is treated as a whole file. This method seeks to byte zero,
    /// reads only the header and tables, validates tensor ranges against the
    /// file length, and leaves the reader at [`Self::data_offset`].
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failures, concurrent file-length changes,
    /// malformed input, unsupported input, or a safety-limit violation.
    pub fn from_reader<R: Read + Seek>(reader: &mut R) -> Result<Self, ReadError> {
        Self::from_reader_with_limits(reader, HeaderLimits::default())
    }

    /// Validates a complete GGUF with caller-provided format and header limits.
    ///
    /// The reader is treated as a whole file. This method seeks to byte zero,
    /// reads only the header and tables, validates tensor ranges against the
    /// file length, and leaves the reader at [`Self::data_offset`].
    ///
    /// # Errors
    ///
    /// Returns an error for I/O failures, concurrent file-length changes,
    /// malformed input, unsupported input, or a supplied-limit violation.
    pub fn from_reader_with_limits<R: Read + Seek>(
        reader: &mut R,
        limits: HeaderLimits,
    ) -> Result<Self, ReadError> {
        let file_size = reader
            .seek(SeekFrom::End(0))
            .map_err(|error| ReadError::io("determining the initial file length", None, error))?;
        reader
            .seek(SeekFrom::Start(0))
            .map_err(|error| ReadError::io("seeking to the file start", Some(0), error))?;

        let parsed = Parser::<'static, _>::new(
            StreamReader::new(reader, file_size, limits.max_header_bytes),
            limits.format,
            false,
        )
        .parse()?;

        let observed_file_size = reader
            .seek(SeekFrom::End(0))
            .map_err(|error| ReadError::io("rechecking the file length", None, error))?;
        if observed_file_size != file_size {
            return Err(ReadError::FileLengthChanged {
                expected: file_size,
                actual: observed_file_size,
            });
        }
        reader
            .seek(SeekFrom::Start(parsed.data_offset))
            .map_err(|error| {
                ReadError::io(
                    "positioning at tensor data",
                    Some(parsed.data_offset),
                    error,
                )
            })?;

        Ok(Self {
            version: parsed.version,
            alignment: parsed.alignment,
            header_size: parsed.header_size,
            data_offset: parsed.data_offset,
            data_size: parsed.data_size,
            file_size,
            metadata_count: parsed.metadata_count,
            tensor_count: parsed.tensor_count,
        })
    }

    #[must_use]
    pub const fn version(self) -> u32 {
        self.version
    }

    #[must_use]
    pub const fn alignment(self) -> u32 {
        self.alignment
    }

    /// Number of bytes occupied by the header and tables, before alignment padding.
    #[must_use]
    pub const fn header_size(self) -> u64 {
        self.header_size
    }

    #[must_use]
    pub const fn data_offset(self) -> u64 {
        self.data_offset
    }

    #[must_use]
    pub const fn data_size(self) -> u64 {
        self.data_size
    }

    #[must_use]
    pub const fn file_size(self) -> u64 {
        self.file_size
    }

    #[must_use]
    pub const fn metadata_count(self) -> u64 {
        self.metadata_count
    }

    #[must_use]
    pub const fn tensor_count(self) -> u64 {
        self.tensor_count
    }

    #[must_use]
    pub const fn tensor_data_range(self) -> Range<u64> {
        self.data_offset..self.data_offset + self.data_size
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Gguf<'a> {
    version: u32,
    alignment: u32,
    data_offset: usize,
    data_size: usize,
    metadata: Vec<MetadataEntry<'a>>,
    tensors: Vec<TensorInfo<'a>>,
    source: &'a [u8],
}

impl<'a> Gguf<'a> {
    /// Parses a GGUF v2 or v3 byte slice with bounded default limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is truncated, malformed, unsupported, or exceeds a safety limit.
    pub fn from_bytes(source: &'a [u8]) -> Result<Self, Error> {
        Self::from_bytes_with_limits(source, Limits::default())
    }

    /// Parses a GGUF v2 or v3 byte slice with caller-provided limits.
    ///
    /// # Errors
    ///
    /// Returns an error when the input is truncated, malformed, unsupported, or exceeds a supplied limit.
    pub fn from_bytes_with_limits(source: &'a [u8], limits: Limits) -> Result<Self, Error> {
        let parsed = Parser::<'a, _>::new(SliceReader::new(source), limits, true).parse()?;
        parsed.into_borrowed(source)
    }

    #[must_use]
    pub const fn version(&self) -> u32 {
        self.version
    }

    #[must_use]
    pub const fn alignment(&self) -> u32 {
        self.alignment
    }

    #[must_use]
    pub const fn data_offset(&self) -> usize {
        self.data_offset
    }

    #[must_use]
    pub fn data(&self) -> &'a [u8] {
        &self.source[self.data_offset..self.data_offset + self.data_size]
    }

    #[must_use]
    pub const fn data_size(&self) -> usize {
        self.data_size
    }

    #[must_use]
    pub fn metadata(&self) -> &[MetadataEntry<'a>] {
        &self.metadata
    }

    #[must_use]
    pub fn tensors(&self) -> &[TensorInfo<'a>] {
        &self.tensors
    }

    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&MetadataValue<'a>> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    /// Returns the absolute file range occupied by a tensor, excluding alignment padding.
    #[must_use]
    pub fn tensor_data_range(&self, index: usize) -> Option<Range<usize>> {
        let tensor = self.tensors.get(index)?;
        let relative_offset = usize::try_from(tensor.offset).ok()?;
        let byte_len = usize::try_from(tensor.byte_len).ok()?;
        let start = self.data_offset.checked_add(relative_offset)?;
        let end = start.checked_add(byte_len)?;
        (end <= self.data_offset.checked_add(self.data_size)?).then_some(start..end)
    }

    /// Returns a tensor's encoded bytes, excluding alignment padding.
    #[must_use]
    pub fn tensor_data(&self, index: usize) -> Option<&'a [u8]> {
        self.source.get(self.tensor_data_range(index)?)
    }
}

struct Parser<'a, R> {
    reader: R,
    limits: Limits,
    retain_output: bool,
    marker: PhantomData<&'a ()>,
}

impl<'a, R: ParseReader<'a>> Parser<'a, R> {
    const fn new(reader: R, limits: Limits, retain_output: bool) -> Self {
        Self {
            reader,
            limits,
            retain_output,
            marker: PhantomData,
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse(mut self) -> Result<RawGguf<'a>, R::Error> {
        let magic = self.reader.read_array::<4>()?;
        if magic != MAGIC {
            return Err(self.reader.error(ErrorKind::InvalidMagic(magic)));
        }

        let version = self.reader.read_u32()?;
        if (2..=CURRENT_VERSION).contains(&version.swap_bytes()) {
            return Err(self.reader.error(ErrorKind::EndiannessMismatch(version)));
        }
        if !(2..=CURRENT_VERSION).contains(&version) {
            return Err(self.reader.error(ErrorKind::UnsupportedVersion(version)));
        }

        let tensor_count = self.read_count("tensor count", self.limits.max_tensors)?;
        let metadata_count = self.read_count("metadata count", self.limits.max_metadata_entries)?;
        let tensor_count_u64 = u64::try_from(tensor_count).map_err(|_| {
            self.reader
                .error(ErrorKind::IntegerOverflow("tensor count"))
        })?;
        let metadata_count_u64 = u64::try_from(metadata_count).map_err(|_| {
            self.reader
                .error(ErrorKind::IntegerOverflow("metadata count"))
        })?;

        self.reader
            .ensure_items(metadata_count, MIN_METADATA_ENTRY_BYTES, "metadata table")?;

        let mut metadata = Vec::new();
        if self.retain_output {
            metadata.try_reserve_exact(metadata_count).map_err(|_| {
                self.reader
                    .error(ErrorKind::AllocationFailed("metadata entries"))
            })?;
        }
        let mut metadata_keys = HashSet::<Cow<'a, str>>::new();
        metadata_keys.try_reserve(metadata_count).map_err(|_| {
            self.reader
                .error(ErrorKind::AllocationFailed("metadata key index"))
        })?;

        let mut alignment = DEFAULT_ALIGNMENT;
        for _ in 0..metadata_count {
            let key = self.read_string("metadata key")?;
            if key.is_empty() {
                return Err(self.reader.error(ErrorKind::EmptyMetadataKey));
            }
            if key.len() > MAX_METADATA_KEY_BYTES {
                return Err(self.reader.error(ErrorKind::MetadataKeyTooLong(key.len())));
            }
            if !is_valid_metadata_key(&key) {
                return Err(self.reader.error(ErrorKind::InvalidMetadataKey));
            }
            if metadata_keys.contains(key.as_ref()) {
                let duplicate = self.own_error_string(key, "duplicate metadata key error")?;
                return Err(self
                    .reader
                    .error(ErrorKind::DuplicateMetadataKey(duplicate)));
            }
            let alignment_key = key.as_ref() == "general.alignment";
            let retained_key = if self.retain_output {
                metadata_keys.insert(key.clone());
                Some(key)
            } else {
                metadata_keys.insert(key);
                None
            };
            let value = self.read_metadata_value()?;
            if alignment_key {
                alignment = match value {
                    RawMetadataValue::Scalar(RawScalarValue::U32(value)) => value,
                    ref other => {
                        return Err(self
                            .reader
                            .error(ErrorKind::InvalidMetadataType(other.value_type() as u32)));
                    }
                };
            }
            if let Some(key) = retained_key {
                metadata.push(RawMetadataEntry { key, value });
            }
        }
        if !alignment.is_power_of_two() {
            return Err(self.reader.error(ErrorKind::InvalidAlignment(alignment)));
        }
        drop(metadata_keys);

        self.reader
            .ensure_items(tensor_count, MIN_TENSOR_INFO_BYTES, "tensor table")?;

        let mut tensors = Vec::new();
        if self.retain_output {
            tensors.try_reserve_exact(tensor_count).map_err(|_| {
                self.reader
                    .error(ErrorKind::AllocationFailed("tensor table"))
            })?;
        }
        let mut tensor_names = HashSet::<Cow<'a, str>>::new();
        tensor_names.try_reserve(tensor_count).map_err(|_| {
            self.reader
                .error(ErrorKind::AllocationFailed("tensor name index"))
        })?;

        let mut expected_offset = 0_u64;
        for _ in 0..tensor_count {
            let name = self.read_string("tensor name")?;
            if name.len() > MAX_TENSOR_NAME_BYTES {
                return Err(self.reader.error(ErrorKind::TensorNameTooLong(name.len())));
            }
            if name.as_bytes().contains(&0) {
                return Err(self.reader.error(ErrorKind::TensorNameContainsNul));
            }
            if tensor_names.contains(name.as_ref()) {
                let duplicate = self.own_error_string(name, "duplicate tensor name error")?;
                return Err(self.reader.error(ErrorKind::DuplicateTensorName(duplicate)));
            }
            let retained_name = if self.retain_output {
                tensor_names.insert(name.clone());
                Some(name)
            } else {
                tensor_names.insert(name);
                None
            };
            let dimension_count = self.reader.read_u32()?;
            if dimension_count > 4 {
                return Err(self
                    .reader
                    .error(ErrorKind::TooManyDimensions(dimension_count)));
            }
            let mut dimensions = [1; 4];
            let mut element_count = 1_u64;
            for dimension in dimensions.iter_mut().take(dimension_count as usize) {
                *dimension = self.reader.read_u64()?;
                if *dimension > i64::MAX as u64 {
                    return Err(self.reader.error(ErrorKind::InvalidDimension(*dimension)));
                }
                element_count = element_count.checked_mul(*dimension).ok_or_else(|| {
                    self.reader
                        .error(ErrorKind::IntegerOverflow("tensor element count"))
                })?;
                if element_count > i64::MAX as u64 {
                    return Err(self
                        .reader
                        .error(ErrorKind::IntegerOverflow("tensor element count")));
                }
            }
            let raw_type = self.reader.read_u32()?;
            let value_type = TensorType::try_from(raw_type)
                .map_err(|()| self.reader.error(ErrorKind::InvalidTensorType(raw_type)))?;
            let block_size = value_type.block_size();
            if dimensions[0] % block_size != 0 {
                return Err(self.reader.error(ErrorKind::TensorRowNotDivisible {
                    elements_per_row: dimensions[0],
                    block_size,
                }));
            }
            let block_count = element_count.checked_div(block_size).ok_or_else(|| {
                self.reader
                    .error(ErrorKind::IntegerOverflow("tensor block count"))
            })?;
            let byte_len = block_count
                .checked_mul(value_type.type_size())
                .ok_or_else(|| {
                    self.reader
                        .error(ErrorKind::IntegerOverflow("tensor byte length"))
                })?;
            let offset = self.reader.read_u64()?;
            if offset % u64::from(alignment) != 0 {
                return Err(self
                    .reader
                    .error(ErrorKind::MisalignedTensorOffset { offset, alignment }));
            }
            if offset != expected_offset {
                return Err(self.reader.error(ErrorKind::UnexpectedTensorOffset {
                    expected: expected_offset,
                    actual: offset,
                }));
            }
            let padded_byte_len =
                align_up_u64(byte_len, u64::from(alignment)).ok_or_else(|| {
                    self.reader
                        .error(ErrorKind::IntegerOverflow("padded tensor byte length"))
                })?;
            expected_offset = expected_offset
                .checked_add(padded_byte_len)
                .ok_or_else(|| {
                    self.reader
                        .error(ErrorKind::IntegerOverflow("tensor data size"))
                })?;
            if let Some(name) = retained_name {
                tensors.push(RawTensorInfo {
                    name,
                    dimensions,
                    dimension_count,
                    value_type,
                    offset,
                    byte_len,
                });
            }
        }
        drop(tensor_names);

        let header_size = self.reader.position();
        let data_offset = if tensor_count == 0 {
            self.reader.position()
        } else {
            align_up(self.reader.position(), alignment as usize)
                .ok_or_else(|| self.reader.error(ErrorKind::IntegerOverflow("data offset")))?
        };
        let data_offset_u64 = u64::try_from(data_offset)
            .map_err(|_| self.reader.error(ErrorKind::IntegerOverflow("data offset")))?;
        let file_size = self.reader.file_size();
        if data_offset_u64 > file_size {
            let needed = usize::try_from(data_offset_u64 - file_size).unwrap_or(usize::MAX);
            return Err(self.reader.error(ErrorKind::UnexpectedEof {
                needed,
                remaining: 0,
            }));
        }
        let available_data_size = file_size - data_offset_u64;
        if available_data_size < expected_offset {
            return Err(self.reader.error(ErrorKind::TensorDataTruncated {
                expected: expected_offset,
                actual: available_data_size,
            }));
        }
        let header_size_u64 = u64::try_from(header_size)
            .map_err(|_| self.reader.error(ErrorKind::IntegerOverflow("header size")))?;

        Ok(RawGguf {
            version,
            alignment,
            header_size: header_size_u64,
            data_offset: data_offset_u64,
            data_size: expected_offset,
            metadata_count: metadata_count_u64,
            tensor_count: tensor_count_u64,
            metadata,
            tensors,
        })
    }

    fn read_count(&mut self, field: &'static str, limit: u64) -> Result<usize, R::Error> {
        let value = self.reader.read_u64()?;
        if value > limit {
            return Err(self.reader.error(ErrorKind::LimitExceeded {
                field,
                value,
                limit,
            }));
        }
        usize::try_from(value).map_err(|_| self.reader.error(ErrorKind::IntegerOverflow(field)))
    }

    fn own_error_string(
        &self,
        value: Cow<'a, str>,
        allocation_field: &'static str,
    ) -> Result<String, R::Error> {
        match value {
            Cow::Owned(value) => Ok(value),
            Cow::Borrowed(value) => {
                let mut owned = String::new();
                owned.try_reserve_exact(value.len()).map_err(|_| {
                    self.reader
                        .error(ErrorKind::AllocationFailed(allocation_field))
                })?;
                owned.push_str(value);
                Ok(owned)
            }
        }
    }

    fn read_string(&mut self, field: &'static str) -> Result<Cow<'a, str>, R::Error> {
        let length = self.reader.read_u64()?;
        if length > self.limits.max_string_bytes {
            return Err(self.reader.error(ErrorKind::LimitExceeded {
                field,
                value: length,
                limit: self.limits.max_string_bytes,
            }));
        }
        let length = usize::try_from(length)
            .map_err(|_| self.reader.error(ErrorKind::IntegerOverflow(field)))?;
        self.reader.read_string(length, field)
    }

    fn read_metadata_value(&mut self) -> Result<RawMetadataValue<'a>, R::Error> {
        let raw_type = self.reader.read_u32()?;
        let value_type = MetadataType::try_from(raw_type)
            .map_err(|()| self.reader.error(ErrorKind::InvalidMetadataType(raw_type)))?;
        if value_type == MetadataType::Array {
            self.read_array().map(RawMetadataValue::Array)
        } else {
            self.read_scalar(value_type).map(RawMetadataValue::Scalar)
        }
    }

    fn read_scalar(&mut self, value_type: MetadataType) -> Result<RawScalarValue<'a>, R::Error> {
        Ok(match value_type {
            MetadataType::U8 => RawScalarValue::U8(self.reader.read_u8()?),
            MetadataType::I8 => RawScalarValue::I8(self.reader.read_u8()?.cast_signed()),
            MetadataType::U16 => RawScalarValue::U16(self.reader.read_u16()?),
            MetadataType::I16 => RawScalarValue::I16(self.reader.read_i16()?),
            MetadataType::U32 => RawScalarValue::U32(self.reader.read_u32()?),
            MetadataType::I32 => RawScalarValue::I32(self.reader.read_i32()?),
            MetadataType::F32 => RawScalarValue::F32(self.reader.read_f32()?),
            MetadataType::Bool => RawScalarValue::Bool(self.read_bool()?),
            MetadataType::String => RawScalarValue::String(self.read_string("metadata string")?),
            MetadataType::U64 => RawScalarValue::U64(self.reader.read_u64()?),
            MetadataType::I64 => RawScalarValue::I64(self.reader.read_i64()?),
            MetadataType::F64 => RawScalarValue::F64(self.reader.read_f64()?),
            MetadataType::Array => return Err(self.reader.error(ErrorKind::NestedMetadataArray)),
        })
    }

    fn read_array(&mut self) -> Result<RawMetadataArray<'a>, R::Error> {
        let raw_element_type = self.reader.read_u32()?;
        let element_type = MetadataType::try_from(raw_element_type).map_err(|()| {
            self.reader
                .error(ErrorKind::InvalidMetadataType(raw_element_type))
        })?;
        if element_type == MetadataType::Array {
            return Err(self.reader.error(ErrorKind::NestedMetadataArray));
        }
        let len = self.read_count("metadata array length", self.limits.max_array_elements)?;
        if element_type == MetadataType::String {
            self.reader
                .ensure_items(len, size_of::<u64>(), "metadata string array headers")?;
            let mut values = Vec::new();
            if self.retain_output {
                values.try_reserve_exact(len).map_err(|_| {
                    self.reader
                        .error(ErrorKind::AllocationFailed("metadata string array"))
                })?;
            }
            for _ in 0..len {
                let value = self.read_string("metadata array string")?;
                if self.retain_output {
                    values.push(value);
                }
            }
            return Ok(RawMetadataArray::Strings(values));
        }
        let width = element_type
            .fixed_width()
            .ok_or_else(|| self.reader.error(ErrorKind::NestedMetadataArray))?;
        let byte_count = len.checked_mul(width).ok_or_else(|| {
            self.reader
                .error(ErrorKind::IntegerOverflow("metadata array byte count"))
        })?;
        let bytes: Cow<'a, [u8]> = if self.retain_output {
            self.reader.read_bytes(byte_count)?
        } else if element_type == MetadataType::Bool {
            self.reader.validate_boolean_bytes(byte_count)?;
            Cow::Borrowed(&[])
        } else {
            self.reader.skip_bytes(byte_count)?;
            Cow::Borrowed(&[])
        };
        if self.retain_output && element_type == MetadataType::Bool {
            for &value in bytes.as_ref() {
                if value > 1 {
                    return Err(self.reader.error(ErrorKind::InvalidBoolean(value)));
                }
            }
        }
        Ok(RawMetadataArray::Fixed {
            element_type,
            len,
            bytes,
        })
    }

    fn read_bool(&mut self) -> Result<bool, R::Error> {
        let value = self.reader.read_u8()?;
        match value {
            0 => Ok(false),
            1 => Ok(true),
            _ => Err(self.reader.error(ErrorKind::InvalidBoolean(value))),
        }
    }
}

#[derive(Debug)]
struct RawGguf<'a> {
    version: u32,
    alignment: u32,
    header_size: u64,
    data_offset: u64,
    data_size: u64,
    metadata_count: u64,
    tensor_count: u64,
    metadata: Vec<RawMetadataEntry<'a>>,
    tensors: Vec<RawTensorInfo<'a>>,
}

impl<'a> RawGguf<'a> {
    fn into_borrowed(self, source: &'a [u8]) -> Result<Gguf<'a>, Error> {
        let error_offset = usize::try_from(self.header_size).unwrap_or(usize::MAX);
        let data_offset = usize::try_from(self.data_offset)
            .map_err(|_| Error::new(error_offset, ErrorKind::IntegerOverflow("data offset")))?;
        let data_size = usize::try_from(self.data_size).map_err(|_| {
            Error::new(error_offset, ErrorKind::IntegerOverflow("tensor data size"))
        })?;
        let metadata = self
            .metadata
            .into_iter()
            .map(RawMetadataEntry::into_borrowed)
            .collect();
        let tensors = self
            .tensors
            .into_iter()
            .map(RawTensorInfo::into_borrowed)
            .collect();

        Ok(Gguf {
            version: self.version,
            alignment: self.alignment,
            data_offset,
            data_size,
            metadata,
            tensors,
            source,
        })
    }
}

#[derive(Debug)]
struct RawMetadataEntry<'a> {
    key: Cow<'a, str>,
    value: RawMetadataValue<'a>,
}

impl<'a> RawMetadataEntry<'a> {
    fn into_borrowed(self) -> MetadataEntry<'a> {
        MetadataEntry {
            key: expect_borrowed_str(self.key),
            value: self.value.into_borrowed(),
        }
    }
}

#[derive(Debug)]
enum RawMetadataValue<'a> {
    Scalar(RawScalarValue<'a>),
    Array(RawMetadataArray<'a>),
}

impl RawMetadataValue<'_> {
    const fn value_type(&self) -> MetadataType {
        match self {
            Self::Scalar(value) => value.value_type(),
            Self::Array(_) => MetadataType::Array,
        }
    }
}

impl<'a> RawMetadataValue<'a> {
    fn into_borrowed(self) -> MetadataValue<'a> {
        match self {
            Self::Scalar(value) => MetadataValue::Scalar(value.into_borrowed()),
            Self::Array(value) => MetadataValue::Array(value.into_borrowed()),
        }
    }
}

#[derive(Debug)]
enum RawScalarValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(Cow<'a, str>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl RawScalarValue<'_> {
    const fn value_type(&self) -> MetadataType {
        match self {
            Self::U8(_) => MetadataType::U8,
            Self::I8(_) => MetadataType::I8,
            Self::U16(_) => MetadataType::U16,
            Self::I16(_) => MetadataType::I16,
            Self::U32(_) => MetadataType::U32,
            Self::I32(_) => MetadataType::I32,
            Self::F32(_) => MetadataType::F32,
            Self::Bool(_) => MetadataType::Bool,
            Self::String(_) => MetadataType::String,
            Self::U64(_) => MetadataType::U64,
            Self::I64(_) => MetadataType::I64,
            Self::F64(_) => MetadataType::F64,
        }
    }
}

impl<'a> RawScalarValue<'a> {
    fn into_borrowed(self) -> ScalarValue<'a> {
        match self {
            Self::U8(value) => ScalarValue::U8(value),
            Self::I8(value) => ScalarValue::I8(value),
            Self::U16(value) => ScalarValue::U16(value),
            Self::I16(value) => ScalarValue::I16(value),
            Self::U32(value) => ScalarValue::U32(value),
            Self::I32(value) => ScalarValue::I32(value),
            Self::F32(value) => ScalarValue::F32(value),
            Self::Bool(value) => ScalarValue::Bool(value),
            Self::String(value) => ScalarValue::String(expect_borrowed_str(value)),
            Self::U64(value) => ScalarValue::U64(value),
            Self::I64(value) => ScalarValue::I64(value),
            Self::F64(value) => ScalarValue::F64(value),
        }
    }
}

#[derive(Debug)]
enum RawMetadataArray<'a> {
    Fixed {
        element_type: MetadataType,
        len: usize,
        bytes: Cow<'a, [u8]>,
    },
    Strings(Vec<Cow<'a, str>>),
}

impl<'a> RawMetadataArray<'a> {
    fn into_borrowed(self) -> MetadataArray<'a> {
        match self {
            Self::Fixed {
                element_type,
                len,
                bytes,
            } => MetadataArray::fixed(element_type, len, expect_borrowed_bytes(bytes)),
            Self::Strings(values) => {
                MetadataArray::strings(values.into_iter().map(expect_borrowed_str).collect())
            }
        }
    }
}

#[derive(Debug)]
struct RawTensorInfo<'a> {
    name: Cow<'a, str>,
    dimensions: [u64; 4],
    dimension_count: u32,
    value_type: TensorType,
    offset: u64,
    byte_len: u64,
}

impl<'a> RawTensorInfo<'a> {
    fn into_borrowed(self) -> TensorInfo<'a> {
        TensorInfo {
            name: expect_borrowed_str(self.name),
            dimensions: self.dimensions,
            dimension_count: self.dimension_count,
            value_type: self.value_type,
            offset: self.offset,
            byte_len: self.byte_len,
        }
    }
}

#[allow(clippy::needless_pass_by_value)]
fn expect_borrowed_str(value: Cow<'_, str>) -> &str {
    match value {
        Cow::Borrowed(value) => value,
        Cow::Owned(_) => unreachable!("slice parser produced owned text"),
    }
}

#[allow(clippy::needless_pass_by_value)]
fn expect_borrowed_bytes(value: Cow<'_, [u8]>) -> &[u8] {
    match value {
        Cow::Borrowed(value) => value,
        Cow::Owned(_) => unreachable!("slice parser produced owned bytes"),
    }
}

fn align_up(value: usize, alignment: usize) -> Option<usize> {
    let mask = alignment.checked_sub(1)?;
    value.checked_add(mask).map(|sum| sum & !mask)
}

fn align_up_u64(value: u64, alignment: u64) -> Option<u64> {
    let mask = alignment.checked_sub(1)?;
    value.checked_add(mask).map(|sum| sum & !mask)
}

fn is_valid_metadata_key(key: &str) -> bool {
    key.split('.').all(|segment| {
        let mut bytes = segment.bytes();
        bytes
            .next()
            .is_some_and(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
            && bytes.all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
    })
}

trait ParseReader<'a> {
    type Error;

    fn position(&self) -> usize;
    fn file_size(&self) -> u64;
    fn error(&self, kind: ErrorKind) -> Self::Error;
    fn ensure_items(
        &self,
        count: usize,
        minimum_item_bytes: usize,
        field: &'static str,
    ) -> Result<(), Self::Error>;
    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], Self::Error>;
    fn read_string(
        &mut self,
        length: usize,
        field: &'static str,
    ) -> Result<Cow<'a, str>, Self::Error>;
    fn read_bytes(&mut self, length: usize) -> Result<Cow<'a, [u8]>, Self::Error>;

    fn skip_bytes(&mut self, length: usize) -> Result<(), Self::Error> {
        self.read_bytes(length).map(drop)
    }

    fn validate_boolean_bytes(&mut self, length: usize) -> Result<(), Self::Error> {
        let bytes = self.read_bytes(length)?;
        for &value in bytes.as_ref() {
            if value > 1 {
                return Err(self.error(ErrorKind::InvalidBoolean(value)));
            }
        }
        Ok(())
    }

    fn read_u8(&mut self) -> Result<u8, Self::Error> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16, Self::Error> {
        Ok(u16::from_le_bytes(self.read_array()?))
    }

    fn read_i16(&mut self) -> Result<i16, Self::Error> {
        Ok(i16::from_le_bytes(self.read_array()?))
    }

    fn read_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(u32::from_le_bytes(self.read_array()?))
    }

    fn read_i32(&mut self) -> Result<i32, Self::Error> {
        Ok(i32::from_le_bytes(self.read_array()?))
    }

    fn read_f32(&mut self) -> Result<f32, Self::Error> {
        Ok(f32::from_le_bytes(self.read_array()?))
    }

    fn read_u64(&mut self) -> Result<u64, Self::Error> {
        Ok(u64::from_le_bytes(self.read_array()?))
    }

    fn read_i64(&mut self) -> Result<i64, Self::Error> {
        Ok(i64::from_le_bytes(self.read_array()?))
    }

    fn read_f64(&mut self) -> Result<f64, Self::Error> {
        Ok(f64::from_le_bytes(self.read_array()?))
    }
}

struct SliceReader<'a> {
    source: &'a [u8],
    position: usize,
}

struct StreamReader<'reader, R> {
    reader: &'reader mut R,
    position: usize,
    file_size: u64,
    max_header_bytes: u64,
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;
    use std::io::{self, Cursor};

    struct HeaderReadGuard {
        inner: Cursor<Vec<u8>>,
        read_limit: u64,
        bytes_read: u64,
    }

    impl HeaderReadGuard {
        fn new(bytes: Vec<u8>, read_limit: usize) -> Self {
            Self {
                inner: Cursor::new(bytes),
                read_limit: u64::try_from(read_limit).expect("test read limit fits u64"),
                bytes_read: 0,
            }
        }
    }

    impl Read for HeaderReadGuard {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let position = self.inner.position();
            if !buffer.is_empty() && position >= self.read_limit {
                return Err(io::Error::other("attempted to read tensor data"));
            }
            let remaining = usize::try_from(self.read_limit - position).unwrap_or(usize::MAX);
            let read_length = buffer.len().min(remaining);
            let read = self.inner.read(&mut buffer[..read_length])?;
            self.bytes_read += u64::try_from(read).expect("test read count fits u64");
            Ok(read)
        }
    }

    impl Seek for HeaderReadGuard {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    struct ShortRead {
        inner: Cursor<Vec<u8>>,
    }

    impl Read for ShortRead {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            let read_length = buffer.len().min(1);
            self.inner.read(&mut buffer[..read_length])
        }
    }

    impl Seek for ShortRead {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    struct ReadFailure {
        inner: Cursor<Vec<u8>>,
    }

    impl Read for ReadFailure {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected read failure",
            ))
        }
    }

    impl Seek for ReadFailure {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            self.inner.seek(position)
        }
    }

    struct SeekFailure;

    impl Read for SeekFailure {
        fn read(&mut self, _buffer: &mut [u8]) -> io::Result<usize> {
            Ok(0)
        }
    }

    impl Seek for SeekFailure {
        fn seek(&mut self, _position: SeekFrom) -> io::Result<u64> {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "injected seek failure",
            ))
        }
    }

    struct ChangingLengthReader {
        inner: Cursor<Vec<u8>>,
        end_seeks: usize,
    }

    impl Read for ChangingLengthReader {
        fn read(&mut self, buffer: &mut [u8]) -> io::Result<usize> {
            self.inner.read(buffer)
        }
    }

    impl Seek for ChangingLengthReader {
        fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
            match position {
                SeekFrom::End(0) => {
                    let actual = self.inner.seek(SeekFrom::End(0))?;
                    self.end_seeks += 1;
                    if self.end_seeks == 1 {
                        Ok(actual)
                    } else {
                        Ok(actual + 1)
                    }
                }
                other => self.inner.seek(other),
            }
        }
    }

    fn push_string(buffer: &mut Vec<u8>, value: &str) {
        buffer.extend_from_slice(&(value.len() as u64).to_le_bytes());
        buffer.extend_from_slice(value.as_bytes());
    }

    fn push_key(buffer: &mut Vec<u8>, key: &str, value_type: MetadataType) {
        push_string(buffer, key);
        buffer.extend_from_slice(&(value_type as u32).to_le_bytes());
    }

    fn valid_file_with_header_size() -> (Vec<u8>, usize) {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        buffer.extend_from_slice(&3_u64.to_le_bytes());

        push_key(&mut buffer, "general.alignment", MetadataType::U32);
        buffer.extend_from_slice(&32_u32.to_le_bytes());

        push_key(&mut buffer, "general.architecture", MetadataType::String);
        push_string(&mut buffer, "llama");

        push_key(&mut buffer, "tokenizer.ggml.tokens", MetadataType::Array);
        buffer.extend_from_slice(&(MetadataType::String as u32).to_le_bytes());
        buffer.extend_from_slice(&2_u64.to_le_bytes());
        push_string(&mut buffer, "a");
        push_string(&mut buffer, "bb");

        push_string(&mut buffer, "token_embd.weight");
        buffer.extend_from_slice(&2_u32.to_le_bytes());
        buffer.extend_from_slice(&4_u64.to_le_bytes());
        buffer.extend_from_slice(&2_u64.to_le_bytes());
        buffer.extend_from_slice(&0_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());

        let header_size = buffer.len();
        while buffer.len() % 32 != 0 {
            buffer.push(0);
        }
        buffer.resize(buffer.len() + 32, 0);
        (buffer, header_size)
    }

    fn valid_file() -> Vec<u8> {
        valid_file_with_header_size().0
    }

    fn single_tensor_file(
        dimensions: &[u64],
        value_type: u32,
        tensor_offset: u64,
        data_bytes: usize,
    ) -> Vec<u8> {
        single_tensor_file_with_name("weight", dimensions, value_type, tensor_offset, data_bytes)
    }

    fn single_tensor_file_with_name(
        name: &str,
        dimensions: &[u64],
        value_type: u32,
        tensor_offset: u64,
        data_bytes: usize,
    ) -> Vec<u8> {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        push_string(&mut buffer, name);
        buffer.extend_from_slice(
            &u32::try_from(dimensions.len())
                .expect("test tensor rank fits u32")
                .to_le_bytes(),
        );
        for &dimension in dimensions {
            buffer.extend_from_slice(&dimension.to_le_bytes());
        }
        buffer.extend_from_slice(&value_type.to_le_bytes());
        buffer.extend_from_slice(&tensor_offset.to_le_bytes());
        while buffer.len() % 32 != 0 {
            buffer.push(0);
        }
        buffer.resize(buffer.len() + data_bytes, 0);
        buffer
    }

    #[test]
    fn parses_metadata_and_tensor_table() {
        let bytes = valid_file();
        let file = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(file.version(), 3);
        assert_eq!(file.alignment(), 32);
        assert_eq!(file.metadata().len(), 3);
        assert_eq!(file.tensors().len(), 1);
        assert_eq!(file.tensors()[0].shape(), &[4, 2]);
        assert_eq!(file.tensors()[0].value_type.raw(), 0);
        assert_eq!(file.tensors()[0].byte_len, 32);
        assert_eq!(file.data_size(), 32);
        assert_eq!(file.data().len(), 32);
        let tensor_range = file.tensor_data_range(0).unwrap();
        assert_eq!(tensor_range, file.data_offset()..file.data_offset() + 32);
        assert_eq!(file.tensor_data(0), Some(&bytes[tensor_range]));
        assert_eq!(file.tensor_data(1), None);

        let MetadataValue::Array(tokens) = file.metadata_value("tokenizer.ggml.tokens").unwrap()
        else {
            panic!("expected token array");
        };
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens.get(0), Some(ScalarValue::String("a")));
        assert_eq!(tokens.get(1), Some(ScalarValue::String("bb")));
    }

    #[test]
    fn validates_seekable_reader_without_reading_tensor_data() {
        let (bytes, header_size) = valid_file_with_header_size();
        let file_size = u64::try_from(bytes.len()).unwrap();
        let mut reader = HeaderReadGuard::new(bytes, header_size);

        let header = GgufHeader::from_reader(&mut reader).unwrap();

        assert_eq!(header.version(), 3);
        assert_eq!(header.alignment(), 32);
        assert_eq!(header.header_size(), u64::try_from(header_size).unwrap());
        let expected_data_offset = u64::try_from(align_up(header_size, 32).unwrap()).unwrap();
        assert_eq!(header.data_offset(), expected_data_offset);
        assert_eq!(header.data_size(), 32);
        assert_eq!(header.file_size(), file_size);
        assert_eq!(header.metadata_count(), 3);
        assert_eq!(header.tensor_count(), 1);
        assert_eq!(
            header.tensor_data_range(),
            expected_data_offset..expected_data_offset + 32
        );
        assert_eq!(reader.bytes_read, u64::try_from(header_size).unwrap());
        assert_eq!(reader.inner.position(), header.data_offset());
    }

    #[test]
    fn seekable_reader_ignores_nonzero_initial_cursor() {
        let bytes = valid_file();
        let mut reader = Cursor::new(bytes);
        reader.set_position(17);

        let header = GgufHeader::from_reader(&mut reader).unwrap();

        assert_eq!(header.version(), 3);
        assert_eq!(reader.position(), header.data_offset());
    }

    #[test]
    fn seekable_reader_accepts_short_reads() {
        let mut reader = ShortRead {
            inner: Cursor::new(valid_file()),
        };

        let header = GgufHeader::from_reader(&mut reader).unwrap();

        assert_eq!(header.tensor_count(), 1);
        assert_eq!(reader.inner.position(), header.data_offset());
    }

    #[test]
    fn seekable_reader_reports_read_failure_operation_and_offset() {
        let mut reader = ReadFailure {
            inner: Cursor::new(valid_file()),
        };

        let error = GgufHeader::from_reader(&mut reader).unwrap_err();
        let message = error.to_string();

        assert!(matches!(
            &error,
            ReadError::Io {
                operation: "reading header bytes",
                offset: Some(0),
                source,
            } if source.kind() == io::ErrorKind::PermissionDenied
        ));
        assert!(message.contains("reading header bytes at byte 0"));
    }

    #[test]
    fn seekable_reader_reports_seek_failure_operation() {
        let mut reader = SeekFailure;

        let error = GgufHeader::from_reader(&mut reader).unwrap_err();
        let message = error.to_string();

        assert!(matches!(
            &error,
            ReadError::Io {
                operation: "determining the initial file length",
                offset: None,
                source,
            } if source.kind() == io::ErrorKind::PermissionDenied
        ));
        assert!(message.contains("determining the initial file length"));
        assert!(!message.contains("failed to read"));
    }

    #[test]
    fn seekable_reader_rejects_file_length_change() {
        let bytes = valid_file();
        let expected = u64::try_from(bytes.len()).unwrap();
        let mut reader = ChangingLengthReader {
            inner: Cursor::new(bytes),
            end_seeks: 0,
        };

        let error = GgufHeader::from_reader(&mut reader).unwrap_err();

        assert!(matches!(
            error,
            ReadError::FileLengthChanged {
                expected: observed_expected,
                actual,
            } if observed_expected == expected && actual == expected + 1
        ));
    }

    #[test]
    fn seekable_reader_enforces_header_byte_limit_before_reading_past_it() {
        let (bytes, header_size) = valid_file_with_header_size();
        let limit = u64::try_from(header_size).unwrap() - 1;
        let mut reader = Cursor::new(bytes);
        let limits = HeaderLimits {
            max_header_bytes: limit,
            ..HeaderLimits::default()
        };

        let error = GgufHeader::from_reader_with_limits(&mut reader, limits).unwrap_err();
        assert!(matches!(
            error,
            ReadError::Parse(Error {
                kind: ErrorKind::LimitExceeded {
                    field: "header bytes",
                    limit: observed_limit,
                    ..
                },
                ..
            }) if observed_limit == limit
        ));
        assert!(reader.position() <= limit);
    }

    #[test]
    fn seekable_reader_rejects_every_truncated_prefix() {
        let bytes = valid_file();
        for length in 0..bytes.len() {
            let expected = Gguf::from_bytes(&bytes[..length]).unwrap_err();
            let mut reader = Cursor::new(bytes[..length].to_vec());
            let actual = GgufHeader::from_reader(&mut reader).unwrap_err();
            assert!(
                matches!(actual, ReadError::Parse(ref error) if error == &expected),
                "different error for prefix length {length}: expected {expected:?}, got {actual:?}"
            );
        }
    }

    #[test]
    fn seekable_reader_rejects_truncated_tensor_data_from_file_length() {
        let bytes = single_tensor_file(&[8], 0, 0, 31);
        let mut reader = Cursor::new(bytes);
        let error = GgufHeader::from_reader(&mut reader).unwrap_err();

        assert!(matches!(
            error,
            ReadError::Parse(Error {
                kind: ErrorKind::TensorDataTruncated {
                    expected: 32,
                    actual: 31
                },
                ..
            })
        ));
    }

    #[test]
    fn seekable_reader_validates_boolean_array_bytes_while_streaming() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut bytes, "flags", MetadataType::Array);
        bytes.extend_from_slice(&(MetadataType::Bool as u32).to_le_bytes());
        bytes.extend_from_slice(&3_u64.to_le_bytes());
        bytes.extend_from_slice(&[0, 2, 1]);

        let expected = Gguf::from_bytes(&bytes).unwrap_err();
        let mut reader = Cursor::new(bytes);
        let actual = GgufHeader::from_reader(&mut reader).unwrap_err();

        assert!(matches!(actual, ReadError::Parse(ref error) if error == &expected));
        assert_eq!(expected.kind, ErrorKind::InvalidBoolean(2));
    }

    #[test]
    fn seekable_reader_rejects_hostile_counts_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"GGUF");
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&u64::MAX.to_le_bytes());
        let mut reader = Cursor::new(bytes);
        let limits = HeaderLimits {
            format: Limits::upstream_compatible(),
            max_header_bytes: 64,
        };

        let error = GgufHeader::from_reader_with_limits(&mut reader, limits).unwrap_err();
        assert!(matches!(
            error,
            ReadError::Parse(Error {
                kind: ErrorKind::IntegerOverflow("metadata table"),
                ..
            })
        ));
    }

    #[test]
    fn parses_version_two_layout() {
        let mut bytes = valid_file();
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());

        let file = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(file.version(), 2);
        assert_eq!(file.tensors().len(), 1);
        assert_eq!(file.metadata().len(), 3);
    }

    #[test]
    fn parses_every_supported_current_tensor_type() {
        for raw_type in 0..TensorType::COUNT {
            let Ok(value_type) = TensorType::try_from(raw_type) else {
                continue;
            };
            let padded_bytes = usize::try_from(
                align_up_u64(value_type.type_size(), u64::from(DEFAULT_ALIGNMENT)).unwrap(),
            )
            .unwrap();
            let bytes = single_tensor_file(&[value_type.block_size()], raw_type, 0, padded_bytes);

            let file = Gguf::from_bytes(&bytes).unwrap();
            assert_eq!(file.tensors()[0].value_type, value_type);
            assert_eq!(file.tensors()[0].byte_len, value_type.type_size());
        }
    }

    #[test]
    fn bounded_deterministic_noise_never_panics() {
        let mut state = 0x4d59_5df4_d0f3_3173_u64;
        for length in 0..512 {
            let mut bytes = Vec::with_capacity(length);
            for _ in 0..length {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1);
                bytes.push((state >> 56) as u8);
            }
            let _ = Gguf::from_bytes(&bytes);
        }
    }

    #[test]
    fn rejects_every_truncated_prefix() {
        let bytes = valid_file();
        for length in 0..bytes.len() {
            assert!(
                Gguf::from_bytes(&bytes[..length]).is_err(),
                "accepted prefix length {length}"
            );
        }
    }

    #[test]
    fn rejects_duplicate_metadata_keys() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&2_u64.to_le_bytes());
        for _ in 0..2 {
            push_key(&mut buffer, "duplicate", MetadataType::U8);
            buffer.push(1);
        }
        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert!(matches!(error.kind, ErrorKind::DuplicateMetadataKey(_)));
    }

    #[test]
    fn distinguishes_unsupported_and_byte_swapped_versions() {
        let mut unsupported = Vec::from(*b"GGUF");
        unsupported.extend_from_slice(&0_u32.to_le_bytes());
        let error = Gguf::from_bytes(&unsupported).unwrap_err();
        assert_eq!(error.kind, ErrorKind::UnsupportedVersion(0));

        let mut byte_swapped = Vec::from(*b"GGUF");
        byte_swapped.extend_from_slice(&3_u32.to_be_bytes());
        let error = Gguf::from_bytes(&byte_swapped).unwrap_err();
        assert_eq!(error.kind, ErrorKind::EndiannessMismatch(0x0300_0000));
    }

    #[test]
    fn rejects_metadata_key_outside_canonical_grammar() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut buffer, "General.Architecture", MetadataType::String);
        push_string(&mut buffer, "llama");

        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidMetadataKey);
    }

    #[test]
    fn accepts_indexed_metadata_key_segments() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(
            &mut buffer,
            "general.base_model.0.name",
            MetadataType::String,
        );
        push_string(&mut buffer, "base");

        let file = Gguf::from_bytes(&buffer).unwrap();
        assert_eq!(file.metadata()[0].key, "general.base_model.0.name");
    }

    #[test]
    fn rejects_metadata_key_longer_than_format_limit() {
        let key = "a".repeat(MAX_METADATA_KEY_BYTES + 1);
        assert_eq!(key.len(), MAX_METADATA_KEY_BYTES + 1);

        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut buffer, &key, MetadataType::U8);
        buffer.push(1);

        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::MetadataKeyTooLong(MAX_METADATA_KEY_BYTES + 1)
        );
    }

    #[test]
    fn rejects_tensor_name_with_interior_nul() {
        let bytes = single_tensor_file_with_name("bad\0name", &[8], 0, 0, 32);
        let error = Gguf::from_bytes(&bytes).unwrap_err();
        assert_eq!(error.kind, ErrorKind::TensorNameContainsNul);
    }

    #[test]
    fn preflights_metadata_table_before_reserving() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&Limits::default().max_metadata_entries.to_le_bytes());

        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::UnexpectedEof {
                needed: usize::try_from(Limits::default().max_metadata_entries).unwrap()
                    * MIN_METADATA_ENTRY_BYTES,
                remaining: 0
            }
        );
    }

    #[test]
    fn preflights_string_array_headers_before_reserving() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut buffer, "values", MetadataType::Array);
        buffer.extend_from_slice(&(MetadataType::String as u32).to_le_bytes());
        buffer.extend_from_slice(&Limits::default().max_array_elements.to_le_bytes());

        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::UnexpectedEof {
                needed: usize::try_from(Limits::default().max_array_elements).unwrap()
                    * size_of::<u64>(),
                remaining: 0
            }
        );
    }

    #[test]
    fn rejects_noncanonical_boolean() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut buffer, "flag", MetadataType::Bool);
        buffer.push(2);
        let error = Gguf::from_bytes(&buffer).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidBoolean(2));
    }

    #[test]
    fn enforces_configured_array_limit_before_allocation() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(b"GGUF");
        buffer.extend_from_slice(&3_u32.to_le_bytes());
        buffer.extend_from_slice(&0_u64.to_le_bytes());
        buffer.extend_from_slice(&1_u64.to_le_bytes());
        push_key(&mut buffer, "values", MetadataType::Array);
        buffer.extend_from_slice(&(MetadataType::U8 as u32).to_le_bytes());
        buffer.extend_from_slice(&9_u64.to_le_bytes());

        let limits = Limits {
            max_array_elements: 8,
            ..Limits::default()
        };
        let error = Gguf::from_bytes_with_limits(&buffer, limits).unwrap_err();
        assert!(matches!(
            error.kind,
            ErrorKind::LimitExceeded {
                field: "metadata array length",
                value: 9,
                limit: 8
            }
        ));
    }

    #[test]
    fn rejects_quantized_row_that_is_not_block_aligned() {
        let bytes = single_tensor_file(&[31], 2, 0, 32);
        let error = Gguf::from_bytes(&bytes).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::TensorRowNotDivisible {
                elements_per_row: 31,
                block_size: 32
            }
        );
    }

    #[test]
    fn rejects_gap_before_first_tensor() {
        let bytes = single_tensor_file(&[8], 0, 32, 32);
        let error = Gguf::from_bytes(&bytes).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::UnexpectedTensorOffset {
                expected: 0,
                actual: 32
            }
        );
    }

    #[test]
    fn rejects_truncated_tensor_data() {
        let bytes = single_tensor_file(&[8], 0, 0, 31);
        let error = Gguf::from_bytes(&bytes).unwrap_err();
        assert_eq!(
            error.kind,
            ErrorKind::TensorDataTruncated {
                expected: 32,
                actual: 31
            }
        );
    }

    #[test]
    fn excludes_trailing_bytes_from_tensor_data() {
        let bytes = single_tensor_file(&[8], 0, 0, 37);
        let file = Gguf::from_bytes(&bytes).unwrap();
        assert_eq!(file.data_size(), 32);
        assert_eq!(file.data().len(), 32);
    }
}

impl<'a> SliceReader<'a> {
    const fn new(source: &'a [u8]) -> Self {
        Self {
            source,
            position: 0,
        }
    }

    fn take(&mut self, length: usize) -> Result<&'a [u8], Error> {
        let remaining = self.source.len().saturating_sub(self.position);
        if length > remaining {
            return Err(Error::new(
                self.position,
                ErrorKind::UnexpectedEof {
                    needed: length,
                    remaining,
                },
            ));
        }
        let end = self.position.checked_add(length).ok_or_else(|| {
            Error::new(self.position, ErrorKind::IntegerOverflow("reader position"))
        })?;
        let bytes = &self.source[self.position..end];
        self.position = end;
        Ok(bytes)
    }
}

impl<'a> ParseReader<'a> for SliceReader<'a> {
    type Error = Error;

    fn position(&self) -> usize {
        self.position
    }

    fn file_size(&self) -> u64 {
        u64::try_from(self.source.len()).expect("slice length fits u64")
    }

    fn error(&self, kind: ErrorKind) -> Self::Error {
        Error::new(self.position, kind)
    }

    fn ensure_items(
        &self,
        count: usize,
        minimum_item_bytes: usize,
        field: &'static str,
    ) -> Result<(), Error> {
        let needed = count
            .checked_mul(minimum_item_bytes)
            .ok_or_else(|| self.error(ErrorKind::IntegerOverflow(field)))?;
        let remaining = self.source.len().saturating_sub(self.position);
        if needed > remaining {
            return Err(self.error(ErrorKind::UnexpectedEof { needed, remaining }));
        }
        Ok(())
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        self.take(N)?.try_into().map_err(|_| {
            self.error(ErrorKind::UnexpectedEof {
                needed: N,
                remaining: 0,
            })
        })
    }

    fn read_string(
        &mut self,
        length: usize,
        field: &'static str,
    ) -> Result<Cow<'a, str>, Self::Error> {
        let bytes = self.take(length)?;
        std::str::from_utf8(bytes)
            .map(Cow::Borrowed)
            .map_err(|_| self.error(ErrorKind::InvalidUtf8(field)))
    }

    fn read_bytes(&mut self, length: usize) -> Result<Cow<'a, [u8]>, Self::Error> {
        self.take(length).map(Cow::Borrowed)
    }
}

impl<'reader, R> StreamReader<'reader, R> {
    const fn new(reader: &'reader mut R, file_size: u64, max_header_bytes: u64) -> Self {
        Self {
            reader,
            position: 0,
            file_size,
            max_header_bytes,
        }
    }

    fn error(&self, kind: ErrorKind) -> ReadError {
        ReadError::Parse(Error::new(self.position, kind))
    }

    fn checked_end(&self, length: usize) -> Result<usize, ReadError> {
        let end = self
            .position
            .checked_add(length)
            .ok_or_else(|| self.error(ErrorKind::IntegerOverflow("reader position")))?;
        let end_u64 = u64::try_from(end)
            .map_err(|_| self.error(ErrorKind::IntegerOverflow("reader position")))?;
        if end_u64 > self.max_header_bytes {
            return Err(self.error(ErrorKind::LimitExceeded {
                field: "header bytes",
                value: end_u64,
                limit: self.max_header_bytes,
            }));
        }
        if end_u64 > self.file_size {
            let remaining = usize::try_from(
                self.file_size
                    .saturating_sub(u64::try_from(self.position).unwrap_or(u64::MAX)),
            )
            .unwrap_or(usize::MAX);
            return Err(self.error(ErrorKind::UnexpectedEof {
                needed: length,
                remaining,
            }));
        }
        Ok(end)
    }

    fn read_owned_bytes(&mut self, length: usize) -> Result<Vec<u8>, ReadError>
    where
        R: Read,
    {
        let end = self.checked_end(length)?;
        let mut bytes = Vec::new();
        bytes
            .try_reserve_exact(length)
            .map_err(|_| self.error(ErrorKind::AllocationFailed("seekable-reader field")))?;
        bytes.resize(length, 0);
        let offset = u64::try_from(self.position).ok();
        self.reader
            .read_exact(&mut bytes)
            .map_err(|error| ReadError::io("reading header bytes", offset, error))?;
        self.position = end;
        Ok(bytes)
    }
}

impl<'a, R: Read + Seek> ParseReader<'a> for StreamReader<'_, R> {
    type Error = ReadError;

    fn position(&self) -> usize {
        self.position
    }

    fn file_size(&self) -> u64 {
        self.file_size
    }

    fn error(&self, kind: ErrorKind) -> Self::Error {
        self.error(kind)
    }

    fn ensure_items(
        &self,
        count: usize,
        minimum_item_bytes: usize,
        field: &'static str,
    ) -> Result<(), Self::Error> {
        let needed = count
            .checked_mul(minimum_item_bytes)
            .ok_or_else(|| self.error(ErrorKind::IntegerOverflow(field)))?;
        self.checked_end(needed).map(|_| ())
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N], Self::Error> {
        let end = self.checked_end(N)?;
        let mut bytes = [0; N];
        let offset = u64::try_from(self.position).ok();
        self.reader
            .read_exact(&mut bytes)
            .map_err(|error| ReadError::io("reading header bytes", offset, error))?;
        self.position = end;
        Ok(bytes)
    }

    fn read_string(
        &mut self,
        length: usize,
        field: &'static str,
    ) -> Result<Cow<'a, str>, Self::Error> {
        let bytes = self.read_owned_bytes(length)?;
        String::from_utf8(bytes)
            .map(Cow::Owned)
            .map_err(|_| self.error(ErrorKind::InvalidUtf8(field)))
    }

    fn read_bytes(&mut self, length: usize) -> Result<Cow<'a, [u8]>, Self::Error> {
        self.read_owned_bytes(length).map(Cow::Owned)
    }

    fn skip_bytes(&mut self, length: usize) -> Result<(), Self::Error> {
        let end = self.checked_end(length)?;
        let end_u64 = u64::try_from(end)
            .map_err(|_| self.error(ErrorKind::IntegerOverflow("reader position")))?;
        self.reader
            .seek(SeekFrom::Start(end_u64))
            .map_err(|error| {
                ReadError::io("skipping fixed-width metadata values", Some(end_u64), error)
            })?;
        self.position = end;
        Ok(())
    }

    fn validate_boolean_bytes(&mut self, length: usize) -> Result<(), Self::Error> {
        const BUFFER_BYTES: usize = 8192;

        let end = self.checked_end(length)?;
        let mut remaining = length;
        let mut buffer = [0; BUFFER_BYTES];
        let mut invalid = None;
        while remaining != 0 {
            let chunk_length = remaining.min(BUFFER_BYTES);
            let consumed = length - remaining;
            let offset = self
                .position
                .checked_add(consumed)
                .and_then(|value| u64::try_from(value).ok());
            self.reader
                .read_exact(&mut buffer[..chunk_length])
                .map_err(|error| ReadError::io("reading boolean metadata values", offset, error))?;
            if invalid.is_none() {
                invalid = buffer[..chunk_length]
                    .iter()
                    .copied()
                    .find(|&value| value > 1);
            }
            remaining -= chunk_length;
        }
        self.position = end;
        if let Some(value) = invalid {
            return Err(self.error(ErrorKind::InvalidBoolean(value)));
        }
        Ok(())
    }
}
