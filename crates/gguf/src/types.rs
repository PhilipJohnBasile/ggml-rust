use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum MetadataType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl TryFrom<u32> for MetadataType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(()),
        }
    }
}

impl MetadataType {
    pub(crate) const fn fixed_width(self) -> Option<usize> {
        match self {
            Self::U8 | Self::I8 | Self::Bool => Some(1),
            Self::U16 | Self::I16 => Some(2),
            Self::U32 | Self::I32 | Self::F32 => Some(4),
            Self::U64 | Self::I64 | Self::F64 => Some(8),
            Self::String | Self::Array => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue<'a> {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(&'a str),
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue<'a> {
    Scalar(ScalarValue<'a>),
    Array(MetadataArray<'a>),
}

impl MetadataValue<'_> {
    #[must_use]
    pub const fn value_type(&self) -> MetadataType {
        match self {
            Self::Scalar(ScalarValue::U8(_)) => MetadataType::U8,
            Self::Scalar(ScalarValue::I8(_)) => MetadataType::I8,
            Self::Scalar(ScalarValue::U16(_)) => MetadataType::U16,
            Self::Scalar(ScalarValue::I16(_)) => MetadataType::I16,
            Self::Scalar(ScalarValue::U32(_)) => MetadataType::U32,
            Self::Scalar(ScalarValue::I32(_)) => MetadataType::I32,
            Self::Scalar(ScalarValue::F32(_)) => MetadataType::F32,
            Self::Scalar(ScalarValue::Bool(_)) => MetadataType::Bool,
            Self::Scalar(ScalarValue::String(_)) => MetadataType::String,
            Self::Scalar(ScalarValue::U64(_)) => MetadataType::U64,
            Self::Scalar(ScalarValue::I64(_)) => MetadataType::I64,
            Self::Scalar(ScalarValue::F64(_)) => MetadataType::F64,
            Self::Array(_) => MetadataType::Array,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataArray<'a> {
    element_type: MetadataType,
    len: usize,
    storage: ArrayStorage<'a>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ArrayStorage<'a> {
    Fixed(&'a [u8]),
    Strings(Vec<&'a str>),
}

impl<'a> MetadataArray<'a> {
    pub(crate) const fn fixed(element_type: MetadataType, len: usize, bytes: &'a [u8]) -> Self {
        Self {
            element_type,
            len,
            storage: ArrayStorage::Fixed(bytes),
        }
    }

    pub(crate) fn strings(values: Vec<&'a str>) -> Self {
        Self {
            element_type: MetadataType::String,
            len: values.len(),
            storage: ArrayStorage::Strings(values),
        }
    }

    #[must_use]
    pub const fn element_type(&self) -> MetadataType {
        self.element_type
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<ScalarValue<'a>> {
        if index >= self.len {
            return None;
        }
        match &self.storage {
            ArrayStorage::Strings(values) => values.get(index).copied().map(ScalarValue::String),
            ArrayStorage::Fixed(bytes) => decode_fixed(self.element_type, bytes, index),
        }
    }
}

fn decode_fixed(value_type: MetadataType, bytes: &[u8], index: usize) -> Option<ScalarValue<'_>> {
    let width = value_type.fixed_width()?;
    let start = index.checked_mul(width)?;
    let value = bytes.get(start..start + width)?;
    Some(match value_type {
        MetadataType::U8 => ScalarValue::U8(value[0]),
        MetadataType::I8 => ScalarValue::I8(value[0].cast_signed()),
        MetadataType::U16 => ScalarValue::U16(u16::from_le_bytes(value.try_into().ok()?)),
        MetadataType::I16 => ScalarValue::I16(i16::from_le_bytes(value.try_into().ok()?)),
        MetadataType::U32 => ScalarValue::U32(u32::from_le_bytes(value.try_into().ok()?)),
        MetadataType::I32 => ScalarValue::I32(i32::from_le_bytes(value.try_into().ok()?)),
        MetadataType::F32 => ScalarValue::F32(f32::from_le_bytes(value.try_into().ok()?)),
        MetadataType::Bool => ScalarValue::Bool(value[0] != 0),
        MetadataType::U64 => ScalarValue::U64(u64::from_le_bytes(value.try_into().ok()?)),
        MetadataType::I64 => ScalarValue::I64(i64::from_le_bytes(value.try_into().ok()?)),
        MetadataType::F64 => ScalarValue::F64(f64::from_le_bytes(value.try_into().ok()?)),
        MetadataType::String | MetadataType::Array => return None,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub struct MetadataEntry<'a> {
    pub key: &'a str,
    pub value: MetadataValue<'a>,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TensorType(u32);

impl TensorType {
    pub const COUNT: u32 = 43;

    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[must_use]
    pub const fn name(self) -> &'static str {
        const NAMES: [&str; TensorType::COUNT as usize] = [
            "F32",
            "F16",
            "Q4_0",
            "Q4_1",
            "REMOVED_4",
            "REMOVED_5",
            "Q5_0",
            "Q5_1",
            "Q8_0",
            "Q8_1",
            "Q2_K",
            "Q3_K",
            "Q4_K",
            "Q5_K",
            "Q6_K",
            "Q8_K",
            "IQ2_XXS",
            "IQ2_XS",
            "IQ3_XXS",
            "IQ1_S",
            "IQ4_NL",
            "IQ3_S",
            "IQ2_S",
            "IQ4_XS",
            "I8",
            "I16",
            "I32",
            "I64",
            "F64",
            "IQ1_M",
            "BF16",
            "REMOVED_31",
            "REMOVED_32",
            "REMOVED_33",
            "TQ1_0",
            "TQ2_0",
            "REMOVED_36",
            "REMOVED_37",
            "REMOVED_38",
            "MXFP4",
            "NVFP4",
            "Q1_0",
            "Q2_0",
        ];
        NAMES[self.0 as usize]
    }

    /// Number of logical elements represented by one stored block.
    #[must_use]
    pub const fn block_size(self) -> u64 {
        const BLOCK_SIZES: [u64; TensorType::COUNT as usize] = [
            1, 1, 32, 32, 0, 0, 32, 32, 32, 32, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256,
            32, 256, 256, 256, 1, 1, 1, 1, 1, 256, 1, 0, 0, 0, 256, 256, 0, 0, 0, 32, 64, 128, 64,
        ];
        BLOCK_SIZES[self.0 as usize]
    }

    /// Number of stored bytes occupied by one block.
    #[must_use]
    pub const fn type_size(self) -> u64 {
        const TYPE_SIZES: [u64; TensorType::COUNT as usize] = [
            4, 2, 18, 20, 0, 0, 22, 24, 34, 36, 84, 110, 144, 176, 210, 292, 66, 74, 98, 50, 18,
            110, 82, 136, 1, 2, 4, 8, 8, 56, 2, 0, 0, 0, 54, 66, 0, 0, 0, 17, 36, 18, 18,
        ];
        TYPE_SIZES[self.0 as usize]
    }
}

impl TryFrom<u32> for TensorType {
    type Error = ();

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        if value >= Self::COUNT || matches!(value, 4 | 5 | 31 | 32 | 33 | 36 | 37 | 38) {
            Err(())
        } else {
            Ok(Self(value))
        }
    }
}

impl fmt::Debug for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}({})", self.name(), self.0)
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo<'a> {
    pub name: &'a str,
    pub dimensions: [u64; 4],
    pub dimension_count: u32,
    pub value_type: TensorType,
    pub offset: u64,
    pub byte_len: u64,
}

impl TensorInfo<'_> {
    #[must_use]
    pub fn shape(&self) -> &[u64] {
        &self.dimensions[..self.dimension_count as usize]
    }
}
