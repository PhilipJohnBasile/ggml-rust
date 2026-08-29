#![forbid(unsafe_code)]

use std::fmt;

/// An owned row-major tensor containing 32-bit floating-point values.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f32>,
}

impl fmt::Debug for Tensor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Tensor")
            .field("shape", &self.shape)
            .field("data", &self.data)
            .finish()
    }
}

/// Failures returned by checked tensor construction and operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TensorError {
    ElementCountOverflow,
    DataLength {
        expected: usize,
        actual: usize,
    },
    ShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    RankMismatch {
        expected: usize,
        actual: usize,
    },
    MatrixShapeMismatch {
        left: Vec<usize>,
        right: Vec<usize>,
    },
    AttentionShapeMismatch {
        queries: Vec<usize>,
        keys: Vec<usize>,
        values: Vec<usize>,
    },
    ZeroDimension,
    InvalidEpsilon,
    InvalidScale,
    RotaryShapeMismatch {
        shape: Vec<usize>,
    },
    InvalidFrequencyBase,
    InvalidPosition,
    NonFiniteInput {
        index: usize,
    },
    NonFiniteOutput {
        operation: &'static str,
        index: usize,
    },
}

impl fmt::Display for TensorError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ElementCountOverflow => formatter.write_str("tensor element count overflowed"),
            Self::DataLength { expected, actual } => write!(
                formatter,
                "tensor data contains {actual} values, expected {expected}"
            ),
            Self::ShapeMismatch { left, right } => {
                write!(formatter, "tensor shapes differ: {left:?} and {right:?}")
            }
            Self::RankMismatch { expected, actual } => {
                write!(formatter, "tensor rank is {actual}, expected {expected}")
            }
            Self::MatrixShapeMismatch { left, right } => {
                write!(
                    formatter,
                    "matrix shapes cannot be multiplied: {left:?} and {right:?}"
                )
            }
            Self::AttentionShapeMismatch {
                queries,
                keys,
                values,
            } => write!(
                formatter,
                "attention shapes are incompatible: queries {queries:?}, keys {keys:?}, values {values:?}"
            ),
            Self::ZeroDimension => formatter.write_str("tensor dimensions must be nonzero"),
            Self::InvalidEpsilon => {
                formatter.write_str("RMSNorm epsilon must be finite and nonnegative")
            }
            Self::InvalidScale => formatter.write_str("attention scale must be finite"),
            Self::RotaryShapeMismatch { shape } => write!(
                formatter,
                "rotary embedding requires a rank-2 head tensor, got {shape:?}"
            ),
            Self::InvalidFrequencyBase => {
                formatter.write_str("rotary frequency base must be finite and positive")
            }
            Self::InvalidPosition => formatter.write_str("rotary position must be finite"),
            Self::NonFiniteInput { index } => {
                write!(formatter, "tensor input at index {index} is not finite")
            }
            Self::NonFiniteOutput { operation, index } => {
                write!(
                    formatter,
                    "tensor operation {operation} produced a non-finite value at index {index}"
                )
            }
        }
    }
}

impl std::error::Error for TensorError {}

impl Tensor {
    /// Creates a tensor from a shape and row-major values.
    ///
    /// An empty shape represents a scalar and therefore requires exactly one
    /// value. Non-scalar dimensions must be nonzero.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, the shape product overflows,
    /// or the data length does not match the shape.
    pub fn from_data<I>(shape: I, data: impl Into<Vec<f32>>) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = usize>,
    {
        let shape = shape.into_iter().collect::<Vec<_>>();
        validate_shape(&shape)?;
        let data = data.into();
        let expected = element_count(&shape)?;
        if data.len() != expected {
            return Err(TensorError::DataLength {
                expected,
                actual: data.len(),
            });
        }
        Ok(Self { shape, data })
    }

    /// Creates a zero-filled tensor with the requested shape.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero or the shape product
    /// overflows.
    pub fn zeros<I>(shape: I) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = usize>,
    {
        let shape = shape.into_iter().collect::<Vec<_>>();
        validate_shape(&shape)?;
        let count = element_count(&shape)?;
        Ok(Self {
            shape,
            data: vec![0.0; count],
        })
    }

    /// Creates a scalar tensor.
    #[must_use]
    pub fn scalar(value: f32) -> Self {
        Self {
            shape: Vec::new(),
            data: vec![value],
        }
    }

    /// Returns the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the tensor rank.
    #[must_use]
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Returns the number of logical elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns whether the tensor has no elements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an immutable view of row-major tensor data.
    #[must_use]
    pub fn data(&self) -> &[f32] {
        &self.data
    }

    /// Consumes the tensor and returns its row-major data.
    #[must_use]
    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Returns the value at a flat row-major index.
    #[must_use]
    pub fn get_flat(&self, index: usize) -> Option<f32> {
        self.data.get(index).copied()
    }

    /// Reshapes the tensor without moving or changing its values.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, the shape product overflows,
    /// or the new shape has a different element count.
    pub fn reshape<I>(mut self, shape: I) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = usize>,
    {
        let shape = shape.into_iter().collect::<Vec<_>>();
        validate_shape(&shape)?;
        let expected = element_count(&shape)?;
        if expected != self.data.len() {
            return Err(TensorError::DataLength {
                expected,
                actual: self.data.len(),
            });
        }
        self.shape = shape;
        Ok(self)
    }

    /// Transposes a rank-2 tensor by copying its values into row-major order.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is not rank 2 or the output size
    /// overflows.
    pub fn transpose_2d(&self) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let output_len = rows
            .checked_mul(columns)
            .ok_or(TensorError::ElementCountOverflow)?;
        let mut result = vec![0.0; output_len];
        for row in 0..rows {
            for column in 0..columns {
                result[column * rows + row] = self.data[row * columns + column];
            }
        }
        checked_output(vec![columns, rows], result, "transpose")
    }

    /// Broadcasts this tensor to a larger, right-aligned shape by copying
    /// values. A source dimension must equal the destination dimension or be
    /// one, matching MLX's broadcasting rule.
    ///
    /// # Errors
    ///
    /// Returns an error when the destination rank is smaller than the source
    /// rank, a dimension is incompatible, or the output size overflows.
    pub fn broadcast_to<I>(&self, shape: I) -> Result<Self, TensorError>
    where
        I: IntoIterator<Item = usize>,
    {
        let destination = shape.into_iter().collect::<Vec<_>>();
        validate_shape(&destination)?;
        if destination.len() < self.rank() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: destination,
            });
        }
        let offset = destination.len() - self.rank();
        for (axis, &source_dimension) in self.shape.iter().enumerate() {
            let target_dimension = destination[offset + axis];
            if source_dimension != target_dimension && source_dimension != 1 {
                return Err(TensorError::ShapeMismatch {
                    left: self.shape.clone(),
                    right: destination,
                });
            }
        }
        let output_len = element_count(&destination)?;
        let mut result = Vec::with_capacity(output_len);
        let mut coordinates = vec![0_usize; destination.len()];
        for _ in 0..output_len {
            let mut source_index = 0_usize;
            for (axis, &source_dimension) in self.shape.iter().enumerate() {
                let destination_axis = offset + axis;
                let coordinate = if source_dimension == 1 {
                    0
                } else {
                    coordinates[destination_axis]
                };
                source_index = source_index * source_dimension + coordinate;
            }
            result.push(self.data[source_index]);
            increment_index(&mut coordinates, &destination);
        }
        checked_output(destination, result, "broadcast")
    }

    /// Checks every value for finiteness.
    ///
    /// # Errors
    ///
    /// Returns an error containing the first non-finite value index.
    pub fn validate_finite(&self) -> Result<(), TensorError> {
        for (index, value) in self.data.iter().enumerate() {
            if !value.is_finite() {
                return Err(TensorError::NonFiniteInput { index });
            }
        }
        Ok(())
    }

    /// Adds two tensors with identical shapes.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, "add", |left, right| left + right)
    }

    /// Multiplies two tensors element by element.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_op(rhs, "mul", |left, right| left * right)
    }

    /// Multiplies every value by one scalar.
    ///
    /// # Errors
    ///
    /// Returns an error when the factor produces a non-finite output value.
    pub fn scale(&self, factor: f32) -> Result<Self, TensorError> {
        let result = self
            .data
            .iter()
            .map(|value| value * factor)
            .collect::<Vec<_>>();
        checked_output(self.shape.clone(), result, "scale")
    }

    /// Multiplies two row-major rank-2 matrices.
    ///
    /// # Errors
    ///
    /// Returns an error when either tensor is not rank 2, inner dimensions do
    /// not match, or a matrix product is non-finite.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        const TILE: usize = 32;
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        if rhs.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rhs.rank(),
            });
        }
        let rows = self.shape[0];
        let inner = self.shape[1];
        let rhs_inner = rhs.shape[0];
        let columns = rhs.shape[1];
        if inner != rhs_inner {
            return Err(TensorError::MatrixShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        let result_len = rows
            .checked_mul(columns)
            .ok_or(TensorError::ElementCountOverflow)?;
        let mut result = vec![0.0; result_len];
        for row_start in (0..rows).step_by(TILE) {
            let row_end = (row_start + TILE).min(rows);
            for inner_start in (0..inner).step_by(TILE) {
                let inner_end = (inner_start + TILE).min(inner);
                for column_start in (0..columns).step_by(TILE) {
                    let column_end = (column_start + TILE).min(columns);
                    for row in row_start..row_end {
                        for index in inner_start..inner_end {
                            let left = self.data[row * inner + index];
                            let rhs_row = index * columns;
                            let result_row = row * columns;
                            for column in column_start..column_end {
                                result[result_row + column] += left * rhs.data[rhs_row + column];
                            }
                        }
                    }
                }
            }
        }
        checked_output(vec![rows, columns], result, "matmul")
    }

    /// Applies `SiLU`, also known as the swish activation, elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn silu(&self) -> Result<Self, TensorError> {
        let result = self
            .data
            .iter()
            .map(|value| *value / (1.0 + (-value).exp()))
            .collect::<Vec<_>>();
        checked_output(self.shape.clone(), result, "silu")
    }

    /// Applies interleaved rotary position embedding to `[heads, head_dim]`.
    ///
    /// Only the first `rotary_dimension` values of each head are rotated.
    /// Values after that prefix are copied unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor rank, dimensions, position, or
    /// frequency base are invalid, or an output value is non-finite.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn rotary_embedding(
        &self,
        rotary_dimension: usize,
        position: f32,
        frequency_base: f32,
    ) -> Result<Self, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::RotaryShapeMismatch {
                shape: self.shape.clone(),
            });
        }
        let heads = self.shape[0];
        let head_dim = self.shape[1];
        if head_dim == 0
            || !head_dim.is_multiple_of(2)
            || rotary_dimension == 0
            || rotary_dimension > head_dim
            || !rotary_dimension.is_multiple_of(2)
        {
            return Err(TensorError::RotaryShapeMismatch {
                shape: self.shape.clone(),
            });
        }
        if !position.is_finite() {
            return Err(TensorError::InvalidPosition);
        }
        if !frequency_base.is_finite() || frequency_base <= 0.0 {
            return Err(TensorError::InvalidFrequencyBase);
        }
        self.validate_finite()?;
        let mut result = self.data.clone();
        #[allow(clippy::cast_precision_loss)]
        let head_dim_f32 = head_dim as f32;
        for head in 0..heads {
            let start = head * head_dim;
            for pair in 0..rotary_dimension / 2 {
                #[allow(clippy::cast_precision_loss)]
                let exponent = -2.0 * pair as f32 / head_dim_f32;
                let angle = position * frequency_base.powf(exponent);
                let (sine, cosine) = angle.sin_cos();
                let first = self.data[start + pair * 2];
                let second = self.data[start + pair * 2 + 1];
                result[start + pair * 2] = first * cosine - second * sine;
                result[start + pair * 2 + 1] = first * sine + second * cosine;
            }
        }
        checked_output(self.shape.clone(), result, "rotary_embedding")
    }

    /// Applies `RMSNorm` independently to every row along the last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when epsilon is invalid or an output value is
    /// non-finite.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn rms_norm(&self, epsilon: f32) -> Result<Self, TensorError> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::InvalidEpsilon);
        }
        let width = self
            .shape
            .last()
            .copied()
            .ok_or(TensorError::ZeroDimension)?;
        let mut result = Vec::with_capacity(self.data.len());
        for row in self.data.chunks_exact(width) {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
            let inverse = (mean_square + epsilon).sqrt().recip();
            result.extend(row.iter().map(|value| value * inverse));
        }
        checked_output(self.shape.clone(), result, "rms_norm")
    }

    /// Applies weighted `RMSNorm` independently to every row along the last
    /// dimension.
    ///
    /// The weight must be a rank-1 tensor whose length equals the input's
    /// final dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when the weight shape or epsilon is invalid, an input
    /// is non-finite, or an output value is non-finite.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn rms_norm_with_weight(&self, weight: &Self, epsilon: f32) -> Result<Self, TensorError> {
        if !epsilon.is_finite() || epsilon < 0.0 {
            return Err(TensorError::InvalidEpsilon);
        }
        let width = self
            .shape
            .last()
            .copied()
            .ok_or(TensorError::ZeroDimension)?;
        if weight.shape != [width] {
            return Err(TensorError::ShapeMismatch {
                left: weight.shape.clone(),
                right: vec![width],
            });
        }
        self.validate_finite()?;
        weight.validate_finite()?;
        let mut result = Vec::with_capacity(self.data.len());
        for row in self.data.chunks_exact(width) {
            let mean_square = row.iter().map(|value| value * value).sum::<f32>() / width as f32;
            let inverse = (mean_square + epsilon).sqrt().recip();
            result.extend(
                row.iter()
                    .zip(weight.data.iter())
                    .map(|(value, scale)| value * inverse * scale),
            );
        }
        checked_output(self.shape.clone(), result, "rms_norm_with_weight")
    }

    /// Applies numerically stable softmax independently along the last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when an input or output value is non-finite.
    pub fn softmax_last_dim(&self) -> Result<Self, TensorError> {
        let width = self
            .shape
            .last()
            .copied()
            .ok_or(TensorError::ZeroDimension)?;
        let mut result = Vec::with_capacity(self.data.len());
        for row in self.data.chunks_exact(width) {
            for (index, value) in row.iter().enumerate() {
                if !value.is_finite() {
                    return Err(TensorError::NonFiniteInput { index });
                }
            }
            let maximum = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exponents = row
                .iter()
                .map(|value| (*value - maximum).exp())
                .collect::<Vec<_>>();
            let total = exponents.iter().sum::<f32>();
            result.extend(exponents.into_iter().map(|value| value / total));
        }
        checked_output(self.shape.clone(), result, "softmax")
    }

    /// Computes scaled dot-product attention for rank-4 tensors.
    ///
    /// Inputs use MLX's `[batch, heads, sequence, features]` convention.
    /// Query heads may be grouped over fewer key/value heads, matching
    /// grouped-query attention. With `causal` enabled, each query can read
    /// keys through its aligned absolute position, including a prefix when
    /// the key sequence is longer than the query sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when ranks, batch sizes, head counts, feature widths,
    /// or sequence lengths are incompatible, the scale is non-finite, an input
    /// is non-finite, or the operation produces a non-finite value.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn scaled_dot_product_attention(
        &self,
        keys: &Self,
        values: &Self,
        scale: f32,
        causal: bool,
    ) -> Result<Self, TensorError> {
        if !scale.is_finite() {
            return Err(TensorError::InvalidScale);
        }
        if self.rank() != 4 || keys.rank() != 4 || values.rank() != 4 {
            return Err(TensorError::AttentionShapeMismatch {
                queries: self.shape.clone(),
                keys: keys.shape.clone(),
                values: values.shape.clone(),
            });
        }
        if self.shape[0] != keys.shape[0]
            || self.shape[0] != values.shape[0]
            || self.shape[3] != keys.shape[3]
            || keys.shape[1] != values.shape[1]
            || !self.shape[1].is_multiple_of(keys.shape[1])
            || self.shape[2] > keys.shape[2]
        {
            return Err(TensorError::AttentionShapeMismatch {
                queries: self.shape.clone(),
                keys: keys.shape.clone(),
                values: values.shape.clone(),
            });
        }
        self.validate_finite()?;
        keys.validate_finite()?;
        values.validate_finite()?;

        let batch = self.shape[0];
        let query_heads = self.shape[1];
        let query_length = self.shape[2];
        let query_width = self.shape[3];
        let key_heads = keys.shape[1];
        let key_length = keys.shape[2];
        let value_width = values.shape[3];
        let head_repeats = query_heads / key_heads;
        let prefix = key_length - query_length;
        let output_shape = vec![batch, query_heads, query_length, value_width];
        let mut result = vec![0.0; element_count(&output_shape)?];

        for batch_index in 0..batch {
            for query_head in 0..query_heads {
                let key_head = query_head / head_repeats;
                for query_index in 0..query_length {
                    let query_start = ((batch_index * query_heads + query_head) * query_length
                        + query_index)
                        * query_width;
                    let allowed_end = if causal {
                        (prefix + query_index + 1).min(key_length)
                    } else {
                        key_length
                    };
                    if allowed_end == 0 {
                        return Err(TensorError::AttentionShapeMismatch {
                            queries: self.shape.clone(),
                            keys: keys.shape.clone(),
                            values: values.shape.clone(),
                        });
                    }
                    let mut scores = vec![f32::NEG_INFINITY; key_length];
                    let mut maximum = f32::NEG_INFINITY;
                    for (key_index, score) in scores.iter_mut().enumerate().take(allowed_end) {
                        let key_start = ((batch_index * key_heads + key_head) * key_length
                            + key_index)
                            * query_width;
                        let mut dot = 0.0_f32;
                        for offset in 0..query_width {
                            dot += self.data[query_start + offset] * keys.data[key_start + offset];
                        }
                        *score = dot * scale;
                        if !score.is_finite() {
                            return Err(TensorError::NonFiniteOutput {
                                operation: "scaled_dot_product_attention",
                                index: query_start,
                            });
                        }
                        maximum = maximum.max(*score);
                    }
                    let mut denominator = 0.0_f32;
                    for score in scores.iter_mut().take(allowed_end) {
                        *score = (*score - maximum).exp();
                        denominator += *score;
                    }
                    if !denominator.is_finite() || denominator <= 0.0 {
                        return Err(TensorError::NonFiniteOutput {
                            operation: "scaled_dot_product_attention",
                            index: query_start,
                        });
                    }
                    let output_start = ((batch_index * query_heads + query_head) * query_length
                        + query_index)
                        * value_width;
                    for (key_index, score) in scores.iter().enumerate().take(allowed_end) {
                        let weight = *score / denominator;
                        let value_start = ((batch_index * key_heads + key_head) * key_length
                            + key_index)
                            * value_width;
                        for offset in 0..value_width {
                            result[output_start + offset] +=
                                weight * values.data[value_start + offset];
                        }
                    }
                }
            }
        }
        checked_output(output_shape, result, "scaled_dot_product_attention")
    }

    fn binary_op<F>(
        &self,
        rhs: &Self,
        operation: &'static str,
        function: F,
    ) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        if self.shape != rhs.shape {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        let result = self
            .data
            .iter()
            .zip(&rhs.data)
            .map(|(left, right)| function(*left, *right))
            .collect::<Vec<_>>();
        checked_output(self.shape.clone(), result, operation)
    }
}

fn validate_shape(shape: &[usize]) -> Result<(), TensorError> {
    if shape.contains(&0) {
        return Err(TensorError::ZeroDimension);
    }
    Ok(())
}

fn element_count(shape: &[usize]) -> Result<usize, TensorError> {
    shape
        .iter()
        .try_fold(1_usize, |count, dimension| count.checked_mul(*dimension))
        .ok_or(TensorError::ElementCountOverflow)
}

fn increment_index(index: &mut [usize], shape: &[usize]) {
    for axis in (0..index.len()).rev() {
        index[axis] += 1;
        if index[axis] < shape[axis] {
            return;
        }
        index[axis] = 0;
    }
}

fn checked_output(
    shape: Vec<usize>,
    data: Vec<f32>,
    operation: &'static str,
) -> Result<Tensor, TensorError> {
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(TensorError::NonFiniteOutput { operation, index });
        }
    }
    Ok(Tensor { shape, data })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_and_reshapes_row_major_data() {
        let tensor = Tensor::from_data([2, 2], [1.0, 2.0, 3.0, 4.0])
            .unwrap()
            .reshape([4])
            .unwrap();
        assert_eq!(tensor.shape(), &[4]);
        assert_eq!(tensor.data(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn transposes_rank_two_data() {
        let tensor = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap()
            .transpose_2d()
            .unwrap();
        assert_eq!(tensor.shape(), &[3, 2]);
        assert_eq!(tensor.data(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn broadcasts_singleton_dimensions() {
        let tensor = Tensor::from_data([2, 1], [1.0, 2.0])
            .unwrap()
            .broadcast_to([2, 3])
            .unwrap();
        assert_eq!(tensor.data(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn rejects_shape_data_mismatch_and_zero_dimensions() {
        assert_eq!(
            Tensor::from_data([2, 2], [1.0, 2.0]).unwrap_err(),
            TensorError::DataLength {
                expected: 4,
                actual: 2
            }
        );
        assert_eq!(
            Tensor::zeros([2, 0]).unwrap_err(),
            TensorError::ZeroDimension
        );
    }

    #[test]
    fn matrix_multiplication_is_row_major() {
        let left = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let right = Tensor::from_data([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();
        let result = left.matmul(&right).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data(), &[58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn elementwise_ops_require_identical_shapes() {
        let left = Tensor::from_data([2], [1.0, 2.0]).unwrap();
        let right = Tensor::from_data([1, 2], [3.0, 4.0]).unwrap();
        assert!(matches!(
            left.add(&right),
            Err(TensorError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn rms_norm_normalizes_each_row() {
        let tensor = Tensor::from_data([2, 2], [3.0, 4.0, 0.0, 5.0]).unwrap();
        let normalized = tensor.rms_norm(0.0).unwrap();
        let expected_scale = 12.5_f32.sqrt().recip();
        assert!((normalized.data()[0] - 3.0 * expected_scale).abs() < 1e-6);
        assert!((normalized.data()[1] - 4.0 * expected_scale).abs() < 1e-6);
        assert_eq!(normalized.data()[2].to_bits(), 0.0_f32.to_bits());
        assert!((normalized.data()[3] - 5.0 * expected_scale).abs() < 1e-6);
    }

    #[test]
    fn weighted_rms_norm_applies_a_final_dimension_vector() {
        let tensor = Tensor::from_data([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let weight = Tensor::from_data([2], [2.0, 3.0]).unwrap();
        let normalized = tensor.rms_norm_with_weight(&weight, 0.0).unwrap();
        let expected = [
            2.0 / 2.5_f32.sqrt(),
            6.0 / 2.5_f32.sqrt(),
            6.0 / 12.5_f32.sqrt(),
            12.0 / 12.5_f32.sqrt(),
        ];
        for (actual, expected) in normalized.data().iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn softmax_is_stable_and_normalized() {
        let tensor =
            Tensor::from_data([2, 3], [1000.0, 1001.0, 1002.0, -1000.0, -1001.0, -1002.0]).unwrap();
        let probabilities = tensor.softmax_last_dim().unwrap();
        let (rows, remainder) = probabilities.data().as_chunks::<3>();
        assert!(remainder.is_empty());
        for row in rows {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        }
        assert!(probabilities.data().iter().all(|value| value.is_finite()));
    }

    #[test]
    fn attention_supports_grouped_heads_and_causal_prefixes() {
        let queries = Tensor::from_data([1, 2, 2, 1], [0.0; 4]).unwrap();
        let keys = Tensor::from_data([1, 1, 3, 1], [0.0; 3]).unwrap();
        let values = Tensor::from_data([1, 1, 3, 1], [1.0, 3.0, 5.0]).unwrap();
        let output = queries
            .scaled_dot_product_attention(&keys, &values, 1.0, true)
            .unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 1]);
        assert_eq!(output.data(), &[2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn attention_rejects_incompatible_shapes_and_scale() {
        let queries = Tensor::from_data([1, 1, 1, 2], [0.0, 0.0]).unwrap();
        let keys = Tensor::from_data([1, 1, 1, 1], [0.0]).unwrap();
        let values = Tensor::from_data([1, 1, 1, 1], [0.0]).unwrap();
        assert!(matches!(
            queries.scaled_dot_product_attention(&keys, &values, 1.0, false),
            Err(TensorError::AttentionShapeMismatch { .. })
        ));
        assert_eq!(
            queries.scaled_dot_product_attention(&queries, &values, f32::NAN, false),
            Err(TensorError::InvalidScale)
        );
    }

    #[test]
    fn rejects_non_finite_results() {
        let tensor = Tensor::from_data([1], [f32::MAX]).unwrap();
        assert_eq!(
            tensor.scale(2.0).unwrap_err(),
            TensorError::NonFiniteOutput {
                operation: "scale",
                index: 0
            }
        );
    }
}
