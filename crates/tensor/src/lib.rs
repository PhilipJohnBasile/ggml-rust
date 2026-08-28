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
    ZeroDimension,
    InvalidEpsilon,
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
            Self::ZeroDimension => formatter.write_str("tensor dimensions must be nonzero"),
            Self::InvalidEpsilon => {
                formatter.write_str("RMSNorm epsilon must be finite and nonnegative")
            }
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
        let mut result = vec![0.0; rows * columns];
        for row in 0..rows {
            for column in 0..columns {
                let mut value = 0.0_f32;
                for index in 0..inner {
                    value += self.data[row * inner + index] * rhs.data[index * columns + column];
                }
                result[row * columns + column] = value;
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

    /// Applies `RMSNorm` independently to every row along the last dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when epsilon is invalid or an output value is
    /// non-finite.
    #[allow(clippy::cast_precision_loss)]
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
