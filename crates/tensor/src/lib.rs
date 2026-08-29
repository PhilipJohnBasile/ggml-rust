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
    InvalidPermutation {
        axes: Vec<usize>,
        rank: usize,
    },
    AttentionShapeMismatch {
        queries: Vec<usize>,
        keys: Vec<usize>,
        values: Vec<usize>,
    },
    ZeroDimension,
    InvalidEpsilon,
    InvalidScale,
    InvalidClamp,
    InvalidSlice(&'static str),
    InvalidAxis {
        axis: usize,
        rank: usize,
    },
    IndexOutOfBounds {
        index: usize,
        upper_bound: usize,
    },
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
            Self::InvalidPermutation { axes, rank } => {
                write!(
                    formatter,
                    "axes {axes:?} are not a permutation of rank {rank}"
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
            Self::InvalidClamp => formatter.write_str("clamp bounds must be finite and ordered"),
            Self::InvalidSlice(reason) => write!(formatter, "invalid slice: {reason}"),
            Self::InvalidAxis { axis, rank } => {
                write!(formatter, "axis {axis} is outside tensor rank {rank}")
            }
            Self::IndexOutOfBounds { index, upper_bound } => write!(
                formatter,
                "row index {index} is outside the table range 0..{upper_bound}"
            ),
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

/// Rotary position scaling shared by CPU tensors and graph execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RotaryScaling {
    /// Use the unscaled token position.
    None,
    /// Divide rotary positions by a linear factor.
    Linear { factor: f32 },
    /// Apply `YaRN` frequency interpolation and magnitude scaling.
    Yarn {
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        original_context_length: usize,
        attention_factor: f32,
        ext_factor: f32,
    },
}

impl RotaryScaling {
    #[allow(clippy::cast_precision_loss)]
    fn phase(
        self,
        position: f32,
        pair: usize,
        head_dim: f32,
        rotary_dimension: usize,
        frequency_base: f32,
    ) -> Result<(f32, f32), TensorError> {
        let exponent = -2.0 * pair as f32 / head_dim;
        let theta_extrap = position * frequency_base.powf(exponent);
        let result = match self {
            Self::None => (theta_extrap, 1.0),
            Self::Linear { factor } => {
                if !factor.is_finite() || factor <= 0.0 {
                    return Err(TensorError::InvalidPosition);
                }
                (theta_extrap / factor, 1.0)
            }
            Self::Yarn {
                factor,
                beta_fast,
                beta_slow,
                original_context_length,
                attention_factor,
                ext_factor,
            } => {
                if !factor.is_finite()
                    || factor <= 0.0
                    || !beta_fast.is_finite()
                    || beta_fast <= 0.0
                    || !beta_slow.is_finite()
                    || beta_slow <= 0.0
                    || original_context_length == 0
                    || !attention_factor.is_finite()
                    || attention_factor <= 0.0
                    || !ext_factor.is_finite()
                    || ext_factor < 0.0
                {
                    return Err(TensorError::InvalidPosition);
                }
                let rotary_dimension_f32 = rotary_dimension as f32;
                let low = (rotary_dimension_f32
                    * ((original_context_length as f32)
                        / (beta_fast * 2.0 * std::f32::consts::PI))
                        .ln()
                    / (2.0 * frequency_base.ln()))
                .floor()
                .max(0.0);
                let high = (rotary_dimension_f32
                    * ((original_context_length as f32)
                        / (beta_slow * 2.0 * std::f32::consts::PI))
                        .ln()
                    / (2.0 * frequency_base.ln()))
                .ceil()
                .min((rotary_dimension.saturating_sub(1)) as f32);
                let ramp = (1.0 - ((pair * 2) as f32 - low) / (0.001_f32.max(high - low)))
                    .clamp(0.0, 1.0)
                    * ext_factor;
                let theta_interp = theta_extrap / factor;
                let angle = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
                let magnitude = attention_factor * (1.0 + 0.1 * factor.ln());
                (angle, magnitude)
            }
        };
        if result.0.is_finite() && result.1.is_finite() {
            Ok(result)
        } else {
            Err(TensorError::NonFiniteOutput {
                operation: "rotary_embedding",
                index: pair,
            })
        }
    }
}

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

    /// Permutes tensor axes by copying values into row-major order.
    ///
    /// `axes[i]` identifies the source axis that becomes output axis `i`.
    /// Arbitrary ranks are supported, including the scalar rank zero case.
    ///
    /// # Errors
    ///
    /// Returns an error when `axes` is not a permutation of the tensor rank or
    /// an output value is non-finite.
    pub fn permute(&self, axes: &[usize]) -> Result<Self, TensorError> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(TensorError::InvalidPermutation {
                axes: axes.to_vec(),
                rank,
            });
        }
        let mut seen = vec![false; rank];
        for &axis in axes {
            if axis >= rank || seen[axis] {
                return Err(TensorError::InvalidPermutation {
                    axes: axes.to_vec(),
                    rank,
                });
            }
            seen[axis] = true;
        }
        if rank == 0 {
            return Ok(self.clone());
        }
        let output_shape = axes
            .iter()
            .map(|&axis| self.shape[axis])
            .collect::<Vec<_>>();
        let output_len = element_count(&output_shape)?;
        let mut result = Vec::with_capacity(output_len);
        let mut coordinates = vec![0_usize; rank];
        for _ in 0..output_len {
            let mut source_coordinates = vec![0_usize; rank];
            for (output_axis, &source_axis) in axes.iter().enumerate() {
                source_coordinates[source_axis] = coordinates[output_axis];
            }
            let mut source_index = 0_usize;
            for (axis, &dimension) in self.shape.iter().enumerate() {
                source_index = source_index * dimension + source_coordinates[axis];
            }
            result.push(self.data[source_index]);
            increment_index(&mut coordinates, &output_shape);
        }
        checked_output(output_shape, result, "permute")
    }

    /// Slices a tensor with positive strides along selected axes.
    ///
    /// Axes not listed are copied in full with stride one. The four index
    /// slices must have equal lengths, and the result remains row-major.
    ///
    /// # Errors
    ///
    /// Returns an error when bounds, axes, strides, or the resulting shape are
    /// invalid.
    pub fn slice(
        &self,
        starts: &[usize],
        ends: &[usize],
        axes: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        let (output_shape, normalized_starts, normalized_strides) =
            normalize_slice(&self.shape, starts, ends, axes, strides)?;
        let output_len = element_count(&output_shape)?;
        let rank = self.rank();
        let mut result = Vec::with_capacity(output_len);
        let mut coordinates = vec![0_usize; rank];
        for _ in 0..output_len {
            let mut source_index = 0_usize;
            for axis in 0..rank {
                let coordinate = normalized_starts[axis]
                    .checked_add(
                        coordinates[axis]
                            .checked_mul(normalized_strides[axis])
                            .ok_or(TensorError::ElementCountOverflow)?,
                    )
                    .ok_or(TensorError::ElementCountOverflow)?;
                source_index = source_index
                    .checked_mul(self.shape[axis])
                    .and_then(|value| value.checked_add(coordinate))
                    .ok_or(TensorError::ElementCountOverflow)?;
            }
            result.push(self.data[source_index]);
            increment_index(&mut coordinates, &output_shape);
        }
        checked_output(output_shape, result, "slice")
    }

    /// Returns a copy with a positive-stride slice replaced by `update`.
    ///
    /// The update shape must equal the shape selected by the slice. The source
    /// tensor is not modified in place.
    ///
    /// # Errors
    ///
    /// Returns an error when bounds, axes, strides, or update shape are
    /// invalid.
    pub fn slice_update(
        &self,
        update: &Self,
        starts: &[usize],
        ends: &[usize],
        axes: &[usize],
        strides: &[usize],
    ) -> Result<Self, TensorError> {
        let (expected_shape, normalized_starts, normalized_strides) =
            normalize_slice(&self.shape, starts, ends, axes, strides)?;
        if update.shape != expected_shape {
            return Err(TensorError::ShapeMismatch {
                left: update.shape.clone(),
                right: expected_shape,
            });
        }
        let mut result = self.data.clone();
        let rank = self.rank();
        let mut coordinates = vec![0_usize; rank];
        for slot in &mut result {
            let mut update_index = 0_usize;
            let mut selected = true;
            for axis in 0..rank {
                let coordinate = coordinates[axis];
                if coordinate < normalized_starts[axis] {
                    selected = false;
                    break;
                }
                let relative = coordinate - normalized_starts[axis];
                if !relative.is_multiple_of(normalized_strides[axis]) {
                    selected = false;
                    break;
                }
                let update_coordinate = relative / normalized_strides[axis];
                if update_coordinate >= expected_shape[axis] {
                    selected = false;
                    break;
                }
                update_index = update_index
                    .checked_mul(expected_shape[axis])
                    .and_then(|value| value.checked_add(update_coordinate))
                    .ok_or(TensorError::ElementCountOverflow)?;
            }
            if selected {
                *slot = update.data[update_index];
            }
            increment_index(&mut coordinates, &self.shape);
        }
        checked_output(self.shape.clone(), result, "slice_update")
    }

    /// Gathers rows from a rank-2 table by integer index.
    ///
    /// The returned tensor has shape `[indices.len(), self.shape[1]]` and
    /// preserves the table's row-major column order. This is the checked
    /// equivalent of GGML `GET_ROWS` and an embedding lookup.
    ///
    /// # Errors
    ///
    /// Returns an error when the table is not rank 2, no indices are supplied,
    /// an index is outside the table, or an output value is non-finite.
    pub fn gather_rows(&self, indices: &[usize]) -> Result<Self, TensorError> {
        if self.rank() != 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        if indices.is_empty() {
            return Err(TensorError::ZeroDimension);
        }
        let rows = self.shape[0];
        let columns = self.shape[1];
        let output_len = indices
            .len()
            .checked_mul(columns)
            .ok_or(TensorError::ElementCountOverflow)?;
        let mut result = Vec::with_capacity(output_len);
        for &index in indices {
            if index >= rows {
                return Err(TensorError::IndexOutOfBounds {
                    index,
                    upper_bound: rows,
                });
            }
            let start = index * columns;
            result.extend_from_slice(&self.data[start..start + columns]);
        }
        checked_output(vec![indices.len(), columns], result, "gather_rows")
    }

    /// Concatenates two tensors along one axis.
    ///
    /// Both tensors must have the same rank and identical dimensions on every
    /// axis except `axis`.
    ///
    /// # Errors
    ///
    /// Returns an error when the axis or input shapes are incompatible, the
    /// output shape overflows, or an output value is non-finite.
    pub fn concatenate(&self, rhs: &Self, axis: usize) -> Result<Self, TensorError> {
        if self.rank() != rhs.rank() || axis >= self.rank() {
            return Err(TensorError::ShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        for dimension in 0..self.rank() {
            if dimension != axis && self.shape[dimension] != rhs.shape[dimension] {
                return Err(TensorError::ShapeMismatch {
                    left: self.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }
        }
        let concatenated = self.shape[axis]
            .checked_add(rhs.shape[axis])
            .ok_or(TensorError::ElementCountOverflow)?;
        let mut output_shape = self.shape.clone();
        output_shape[axis] = concatenated;
        let output_len = element_count(&output_shape)?;
        let mut result = Vec::with_capacity(output_len);
        let mut coordinates = vec![0_usize; self.rank()];
        for _ in 0..output_len {
            let mut source_coordinates = coordinates.clone();
            let source = if coordinates[axis] < self.shape[axis] {
                &self.data
            } else {
                source_coordinates[axis] -= self.shape[axis];
                &rhs.data
            };
            let source_shape = if coordinates[axis] < self.shape[axis] {
                &self.shape
            } else {
                &rhs.shape
            };
            let mut source_index = 0_usize;
            for (dimension, &size) in source_shape.iter().enumerate() {
                source_index = source_index * size + source_coordinates[dimension];
            }
            result.push(source[source_index]);
            increment_index(&mut coordinates, &output_shape);
        }
        checked_output(output_shape, result, "concatenate")
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

    /// Adds two tensors with right-aligned singleton broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn add(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(rhs, "add", |left, right| left + right)
    }

    /// Multiplies two tensors element by element with right-aligned
    /// singleton broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn mul(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(rhs, "mul", |left, right| left * right)
    }

    /// Subtracts two tensors element by element with right-aligned singleton
    /// broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn sub(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(rhs, "sub", |left, right| left - right)
    }

    /// Divides two tensors element by element with right-aligned singleton
    /// broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when shapes differ or an output value is non-finite.
    pub fn div(&self, rhs: &Self) -> Result<Self, TensorError> {
        self.binary_broadcast(rhs, "div", |left, right| left / right)
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

    /// Negates every value elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn neg(&self) -> Result<Self, TensorError> {
        self.unary_op("neg", |value| -value)
    }

    /// Computes the absolute value elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn abs(&self) -> Result<Self, TensorError> {
        self.unary_op("abs", f32::abs)
    }

    /// Computes the square elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn sqr(&self) -> Result<Self, TensorError> {
        self.unary_op("sqr", |value| value * value)
    }

    /// Computes the square root elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an input is negative or an output value is
    /// non-finite.
    pub fn sqrt(&self) -> Result<Self, TensorError> {
        self.unary_op("sqrt", f32::sqrt)
    }

    /// Computes the natural exponential elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn exp(&self) -> Result<Self, TensorError> {
        self.unary_op("exp", f32::exp)
    }

    /// Computes the natural logarithm elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an input is non-positive or an output value is
    /// non-finite.
    pub fn log(&self) -> Result<Self, TensorError> {
        self.unary_op("log", f32::ln)
    }

    /// Computes the sine elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn sin(&self) -> Result<Self, TensorError> {
        self.unary_op("sin", f32::sin)
    }

    /// Computes the cosine elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn cos(&self) -> Result<Self, TensorError> {
        self.unary_op("cos", f32::cos)
    }

    /// Computes the hyperbolic tangent elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn tanh(&self) -> Result<Self, TensorError> {
        self.unary_op("tanh", f32::tanh)
    }

    /// Computes the logistic sigmoid elementwise.
    ///
    /// # Errors
    ///
    /// Returns an error when an output value is non-finite.
    pub fn sigmoid(&self) -> Result<Self, TensorError> {
        self.unary_op("sigmoid", |value| 1.0 / (1.0 + (-value).exp()))
    }

    /// Clamps every value to the inclusive interval `[minimum, maximum]`.
    ///
    /// # Errors
    ///
    /// Returns an error when bounds are non-finite, the interval is inverted,
    /// or an output value is non-finite.
    pub fn clamp(&self, minimum: f32, maximum: f32) -> Result<Self, TensorError> {
        if !minimum.is_finite() || !maximum.is_finite() || minimum > maximum {
            return Err(TensorError::InvalidClamp);
        }
        let result = self
            .data
            .iter()
            .map(|value| value.clamp(minimum, maximum))
            .collect::<Vec<_>>();
        checked_output(self.shape.clone(), result, "clamp")
    }

    /// Multiplies row-major matrices, including broadcasted batch dimensions.
    ///
    /// The final two dimensions are interpreted as matrix dimensions. Leading
    /// dimensions are broadcast with the same right-aligned singleton rules as
    /// elementwise arithmetic.
    ///
    /// # Errors
    ///
    /// Returns an error when either tensor has rank below two, inner
    /// dimensions or batch dimensions do not match, or a matrix product is
    /// non-finite.
    pub fn matmul(&self, rhs: &Self) -> Result<Self, TensorError> {
        if self.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: self.rank(),
            });
        }
        if rhs.rank() < 2 {
            return Err(TensorError::RankMismatch {
                expected: 2,
                actual: rhs.rank(),
            });
        }
        let rows = self.shape[self.rank() - 2];
        let inner = self.shape[self.rank() - 1];
        let rhs_inner = rhs.shape[rhs.rank() - 2];
        let columns = rhs.shape[rhs.rank() - 1];
        if inner != rhs_inner {
            return Err(TensorError::MatrixShapeMismatch {
                left: self.shape.clone(),
                right: rhs.shape.clone(),
            });
        }
        let batch_rank = self
            .rank()
            .saturating_sub(2)
            .max(rhs.rank().saturating_sub(2));
        let left_batch_offset = batch_rank - (self.rank() - 2);
        let right_batch_offset = batch_rank - (rhs.rank() - 2);
        let mut output_shape = Vec::with_capacity(batch_rank + 2);
        for axis in 0..batch_rank {
            let left_dimension = if axis < left_batch_offset {
                1
            } else {
                self.shape[axis - left_batch_offset]
            };
            let right_dimension = if axis < right_batch_offset {
                1
            } else {
                rhs.shape[axis - right_batch_offset]
            };
            if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
                return Err(TensorError::MatrixShapeMismatch {
                    left: self.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }
            output_shape.push(left_dimension.max(right_dimension));
        }
        output_shape.extend([rows, columns]);
        let output_len = element_count(&output_shape)?;
        let mut result = vec![0.0; output_len];
        let batch_count = output_shape[..batch_rank]
            .iter()
            .copied()
            .product::<usize>();
        let mut batch_coordinates = vec![0_usize; batch_rank];
        for batch_index in 0..batch_count {
            let mut left_batch_index = 0_usize;
            for (axis, &dimension) in self.shape[..self.rank() - 2].iter().enumerate() {
                let output_axis = left_batch_offset + axis;
                let coordinate = if dimension == 1 {
                    0
                } else {
                    batch_coordinates[output_axis]
                };
                left_batch_index = left_batch_index * dimension + coordinate;
            }
            let mut right_batch_index = 0_usize;
            for (axis, &dimension) in rhs.shape[..rhs.rank() - 2].iter().enumerate() {
                let output_axis = right_batch_offset + axis;
                let coordinate = if dimension == 1 {
                    0
                } else {
                    batch_coordinates[output_axis]
                };
                right_batch_index = right_batch_index * dimension + coordinate;
            }
            let left_base = left_batch_index * rows * inner;
            let right_base = right_batch_index * inner * columns;
            let output_base = batch_index * rows * columns;
            for row in 0..rows {
                for column in 0..columns {
                    let mut sum = 0.0_f32;
                    for index in 0..inner {
                        sum += self.data[left_base + row * inner + index]
                            * rhs.data[right_base + index * columns + column];
                    }
                    result[output_base + row * columns + column] = sum;
                }
            }
            increment_index(&mut batch_coordinates, &output_shape[..batch_rank]);
        }
        checked_output(output_shape, result, "matmul")
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
        self.rotary_embedding_with_scaling(
            rotary_dimension,
            position,
            frequency_base,
            RotaryScaling::None,
        )
    }

    /// Applies interleaved rotary position embedding with linear or `YaRN`
    /// scaling to `[heads, head_dim]`.
    ///
    /// Only the first `rotary_dimension` values of each head are rotated.
    /// Scaling and magnitude are applied per rotary pair.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor rank, dimensions, scaling
    /// parameters, position, or frequency base are invalid, or an output
    /// value is non-finite.
    #[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
    pub fn rotary_embedding_with_scaling(
        &self,
        rotary_dimension: usize,
        position: f32,
        frequency_base: f32,
        scaling: RotaryScaling,
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
        let head_dim_f32 = head_dim as f32;
        for head in 0..heads {
            let start = head * head_dim;
            for pair in 0..rotary_dimension / 2 {
                let (angle, magnitude) = scaling.phase(
                    position,
                    pair,
                    head_dim_f32,
                    rotary_dimension,
                    frequency_base,
                )?;
                let (sine, cosine) = angle.sin_cos();
                let first = self.data[start + pair * 2];
                let second = self.data[start + pair * 2 + 1];
                result[start + pair * 2] = (first * cosine - second * sine) * magnitude;
                result[start + pair * 2 + 1] = (first * sine + second * cosine) * magnitude;
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

    /// Sums values independently along the final dimension.
    ///
    /// The output removes the final dimension. A rank-1 input therefore
    /// produces a one-element rank-1 tensor rather than a scalar tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is scalar or an output value is
    /// non-finite.
    pub fn sum_last_dim(&self) -> Result<Self, TensorError> {
        let axis = self
            .shape
            .len()
            .checked_sub(1)
            .ok_or(TensorError::ZeroDimension)?;
        self.reduce_axis(axis, false, false)
    }

    /// Computes the arithmetic mean independently along the final dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is scalar or an output value is
    /// non-finite.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean_last_dim(&self) -> Result<Self, TensorError> {
        let axis = self
            .shape
            .len()
            .checked_sub(1)
            .ok_or(TensorError::ZeroDimension)?;
        self.reduce_axis(axis, false, true)
    }

    /// Sums values along one axis.
    ///
    /// When `keepdims` is true the reduced axis remains as a singleton
    /// dimension. A rank-1 reduction without `keepdims` uses `[1]` because
    /// this checked tensor representation does not permit scalar shapes.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is scalar, the axis is invalid, or an
    /// output value is non-finite.
    pub fn sum(&self, axis: usize, keepdims: bool) -> Result<Self, TensorError> {
        self.reduce_axis(axis, keepdims, false)
    }

    /// Computes the arithmetic mean along one axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the tensor is scalar, the axis is invalid, or an
    /// output value is non-finite.
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self, axis: usize, keepdims: bool) -> Result<Self, TensorError> {
        self.reduce_axis(axis, keepdims, true)
    }

    /// Computes an inclusive cumulative sum along one axis.
    ///
    /// When `reverse` is true, accumulation proceeds from the end of the axis
    /// toward the beginning while preserving the original output layout.
    ///
    /// # Errors
    ///
    /// Returns an error when `axis` is outside the tensor rank or an internal
    /// shape calculation overflows.
    pub fn cumsum(&self, axis: usize, reverse: bool) -> Result<Self, TensorError> {
        let rank = self.shape.len();
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }
        let axis_width = self.shape[axis];
        let inner = self.shape[axis + 1..]
            .iter()
            .try_fold(1_usize, |value, &dimension| value.checked_mul(dimension))
            .ok_or(TensorError::ElementCountOverflow)?;
        let block = axis_width
            .checked_mul(inner)
            .ok_or(TensorError::ElementCountOverflow)?;
        let outer = self.len() / block;
        let mut output = vec![0.0; self.len()];
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let mut total = 0.0_f32;
                if reverse {
                    for position in (0..axis_width).rev() {
                        let index = (outer_index * axis_width + position) * inner + inner_index;
                        total += self.data[index];
                        output[index] = total;
                    }
                } else {
                    for position in 0..axis_width {
                        let index = (outer_index * axis_width + position) * inner + inner_index;
                        total += self.data[index];
                        output[index] = total;
                    }
                }
            }
        }
        Self::from_data(self.shape.clone(), output)
    }

    #[allow(clippy::cast_precision_loss)]
    fn reduce_axis(&self, axis: usize, keepdims: bool, mean: bool) -> Result<Self, TensorError> {
        let rank = self.shape.len();
        if rank == 0 {
            return Err(TensorError::ZeroDimension);
        }
        if axis >= rank {
            return Err(TensorError::InvalidAxis { axis, rank });
        }
        let width = self.shape[axis];
        let mut output_shape = self.shape.clone();
        if keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
            if output_shape.is_empty() {
                output_shape.push(1);
            }
        }
        let output_len = element_count(&output_shape)?;
        let mut result = Vec::with_capacity(output_len);
        let mut output_coordinates = vec![0_usize; output_shape.len()];
        for _ in 0..output_len {
            let mut total = 0.0_f32;
            for reduced_coordinate in 0..width {
                let mut source_coordinates = vec![0_usize; rank];
                if keepdims {
                    for (source_axis, coordinate) in source_coordinates.iter_mut().enumerate() {
                        *coordinate = if source_axis == axis {
                            reduced_coordinate
                        } else {
                            output_coordinates[source_axis]
                        };
                    }
                } else {
                    let mut output_axis = 0;
                    for (source_axis, coordinate) in source_coordinates.iter_mut().enumerate() {
                        *coordinate = if source_axis == axis {
                            reduced_coordinate
                        } else {
                            let coordinate = if output_shape.len() == 1 && rank == 1 {
                                0
                            } else {
                                output_coordinates[output_axis]
                            };
                            output_axis += 1;
                            coordinate
                        };
                    }
                }
                let mut source_index = 0_usize;
                for (source_axis, &dimension) in self.shape.iter().enumerate() {
                    source_index = source_index * dimension + source_coordinates[source_axis];
                }
                total += self.data[source_index];
            }
            result.push(if mean { total / width as f32 } else { total });
            increment_index(&mut output_coordinates, &output_shape);
        }
        checked_output(output_shape, result, if mean { "mean" } else { "sum" })
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

    fn binary_broadcast<F>(
        &self,
        rhs: &Self,
        operation: &'static str,
        function: F,
    ) -> Result<Self, TensorError>
    where
        F: Fn(f32, f32) -> f32,
    {
        let rank = self.rank().max(rhs.rank());
        let left_offset = rank - self.rank();
        let right_offset = rank - rhs.rank();
        let mut output_shape = Vec::with_capacity(rank);
        for axis in 0..rank {
            let left_dimension = if axis < left_offset {
                1
            } else {
                self.shape[axis - left_offset]
            };
            let right_dimension = if axis < right_offset {
                1
            } else {
                rhs.shape[axis - right_offset]
            };
            if left_dimension != right_dimension && left_dimension != 1 && right_dimension != 1 {
                return Err(TensorError::ShapeMismatch {
                    left: self.shape.clone(),
                    right: rhs.shape.clone(),
                });
            }
            output_shape.push(left_dimension.max(right_dimension));
        }
        let output_len = element_count(&output_shape)?;
        let mut result = Vec::with_capacity(output_len);
        let mut coordinates = vec![0_usize; rank];
        for _ in 0..output_len {
            let mut left_index = 0_usize;
            for (axis, &dimension) in self.shape.iter().enumerate() {
                let coordinate = if dimension == 1 {
                    0
                } else {
                    coordinates[left_offset + axis]
                };
                left_index = left_index * dimension + coordinate;
            }
            let mut right_index = 0_usize;
            for (axis, &dimension) in rhs.shape.iter().enumerate() {
                let coordinate = if dimension == 1 {
                    0
                } else {
                    coordinates[right_offset + axis]
                };
                right_index = right_index * dimension + coordinate;
            }
            result.push(function(self.data[left_index], rhs.data[right_index]));
            increment_index(&mut coordinates, &output_shape);
        }
        checked_output(output_shape, result, operation)
    }

    fn unary_op<F>(&self, operation: &'static str, function: F) -> Result<Self, TensorError>
    where
        F: Fn(f32) -> f32,
    {
        let result = self.data.iter().copied().map(function).collect::<Vec<_>>();
        checked_output(self.shape.clone(), result, operation)
    }
}

fn validate_shape(shape: &[usize]) -> Result<(), TensorError> {
    if shape.contains(&0) {
        return Err(TensorError::ZeroDimension);
    }
    Ok(())
}

type SliceNormalization = (Vec<usize>, Vec<usize>, Vec<usize>);

fn normalize_slice(
    shape: &[usize],
    starts: &[usize],
    ends: &[usize],
    axes: &[usize],
    strides: &[usize],
) -> Result<SliceNormalization, TensorError> {
    validate_shape(shape)?;
    if starts.len() != ends.len() || starts.len() != axes.len() || starts.len() != strides.len() {
        return Err(TensorError::InvalidSlice(
            "index arrays must have equal lengths",
        ));
    }
    let mut output_shape = shape.to_vec();
    let mut normalized_starts = vec![0_usize; shape.len()];
    let mut normalized_strides = vec![1_usize; shape.len()];
    let mut seen = vec![false; shape.len()];
    for (((&start, &end), &axis), &stride) in starts.iter().zip(ends).zip(axes).zip(strides) {
        if axis >= shape.len() || seen[axis] || stride == 0 || start > end || end > shape[axis] {
            return Err(TensorError::InvalidSlice(
                "bounds, axes, or strides are invalid",
            ));
        }
        let span = end - start;
        let count = span
            .checked_add(stride - 1)
            .ok_or(TensorError::ElementCountOverflow)?
            / stride;
        if count == 0 {
            return Err(TensorError::InvalidSlice(
                "slice output dimensions must be nonzero",
            ));
        }
        seen[axis] = true;
        normalized_starts[axis] = start;
        normalized_strides[axis] = stride;
        output_shape[axis] = count;
    }
    Ok((output_shape, normalized_starts, normalized_strides))
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
    #[allow(clippy::cast_precision_loss)]
    fn permutes_arbitrary_rank_data() {
        let tensor = Tensor::from_data(
            [2, 3, 2],
            (0..12).map(|value| value as f32).collect::<Vec<_>>(),
        )
        .unwrap();
        let permuted = tensor.permute(&[2, 0, 1]).unwrap();
        assert_eq!(permuted.shape(), &[2, 2, 3]);
        assert_eq!(
            permuted.data(),
            &[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0]
        );
        assert_eq!(
            tensor.permute(&[0, 0, 1]).unwrap_err(),
            TensorError::InvalidPermutation {
                axes: vec![0, 0, 1],
                rank: 3,
            }
        );
    }

    #[test]
    fn slices_and_updates_strided_regions() {
        let tensor = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sliced = tensor.slice(&[1], &[3], &[1], &[1]).unwrap();
        assert_eq!(sliced.shape(), &[2, 2]);
        assert_eq!(sliced.data(), &[2.0, 3.0, 5.0, 6.0]);
        let strided = tensor.slice(&[0], &[3], &[1], &[2]).unwrap();
        assert_eq!(strided.data(), &[1.0, 3.0, 4.0, 6.0]);
        let update = Tensor::from_data([2, 2], [9.0, 8.0, 7.0, 6.0]).unwrap();
        let updated = tensor
            .slice_update(&update, &[1], &[3], &[1], &[1])
            .unwrap();
        assert_eq!(updated.data(), &[1.0, 9.0, 8.0, 4.0, 7.0, 6.0]);
        assert_eq!(
            tensor.slice(&[0], &[0], &[1], &[1]).unwrap_err(),
            TensorError::InvalidSlice("slice output dimensions must be nonzero")
        );
    }

    #[test]
    fn gathers_rows_in_requested_order() {
        let table = Tensor::from_data([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let gathered = table.gather_rows(&[2, 0, 2]).unwrap();
        assert_eq!(gathered.shape(), &[3, 2]);
        assert_eq!(gathered.data(), &[5.0, 6.0, 1.0, 2.0, 5.0, 6.0]);
        assert_eq!(
            table.gather_rows(&[3]).unwrap_err(),
            TensorError::IndexOutOfBounds {
                index: 3,
                upper_bound: 3,
            }
        );
    }

    #[test]
    fn concatenates_along_each_supported_axis() {
        let left = Tensor::from_data([2, 1], [1.0, 2.0]).unwrap();
        let right = Tensor::from_data([2, 2], [3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            left.concatenate(&right, 1).unwrap().data(),
            &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]
        );
        let upper = Tensor::from_data([1, 2], [7.0, 8.0]).unwrap();
        let lower = Tensor::from_data([2, 2], [9.0, 10.0, 11.0, 12.0]).unwrap();
        assert_eq!(
            upper.concatenate(&lower, 0).unwrap().data(),
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
        );
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
    fn rotary_embedding_applies_yarn_scaling() {
        let tensor = Tensor::from_data([1, 2], [1.0, 2.0]).unwrap();
        let output = tensor
            .rotary_embedding_with_scaling(
                2,
                0.0,
                10_000.0,
                RotaryScaling::Yarn {
                    factor: 4.0,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    original_context_length: 32,
                    attention_factor: 1.0,
                    ext_factor: 1.0,
                },
            )
            .unwrap();
        let magnitude = 1.0 + 0.1 * 4.0_f32.ln();
        assert!((output.data()[0] - magnitude).abs() < 1.0e-6);
        assert!((output.data()[1] - 2.0 * magnitude).abs() < 1.0e-6);
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
    fn matrix_multiplication_broadcasts_batch_dimensions() {
        let left = Tensor::from_data([2, 2, 1], [1.0, 2.0, 3.0, 4.0]).unwrap();
        let right = Tensor::from_data([1, 1, 2], [5.0, 6.0]).unwrap();
        let result = left.matmul(&right).unwrap();
        assert_eq!(result.shape(), &[2, 2, 2]);
        assert_eq!(
            result.data(),
            &[5.0, 6.0, 10.0, 12.0, 15.0, 18.0, 20.0, 24.0]
        );
    }

    #[test]
    fn elementwise_ops_broadcast_compatible_shapes() {
        let left = Tensor::from_data([2, 1], [1.0, 2.0]).unwrap();
        let right = Tensor::from_data([3, 2], [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        assert!(matches!(
            left.add(&right),
            Err(TensorError::ShapeMismatch { .. })
        ));
        let broadcast = left.add(&Tensor::from_data([1, 2], [3.0, 4.0]).unwrap());
        assert_eq!(broadcast.unwrap().data(), &[4.0, 5.0, 5.0, 6.0]);
        assert_eq!(left.add(&Tensor::scalar(2.0)).unwrap().data(), &[3.0, 4.0]);
    }

    #[test]
    fn extended_elementwise_ops_match_scalar_math() {
        let tensor = Tensor::from_data([4], [-2.0, 0.5, 1.0, 4.0]).unwrap();
        assert_eq!(
            tensor.sub(&Tensor::scalar(1.0)).unwrap().data(),
            &[-3.0, -0.5, 0.0, 3.0]
        );
        let rhs = Tensor::from_data([4], [1.0, 0.5, 2.0, 2.0]).unwrap();
        assert_eq!(tensor.sub(&rhs).unwrap().data(), &[-3.0, 0.0, -1.0, 2.0]);
        assert_eq!(tensor.div(&rhs).unwrap().data(), &[-2.0, 1.0, 0.5, 2.0]);
        assert_eq!(tensor.neg().unwrap().data(), &[2.0, -0.5, -1.0, -4.0]);
        assert_eq!(tensor.abs().unwrap().data(), &[2.0, 0.5, 1.0, 4.0]);
        assert_eq!(tensor.sqr().unwrap().data(), &[4.0, 0.25, 1.0, 16.0]);
        assert_eq!(
            tensor.clamp(-1.0, 1.0).unwrap().data(),
            &[-1.0, 0.5, 1.0, 1.0]
        );
        let positive = Tensor::from_data([3], [0.5, 1.0, 4.0]).unwrap();
        assert!((positive.exp().unwrap().data()[0] - 0.5_f32.exp()).abs() < 1.0e-6);
        assert!((positive.log().unwrap().data()[1] - 0.0).abs() < 1.0e-6);
        let angles = Tensor::from_data(
            [3],
            [0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
        )
        .unwrap();
        let sine = angles.sin().unwrap();
        let cosine = angles.cos().unwrap();
        assert!(sine.data()[0].abs() < 1.0e-6);
        assert!((sine.data()[1] - 1.0).abs() < 1.0e-6);
        assert!(cosine.data()[1].abs() < 1.0e-6);
        assert!((cosine.data()[2] + 1.0).abs() < 1.0e-6);
        assert!((tensor.tanh().unwrap().data()[1] - 0.5_f32.tanh()).abs() < 1.0e-6);
        assert!((positive.sigmoid().unwrap().data()[0] - 0.622_459_35).abs() < 1.0e-6);
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
    fn reduces_sum_and_mean_along_the_final_dimension() {
        let tensor = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 7.0]).unwrap();
        assert_eq!(tensor.sum_last_dim().unwrap().shape(), &[2]);
        assert_eq!(tensor.sum_last_dim().unwrap().data(), &[6.0, 16.0]);
        assert_eq!(tensor.mean_last_dim().unwrap().data(), &[2.0, 16.0 / 3.0]);
        let vector = Tensor::from_data([3], [1.0, 2.0, 6.0]).unwrap();
        assert_eq!(vector.sum_last_dim().unwrap().shape(), &[1]);
        assert_eq!(vector.sum_last_dim().unwrap().data(), &[9.0]);
    }

    #[test]
    fn cumulative_sum_supports_arbitrary_and_reverse_axes() {
        let tensor = Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            tensor.cumsum(1, false).unwrap().data(),
            &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]
        );
        assert_eq!(
            tensor.cumsum(0, true).unwrap().data(),
            &[5.0, 7.0, 9.0, 4.0, 5.0, 6.0]
        );
        assert!(matches!(
            tensor.cumsum(2, false),
            Err(TensorError::InvalidAxis { axis: 2, rank: 2 })
        ));
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
