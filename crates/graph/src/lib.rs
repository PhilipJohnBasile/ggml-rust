#![forbid(unsafe_code)]

use std::fmt;

use ggml_tensor::{RotaryScaling, Tensor, TensorError};

/// A handle to one value in a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(usize);

impl ValueId {
    /// Returns the insertion index represented by this handle.
    #[must_use]
    pub const fn index(self) -> usize {
        self.0
    }
}

/// Failures returned while constructing or evaluating a graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    InvalidValue(ValueId),
    InputCount {
        expected: usize,
        actual: usize,
    },
    InputShape {
        index: usize,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    NoOutputs,
    Tensor(TensorError),
}

impl fmt::Display for GraphError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidValue(value) => write!(formatter, "graph value {} is invalid", value.0),
            Self::InputCount { expected, actual } => write!(
                formatter,
                "graph received {actual} inputs, expected {expected}"
            ),
            Self::InputShape {
                index,
                expected,
                actual,
            } => write!(
                formatter,
                "graph input {index} has shape {actual:?}, expected {expected:?}"
            ),
            Self::NoOutputs => formatter.write_str("graph has no requested outputs"),
            Self::Tensor(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GraphError {}

impl From<TensorError> for GraphError {
    fn from(value: TensorError) -> Self {
        Self::Tensor(value)
    }
}

#[derive(Debug, Clone)]
enum Operation {
    Input {
        index: usize,
        shape: Vec<usize>,
    },
    Constant(Tensor),
    Add,
    Maximum,
    Minimum,
    Subtract,
    Multiply,
    Divide,
    Matmul,
    Reshape(Vec<usize>),
    Transpose2d,
    Permute(Vec<usize>),
    Slice {
        starts: Vec<usize>,
        ends: Vec<usize>,
        axes: Vec<usize>,
        strides: Vec<usize>,
    },
    SliceUpdate {
        starts: Vec<usize>,
        ends: Vec<usize>,
        axes: Vec<usize>,
        strides: Vec<usize>,
    },
    GatherRows(Vec<usize>),
    Concatenate {
        axis: usize,
    },
    Broadcast(Vec<usize>),
    RmsNorm {
        epsilon: f32,
    },
    RmsNormWeighted {
        epsilon: f32,
    },
    Scale {
        factor: f32,
    },
    Negate,
    Absolute,
    Square,
    SquareRoot,
    Reciprocal,
    Rsqrt,
    Exponential,
    Logarithm,
    Sine,
    Cosine,
    Tanh,
    Sigmoid,
    SumLastDim,
    MeanLastDim,
    Sum {
        axis: usize,
        keepdims: bool,
    },
    Mean {
        axis: usize,
        keepdims: bool,
    },
    Cumsum {
        axis: usize,
        reverse: bool,
    },
    Softmax {
        axis: usize,
    },
    Clamp {
        minimum: f32,
        maximum: f32,
    },
    Silu,
    SoftmaxLastDim,
    Rotary {
        rotary_dimension: usize,
        position: f32,
        frequency_base: f32,
        scaling: RotaryScaling,
    },
    Attention {
        scale: f32,
        causal: bool,
    },
}

#[derive(Debug, Clone)]
struct Node {
    operation: Operation,
    inputs: Vec<ValueId>,
}

/// A compact insertion-ordered tensor graph.
#[derive(Debug, Clone, Default)]
pub struct Graph {
    nodes: Vec<Node>,
    input_count: usize,
}

impl Graph {
    /// Creates an empty graph.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            input_count: 0,
        }
    }

    /// Adds a typed graph input.
    ///
    /// # Errors
    ///
    /// Returns an error when the declared shape is invalid.
    pub fn input<I>(&mut self, shape: I) -> Result<ValueId, GraphError>
    where
        I: IntoIterator<Item = usize>,
    {
        let shape = shape.into_iter().collect::<Vec<_>>();
        Tensor::zeros(shape.clone())?;
        let index = self.input_count;
        self.input_count += 1;
        Ok(self.push(Operation::Input { index, shape }, Vec::new()))
    }

    /// Adds an owned constant tensor.
    pub fn constant(&mut self, tensor: Tensor) -> ValueId {
        self.push(Operation::Constant(tensor), Vec::new())
    }

    /// Adds elementwise addition.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn add(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Add, left, right)
    }

    /// Adds an elementwise maximum node with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when either input handle is invalid.
    pub fn maximum(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Maximum, left, right)
    }

    /// Adds an elementwise minimum node with broadcasting.
    ///
    /// # Errors
    ///
    /// Returns an error when either input handle is invalid.
    pub fn minimum(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Minimum, left, right)
    }

    /// Adds elementwise multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn multiply(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Multiply, left, right)
    }

    /// Adds elementwise subtraction.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn subtract(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Subtract, left, right)
    }

    /// Adds elementwise division.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn divide(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Divide, left, right)
    }

    /// Adds a matrix multiplication node.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn matmul(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Matmul, left, right)
    }

    /// Adds a reshape node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid or the target shape
    /// is invalid.
    pub fn reshape<I>(&mut self, input: ValueId, shape: I) -> Result<ValueId, GraphError>
    where
        I: IntoIterator<Item = usize>,
    {
        self.require(input)?;
        let shape = shape.into_iter().collect::<Vec<_>>();
        Tensor::zeros(shape.clone())?;
        Ok(self.push(Operation::Reshape(shape), vec![input]))
    }

    /// Adds a rank-2 transpose node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn transpose_2d(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Transpose2d, input)
    }

    /// Adds an arbitrary-rank axis permutation node.
    ///
    /// `axes[i]` identifies the source axis that becomes output axis `i`.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid or `axes` is not a
    /// permutation of the input rank. Rank validation is completed during
    /// evaluation because graph values are shape-polymorphic until then.
    pub fn permute(
        &mut self,
        input: ValueId,
        axes: impl Into<Vec<usize>>,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(Operation::Permute(axes.into()), vec![input]))
    }

    /// Adds a positive-stride slice node.
    ///
    /// Axes not listed are copied in full with stride one. Bounds and output
    /// dimensions are checked when the graph is evaluated against a concrete
    /// tensor shape.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn slice(
        &mut self,
        input: ValueId,
        starts: impl Into<Vec<usize>>,
        ends: impl Into<Vec<usize>>,
        axes: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(
            Operation::Slice {
                starts: starts.into(),
                ends: ends.into(),
                axes: axes.into(),
                strides: strides.into(),
            },
            vec![input],
        ))
    }

    /// Adds a slice-update node that returns a new tensor.
    ///
    /// The update tensor must match the selected region. Bounds, strides, and
    /// shape compatibility are checked when the graph is evaluated.
    ///
    /// # Errors
    ///
    /// Returns an error when either input handle is invalid.
    pub fn slice_update(
        &mut self,
        input: ValueId,
        update: ValueId,
        starts: impl Into<Vec<usize>>,
        ends: impl Into<Vec<usize>>,
        axes: impl Into<Vec<usize>>,
        strides: impl Into<Vec<usize>>,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        self.require(update)?;
        Ok(self.push(
            Operation::SliceUpdate {
                starts: starts.into(),
                ends: ends.into(),
                axes: axes.into(),
                strides: strides.into(),
            },
            vec![input, update],
        ))
    }

    /// Adds a row-gather node for embedding and lookup tables.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid or no indices are
    /// supplied.
    pub fn gather_rows(
        &mut self,
        input: ValueId,
        indices: impl Into<Vec<usize>>,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        let indices = indices.into();
        if indices.is_empty() {
            return Err(GraphError::Tensor(TensorError::ZeroDimension));
        }
        Ok(self.push(Operation::GatherRows(indices), vec![input]))
    }

    /// Adds a concatenation node along one axis.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is invalid. Input rank and axis
    /// compatibility are checked during evaluation.
    pub fn concatenate(
        &mut self,
        left: ValueId,
        right: ValueId,
        axis: usize,
    ) -> Result<ValueId, GraphError> {
        self.require(left)?;
        self.require(right)?;
        Ok(self.push(Operation::Concatenate { axis }, vec![left, right]))
    }

    /// Adds a right-aligned broadcast node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle or target shape is invalid.
    pub fn broadcast_to<I>(&mut self, input: ValueId, shape: I) -> Result<ValueId, GraphError>
    where
        I: IntoIterator<Item = usize>,
    {
        self.require(input)?;
        let shape = shape.into_iter().collect::<Vec<_>>();
        Tensor::zeros(shape.clone())?;
        Ok(self.push(Operation::Broadcast(shape), vec![input]))
    }

    /// Adds an `RMSNorm` node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn rms_norm(&mut self, input: ValueId, epsilon: f32) -> Result<ValueId, GraphError> {
        self.unary(Operation::RmsNorm { epsilon }, input)
    }

    /// Adds a weighted `RMSNorm` node.
    ///
    /// The weight must evaluate to a rank-1 tensor whose length matches the
    /// input's final dimension.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is invalid.
    pub fn rms_norm_with_weight(
        &mut self,
        input: ValueId,
        weight: ValueId,
        epsilon: f32,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        self.require(weight)?;
        Ok(self.push(Operation::RmsNormWeighted { epsilon }, vec![input, weight]))
    }

    /// Adds a scalar multiplication node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn scale(&mut self, input: ValueId, factor: f32) -> Result<ValueId, GraphError> {
        self.unary(Operation::Scale { factor }, input)
    }

    /// Adds an elementwise negation node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn negate(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Negate, input)
    }

    /// Adds an elementwise absolute-value node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn absolute(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Absolute, input)
    }

    /// Adds an elementwise square node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn square(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Square, input)
    }

    /// Adds an elementwise square-root node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn square_root(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::SquareRoot, input)
    }

    /// Adds an elementwise reciprocal node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn reciprocal(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Reciprocal, input)
    }

    /// Adds an elementwise reciprocal square-root node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn rsqrt(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Rsqrt, input)
    }

    /// Adds an elementwise exponential node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn exponential(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Exponential, input)
    }

    /// Adds an elementwise natural-logarithm node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn logarithm(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Logarithm, input)
    }

    /// Adds an elementwise sine node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn sine(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Sine, input)
    }

    /// Adds an elementwise cosine node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn cosine(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Cosine, input)
    }

    /// Adds an elementwise hyperbolic-tangent node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn tanh(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Tanh, input)
    }

    /// Adds an elementwise logistic-sigmoid node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn sigmoid(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Sigmoid, input)
    }

    /// Adds an inclusive elementwise clamp node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid or bounds are
    /// non-finite or inverted.
    pub fn clamp(
        &mut self,
        input: ValueId,
        minimum: f32,
        maximum: f32,
    ) -> Result<ValueId, GraphError> {
        self.unary(Operation::Clamp { minimum, maximum }, input)
    }

    /// Adds a `SiLU` node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn silu(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::Silu, input)
    }

    /// Adds a stable last-dimension softmax node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn softmax_last_dim(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::SoftmaxLastDim, input)
    }

    /// Adds a numerically stable softmax node along one axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid. Axis validation is
    /// completed during evaluation against the concrete tensor rank.
    pub fn softmax(&mut self, input: ValueId, axis: usize) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(Operation::Softmax { axis }, vec![input]))
    }

    /// Adds a final-dimension sum reduction.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn sum_last_dim(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::SumLastDim, input)
    }

    /// Adds a final-dimension mean reduction.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn mean_last_dim(&mut self, input: ValueId) -> Result<ValueId, GraphError> {
        self.unary(Operation::MeanLastDim, input)
    }

    /// Adds an axis-specific sum reduction node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid. Axis validation is
    /// completed during evaluation against the concrete tensor rank.
    pub fn sum(
        &mut self,
        input: ValueId,
        axis: usize,
        keepdims: bool,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(Operation::Sum { axis, keepdims }, vec![input]))
    }

    /// Adds an axis-specific mean reduction node.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid. Axis validation is
    /// completed during evaluation against the concrete tensor rank.
    pub fn mean(
        &mut self,
        input: ValueId,
        axis: usize,
        keepdims: bool,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(Operation::Mean { axis, keepdims }, vec![input]))
    }

    /// Adds an inclusive cumulative sum node along one axis.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid. Axis validation is
    /// completed during evaluation against the concrete tensor rank.
    pub fn cumsum(
        &mut self,
        input: ValueId,
        axis: usize,
        reverse: bool,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(Operation::Cumsum { axis, reverse }, vec![input]))
    }

    /// Adds an interleaved rotary position embedding node.
    ///
    /// The input must evaluate to `[heads, head_dim]`; only the leading
    /// `rotary_dimension` values are rotated.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn rotary_embedding(
        &mut self,
        input: ValueId,
        rotary_dimension: usize,
        position: f32,
        frequency_base: f32,
    ) -> Result<ValueId, GraphError> {
        self.rotary_embedding_with_scaling(
            input,
            rotary_dimension,
            position,
            frequency_base,
            RotaryScaling::None,
        )
    }

    /// Adds an interleaved rotary position embedding node with linear or `YaRN`
    /// scaling.
    ///
    /// The input must evaluate to `[heads, head_dim]`; only the leading
    /// `rotary_dimension` values are rotated.
    ///
    /// # Errors
    ///
    /// Returns an error when the input handle is invalid.
    pub fn rotary_embedding_with_scaling(
        &mut self,
        input: ValueId,
        rotary_dimension: usize,
        position: f32,
        frequency_base: f32,
        scaling: RotaryScaling,
    ) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(
            Operation::Rotary {
                rotary_dimension,
                position,
                frequency_base,
                scaling,
            },
            vec![input],
        ))
    }

    /// Adds a rank-4 grouped-query scaled dot-product attention node.
    ///
    /// # Errors
    ///
    /// Returns an error when one of the input handles is invalid.
    pub fn scaled_dot_product_attention(
        &mut self,
        queries: ValueId,
        keys: ValueId,
        values: ValueId,
        scale: f32,
        causal: bool,
    ) -> Result<ValueId, GraphError> {
        self.require(queries)?;
        self.require(keys)?;
        self.require(values)?;
        Ok(self.push(
            Operation::Attention { scale, causal },
            vec![queries, keys, values],
        ))
    }

    /// Evaluates requested graph values with the checked CPU tensor backend.
    ///
    /// Inputs are supplied in the order in which [`Graph::input`] was called.
    /// Values are evaluated once in insertion order, so shared subgraphs are
    /// not recomputed.
    ///
    /// # Errors
    ///
    /// Returns an error when input count or shape validation fails, a requested
    /// value is invalid, or one of the tensor operations fails.
    #[allow(clippy::too_many_lines)]
    pub fn evaluate(
        &self,
        inputs: &[Tensor],
        outputs: &[ValueId],
    ) -> Result<Vec<Tensor>, GraphError> {
        if outputs.is_empty() {
            return Err(GraphError::NoOutputs);
        }
        if inputs.len() != self.input_count {
            return Err(GraphError::InputCount {
                expected: self.input_count,
                actual: inputs.len(),
            });
        }
        for node in &self.nodes {
            if let Operation::Input { index, shape } = &node.operation {
                let input = &inputs[*index];
                if input.shape() != shape.as_slice() {
                    return Err(GraphError::InputShape {
                        index: *index,
                        expected: shape.clone(),
                        actual: input.shape().to_vec(),
                    });
                }
            }
        }
        let mut values: Vec<Tensor> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let value = match &node.operation {
                Operation::Input { index, .. } => inputs[*index].clone(),
                Operation::Constant(tensor) => tensor.clone(),
                Operation::Add => {
                    values[self.input_index(node, 0)?].add(&values[self.input_index(node, 1)?])?
                }
                Operation::Maximum => values[self.input_index(node, 0)?]
                    .maximum(&values[self.input_index(node, 1)?])?,
                Operation::Minimum => values[self.input_index(node, 0)?]
                    .minimum(&values[self.input_index(node, 1)?])?,
                Operation::Subtract => {
                    values[self.input_index(node, 0)?].sub(&values[self.input_index(node, 1)?])?
                }
                Operation::Multiply => {
                    values[self.input_index(node, 0)?].mul(&values[self.input_index(node, 1)?])?
                }
                Operation::Divide => {
                    values[self.input_index(node, 0)?].div(&values[self.input_index(node, 1)?])?
                }
                Operation::Matmul => values[self.input_index(node, 0)?]
                    .matmul(&values[self.input_index(node, 1)?])?,
                Operation::Reshape(shape) => values[self.input_index(node, 0)?]
                    .clone()
                    .reshape(shape.clone())?,
                Operation::Transpose2d => values[self.input_index(node, 0)?].transpose_2d()?,
                Operation::Permute(axes) => values[self.input_index(node, 0)?].permute(axes)?,
                Operation::Slice {
                    starts,
                    ends,
                    axes,
                    strides,
                } => values[self.input_index(node, 0)?].slice(starts, ends, axes, strides)?,
                Operation::SliceUpdate {
                    starts,
                    ends,
                    axes,
                    strides,
                } => values[self.input_index(node, 0)?].slice_update(
                    &values[self.input_index(node, 1)?],
                    starts,
                    ends,
                    axes,
                    strides,
                )?,
                Operation::GatherRows(indices) => {
                    values[self.input_index(node, 0)?].gather_rows(indices)?
                }
                Operation::Concatenate { axis } => values[self.input_index(node, 0)?]
                    .concatenate(&values[self.input_index(node, 1)?], *axis)?,
                Operation::Broadcast(shape) => {
                    values[self.input_index(node, 0)?].broadcast_to(shape.clone())?
                }
                Operation::RmsNorm { epsilon } => {
                    values[self.input_index(node, 0)?].rms_norm(*epsilon)?
                }
                Operation::RmsNormWeighted { epsilon } => values[self.input_index(node, 0)?]
                    .rms_norm_with_weight(&values[self.input_index(node, 1)?], *epsilon)?,
                Operation::Scale { factor } => values[self.input_index(node, 0)?].scale(*factor)?,
                Operation::Negate => values[self.input_index(node, 0)?].neg()?,
                Operation::Absolute => values[self.input_index(node, 0)?].abs()?,
                Operation::Square => values[self.input_index(node, 0)?].sqr()?,
                Operation::SquareRoot => values[self.input_index(node, 0)?].sqrt()?,
                Operation::Reciprocal => values[self.input_index(node, 0)?].reciprocal()?,
                Operation::Rsqrt => values[self.input_index(node, 0)?].rsqrt()?,
                Operation::Exponential => values[self.input_index(node, 0)?].exp()?,
                Operation::Logarithm => values[self.input_index(node, 0)?].log()?,
                Operation::Sine => values[self.input_index(node, 0)?].sin()?,
                Operation::Cosine => values[self.input_index(node, 0)?].cos()?,
                Operation::Tanh => values[self.input_index(node, 0)?].tanh()?,
                Operation::Sigmoid => values[self.input_index(node, 0)?].sigmoid()?,
                Operation::Clamp { minimum, maximum } => {
                    values[self.input_index(node, 0)?].clamp(*minimum, *maximum)?
                }
                Operation::Silu => values[self.input_index(node, 0)?].silu()?,
                Operation::SoftmaxLastDim => {
                    values[self.input_index(node, 0)?].softmax_last_dim()?
                }
                Operation::SumLastDim => values[self.input_index(node, 0)?].sum_last_dim()?,
                Operation::MeanLastDim => values[self.input_index(node, 0)?].mean_last_dim()?,
                Operation::Sum { axis, keepdims } => {
                    values[self.input_index(node, 0)?].sum(*axis, *keepdims)?
                }
                Operation::Mean { axis, keepdims } => {
                    values[self.input_index(node, 0)?].mean(*axis, *keepdims)?
                }
                Operation::Cumsum { axis, reverse } => {
                    values[self.input_index(node, 0)?].cumsum(*axis, *reverse)?
                }
                Operation::Softmax { axis } => values[self.input_index(node, 0)?].softmax(*axis)?,
                Operation::Rotary {
                    rotary_dimension,
                    position,
                    frequency_base,
                    scaling,
                } => values[self.input_index(node, 0)?].rotary_embedding_with_scaling(
                    *rotary_dimension,
                    *position,
                    *frequency_base,
                    *scaling,
                )?,
                Operation::Attention { scale, causal } => values[self.input_index(node, 0)?]
                    .scaled_dot_product_attention(
                        &values[self.input_index(node, 1)?],
                        &values[self.input_index(node, 2)?],
                        *scale,
                        *causal,
                    )?,
            };
            values.push(value);
        }
        outputs
            .iter()
            .map(|output| {
                values
                    .get(output.0)
                    .cloned()
                    .ok_or(GraphError::InvalidValue(*output))
            })
            .collect()
    }

    /// Returns the number of graph nodes.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns whether the graph contains no nodes.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn push(&mut self, operation: Operation, inputs: Vec<ValueId>) -> ValueId {
        let value = ValueId(self.nodes.len());
        self.nodes.push(Node { operation, inputs });
        value
    }

    fn require(&self, value: ValueId) -> Result<(), GraphError> {
        self.nodes
            .get(value.0)
            .map(|_| ())
            .ok_or(GraphError::InvalidValue(value))
    }

    fn input_index(&self, node: &Node, index: usize) -> Result<usize, GraphError> {
        let value = node
            .inputs
            .get(index)
            .copied()
            .ok_or(GraphError::InvalidValue(ValueId(usize::MAX)))?;
        if value.0 >= self.nodes.len() {
            return Err(GraphError::InvalidValue(value));
        }
        Ok(value.0)
    }

    fn unary(&mut self, operation: Operation, input: ValueId) -> Result<ValueId, GraphError> {
        self.require(input)?;
        Ok(self.push(operation, vec![input]))
    }

    fn binary(
        &mut self,
        operation: Operation,
        left: ValueId,
        right: ValueId,
    ) -> Result<ValueId, GraphError> {
        self.require(left)?;
        self.require(right)?;
        Ok(self.push(operation, vec![left, right]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluates_shared_linear_graph() {
        let mut graph = Graph::new();
        let input = graph.input([2, 2]).unwrap();
        let weights = graph.constant(Tensor::from_data([2, 2], [2.0, 0.0, 0.0, 2.0]).unwrap());
        let projected = graph.matmul(input, weights).unwrap();
        let residual = graph.add(projected, input).unwrap();
        let output = graph.silu(residual).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap()],
                &[output],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[2, 2]);
        assert!(result[0].data().iter().all(|value| value.is_finite()));
        assert_eq!(graph.len(), 5);
    }

    #[test]
    fn evaluates_broadcasted_minimum_and_maximum() {
        let mut graph = Graph::new();
        let input = graph.input([2, 2]).unwrap();
        let bound = graph.constant(Tensor::from_data([2], [0.0, 3.0]).unwrap());
        let lower = graph.minimum(input, bound).unwrap();
        let upper = graph.maximum(input, bound).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 2], [-1.0, 2.0, 4.0, 5.0]).unwrap()],
                &[lower, upper],
            )
            .unwrap();
        assert_eq!(result[0].data(), &[-1.0, 2.0, 0.0, 3.0]);
        assert_eq!(result[1].data(), &[0.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn evaluates_attention_graph_with_causal_grouped_heads() {
        let mut graph = Graph::new();
        let queries = graph.input([1, 2, 2, 1]).unwrap();
        let keys = graph.input([1, 1, 3, 1]).unwrap();
        let values = graph.input([1, 1, 3, 1]).unwrap();
        let attention = graph
            .scaled_dot_product_attention(queries, keys, values, 1.0, true)
            .unwrap();
        let result = graph
            .evaluate(
                &[
                    Tensor::from_data([1, 2, 2, 1], [0.0; 4]).unwrap(),
                    Tensor::from_data([1, 1, 3, 1], [0.0; 3]).unwrap(),
                    Tensor::from_data([1, 1, 3, 1], [1.0, 3.0, 5.0]).unwrap(),
                ],
                &[attention],
            )
            .unwrap();
        assert_eq!(result[0].data(), &[2.0, 3.0, 2.0, 3.0]);
    }

    #[test]
    fn evaluates_rotary_embedding_graph() {
        let mut graph = Graph::new();
        let input = graph.input([2, 2]).unwrap();
        let rotated = graph.rotary_embedding(input, 2, 1.0, 10_000.0).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 2], [1.0, 0.0, 0.0, 1.0]).unwrap()],
                &[rotated],
            )
            .unwrap();
        assert!((result[0].data()[0] - 0.540_302_3).abs() < 1.0e-5);
        assert!((result[0].data()[1] - 0.841_470_96).abs() < 1.0e-5);
        assert!((result[0].data()[2] + 0.841_470_96).abs() < 1.0e-5);
        assert!((result[0].data()[3] - 0.540_302_3).abs() < 1.0e-5);
    }

    #[test]
    fn evaluates_scaled_rotary_embedding_graph() {
        let mut graph = Graph::new();
        let input = graph.input([1, 2]).unwrap();
        let rotated = graph
            .rotary_embedding_with_scaling(
                input,
                2,
                0.0,
                10_000.0,
                RotaryScaling::Linear { factor: 2.0 },
            )
            .unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([1, 2], [1.0, 0.0]).unwrap()],
                &[rotated],
            )
            .unwrap();
        assert_eq!(result[0].data(), &[1.0, 0.0]);
    }

    #[test]
    fn evaluates_weighted_rms_norm_graph() {
        let mut graph = Graph::new();
        let input = graph.input([2, 2]).unwrap();
        let weight = graph.constant(Tensor::from_data([2], [2.0, 3.0]).unwrap());
        let normalized = graph.rms_norm_with_weight(input, weight, 0.0).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 2], [1.0, 1.0, 1.0, 1.0]).unwrap()],
                &[normalized],
            )
            .unwrap();
        assert!((result[0].data()[0] - 2.0).abs() < 1.0e-6);
        assert!((result[0].data()[1] - 3.0).abs() < 1.0e-6);
        assert!((result[0].data()[2] - 2.0).abs() < 1.0e-6);
        assert!((result[0].data()[3] - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn evaluates_embedding_row_gather_graph() {
        let mut graph = Graph::new();
        let table = graph.input([3, 2]).unwrap();
        let gathered = graph.gather_rows(table, vec![2, 0]).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([3, 2], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                &[gathered],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0].data(), &[5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn evaluates_concatenation_graph() {
        let mut graph = Graph::new();
        let left = graph.input([2, 1]).unwrap();
        let right = graph.input([2, 2]).unwrap();
        let joined = graph.concatenate(left, right, 1).unwrap();
        let result = graph
            .evaluate(
                &[
                    Tensor::from_data([2, 1], [1.0, 2.0]).unwrap(),
                    Tensor::from_data([2, 2], [3.0, 4.0, 5.0, 6.0]).unwrap(),
                ],
                &[joined],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[2, 3]);
        assert_eq!(result[0].data(), &[1.0, 3.0, 4.0, 2.0, 5.0, 6.0]);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn evaluates_arbitrary_permutation_graph() {
        let mut graph = Graph::new();
        let input = graph.input([2, 3, 2]).unwrap();
        let permuted = graph.permute(input, vec![2, 0, 1]).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data(
                    [2, 3, 2],
                    (0..12).map(|value| value as f32).collect::<Vec<_>>(),
                )
                .unwrap()],
                &[permuted],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[2, 2, 3]);
        assert_eq!(result[0].data()[..6], [0.0, 2.0, 4.0, 6.0, 8.0, 10.0]);
    }

    #[test]
    fn evaluates_strided_slice_and_update_graph() {
        let mut graph = Graph::new();
        let input = graph.input([2, 3]).unwrap();
        let sliced = graph.slice(input, [0], [3], [1], [2]).unwrap();
        let update = graph.constant(Tensor::from_data([2, 2], [9.0, 8.0, 7.0, 6.0]).unwrap());
        let updated = graph
            .slice_update(input, update, [1], [3], [1], [1])
            .unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                &[sliced, updated],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[2, 2]);
        assert_eq!(result[0].data(), &[1.0, 3.0, 4.0, 6.0]);
        assert_eq!(result[1].data(), &[1.0, 9.0, 8.0, 4.0, 7.0, 6.0]);
    }

    #[test]
    fn evaluates_trigonometric_graph_nodes() {
        let mut graph = Graph::new();
        let input = graph.input([3]).unwrap();
        let sine = graph.sine(input).unwrap();
        let cosine = graph.cosine(input).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data(
                    [3],
                    [0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI],
                )
                .unwrap()],
                &[sine, cosine],
            )
            .unwrap();
        assert!((result[0].data()[1] - 1.0).abs() < 1.0e-6);
        assert!(result[0].data()[2].abs() < 1.0e-6);
        assert!(result[1].data()[1].abs() < 1.0e-6);
        assert!((result[1].data()[2] + 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn evaluates_reciprocal_graph_nodes() {
        let mut graph = Graph::new();
        let input = graph.input([3]).unwrap();
        let reciprocal = graph.reciprocal(input).unwrap();
        let rsqrt = graph.rsqrt(input).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([3], [0.25, 1.0, 4.0]).unwrap()],
                &[reciprocal, rsqrt],
            )
            .unwrap();
        assert_eq!(result[0].data(), &[4.0, 1.0, 0.25]);
        assert_eq!(result[1].data(), &[2.0, 1.0, 0.5]);
    }

    #[test]
    fn evaluates_axis_reductions_with_and_without_keepdims() {
        let mut graph = Graph::new();
        let input = graph.input([2, 3]).unwrap();
        let sums = graph.sum(input, 0, false).unwrap();
        let means = graph.mean(input, 1, true).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                &[sums, means],
            )
            .unwrap();
        assert_eq!(result[0].shape(), &[3]);
        assert_eq!(result[0].data(), &[5.0, 7.0, 9.0]);
        assert_eq!(result[1].shape(), &[2, 1]);
        assert_eq!(result[1].data(), &[2.0, 5.0]);
    }

    #[test]
    fn evaluates_cumulative_sum_nodes() {
        let mut graph = Graph::new();
        let input = graph.input([2, 3]).unwrap();
        let forward = graph.cumsum(input, 1, false).unwrap();
        let reverse = graph.cumsum(input, 0, true).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                &[forward, reverse],
            )
            .unwrap();
        assert_eq!(result[0].data(), &[1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
        assert_eq!(result[1].data(), &[5.0, 7.0, 9.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn evaluates_axis_softmax_nodes() {
        let mut graph = Graph::new();
        let input = graph.input([2, 2]).unwrap();
        let output = graph.softmax(input, 0).unwrap();
        let result = graph
            .evaluate(
                &[Tensor::from_data([2, 2], [1.0, 2.0, 3.0, 4.0]).unwrap()],
                &[output],
            )
            .unwrap();
        assert!((result[0].data()[0] + result[0].data()[2] - 1.0).abs() < 1.0e-6);
        assert!((result[0].data()[1] + result[0].data()[3] - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn validates_input_shapes_and_outputs() {
        let mut graph = Graph::new();
        let input = graph.input([2]).unwrap();
        assert_eq!(
            graph.evaluate(&[Tensor::from_data([1], [1.0]).unwrap()], &[input]),
            Err(GraphError::InputShape {
                index: 0,
                expected: vec![2],
                actual: vec![1],
            })
        );
        assert_eq!(
            graph.evaluate(&[Tensor::from_data([2], [1.0, 2.0]).unwrap()], &[]),
            Err(GraphError::NoOutputs)
        );
    }
}
