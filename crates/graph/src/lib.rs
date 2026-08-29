#![forbid(unsafe_code)]

use std::fmt;

use ggml_tensor::{Tensor, TensorError};

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
    Input { index: usize, shape: Vec<usize> },
    Constant(Tensor),
    Add,
    Multiply,
    Matmul,
    Reshape(Vec<usize>),
    Transpose2d,
    Broadcast(Vec<usize>),
    RmsNorm { epsilon: f32 },
    Silu,
    SoftmaxLastDim,
    Attention { scale: f32, causal: bool },
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

    /// Adds elementwise multiplication.
    ///
    /// # Errors
    ///
    /// Returns an error when either handle is not in this graph.
    pub fn multiply(&mut self, left: ValueId, right: ValueId) -> Result<ValueId, GraphError> {
        self.binary(Operation::Multiply, left, right)
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
            let value =
                match &node.operation {
                    Operation::Input { index, .. } => inputs[*index].clone(),
                    Operation::Constant(tensor) => tensor.clone(),
                    Operation::Add => values[self.input_index(node, 0)?]
                        .add(&values[self.input_index(node, 1)?])?,
                    Operation::Multiply => values[self.input_index(node, 0)?]
                        .mul(&values[self.input_index(node, 1)?])?,
                    Operation::Matmul => values[self.input_index(node, 0)?]
                        .matmul(&values[self.input_index(node, 1)?])?,
                    Operation::Reshape(shape) => values[self.input_index(node, 0)?]
                        .clone()
                        .reshape(shape.clone())?,
                    Operation::Transpose2d => values[self.input_index(node, 0)?].transpose_2d()?,
                    Operation::Broadcast(shape) => {
                        values[self.input_index(node, 0)?].broadcast_to(shape.clone())?
                    }
                    Operation::RmsNorm { epsilon } => {
                        values[self.input_index(node, 0)?].rms_norm(*epsilon)?
                    }
                    Operation::Silu => values[self.input_index(node, 0)?].silu()?,
                    Operation::SoftmaxLastDim => {
                        values[self.input_index(node, 0)?].softmax_last_dim()?
                    }
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
