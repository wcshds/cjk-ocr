use burn::{
    config::Config,
    module::Module,
    nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig, Relu},
    tensor::{backend::Backend, Tensor},
};

/// Configuration to create a [position-wise feed-forward](PositionWiseFeedForward) layer.
#[derive(Config)]
pub struct PositionWiseFeedForwardConfig {
    /// The size of the input and output features.
    pub d_model: usize,
    /// The size of the hidden inner features.
    pub d_ff: usize,
    /// The dropout rate. Default: 0.1
    #[config(default = 0.1)]
    pub dropout: f64,
    /// The type of function used to initialize neural network parameters
    #[config(
        default = "Initializer::KaimingUniform{gain:1.0/num_traits::Float::sqrt(3.0), fan_out_only:false}"
    )]
    pub initializer: Initializer,
}

/// Applies the position-wise feed-forward network to the input tensor.
///
/// # Params
///
/// - linear inner: Linear layer with `d_model` input features and `d_ff` output features.
/// - linear outer: Linear layer with `d_ff` input features and `d_model` output features.
#[derive(Module, Debug)]
pub struct PositionWiseFeedForward<B: Backend> {
    linear_inner: Linear<B>,
    linear_outer: Linear<B>,
    dropout: Dropout,
    relu: Relu,
}

impl PositionWiseFeedForwardConfig {
    /// Initialize a new [position-wise feed-forward](PositionWiseFeedForward) module.
    pub fn init<B: Backend>(&self, device: &B::Device) -> PositionWiseFeedForward<B> {
        PositionWiseFeedForward {
            linear_inner: LinearConfig::new(self.d_model, self.d_ff)
                .with_initializer(self.initializer.clone())
                .init(device),
            linear_outer: LinearConfig::new(self.d_ff, self.d_model)
                .with_initializer(self.initializer.clone())
                .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            relu: Relu::new(),
        }
    }
}

impl<B: Backend> PositionWiseFeedForward<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - tensor: `[batch_size, seq_length, d_model]`
    /// - output: `[batch_size, seq_length, d_model]`
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear_inner.forward(input);
        let x = self.relu.forward(x);
        let x = self.dropout.forward(x);

        self.linear_outer.forward(x)
    }
}
