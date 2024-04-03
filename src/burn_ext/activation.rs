use burn::{
    module::Module,
    tensor::{backend::Backend, Tensor},
};

/// Applies the rectified linear unit function (maximum value is 6) element-wise:
///
/// `y = min(max(0, x), 6)`
#[derive(Module, Clone, Debug, Default)]
pub struct Relu6 {}

impl Relu6 {
    /// Create the module.
    pub fn new() -> Self {
        Self {}
    }

    /// Applies the forward pass on the input tensor.
    ///
    /// # Shapes
    ///
    /// - input: `[..., any]`
    /// - output: `[..., any]`
    pub fn forward<B: Backend, const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let tmp = burn::tensor::activation::relu(input);
        tmp.clamp_max(6.0)
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};

    use super::*;

    #[test]
    fn test_relu6() {
        let device = NdArrayDevice::Cpu;

        let relu6 = Relu6::new();
        let tensor: Tensor<NdArray, 3> = Tensor::random(
            [2, 3, 4],
            burn::tensor::Distribution::Normal(0.0, 10.0),
            &device,
        );
        let res = relu6.forward(tensor.clone());

        println!("{}", tensor);
        println!("{}", res);
    }
}
