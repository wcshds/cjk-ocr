use burn::{
    nn::{
        conv::{Conv2d, Conv2dConfig},
        pool::{MaxPool2d, MaxPool2dConfig},
    },
    tensor::{backend::Backend, Bool, Tensor},
};

pub fn convolution<B: Backend>(
    device: &B::Device,
    in_channels: usize,
    out_channels: usize,
    groups: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    bias: bool,
) -> Conv2d<B> {
    Conv2dConfig::new([in_channels, out_channels], kernel_size)
        .with_groups(groups)
        .with_stride(stride)
        .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
        .with_bias(bias)
        .with_initializer(burn::nn::Initializer::KaimingNormal {
            gain: 2f64.sqrt(), // recommended gain value for relu
            fan_out_only: true,
        })
        .init(device)
}

pub fn max_pool_2d(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> MaxPool2d {
    MaxPool2dConfig::new(kernel_size)
        .with_strides(stride)
        .with_padding(burn::nn::PaddingConfig2d::Explicit(padding[0], padding[1]))
        .init()
}

pub fn generate_autoregressive_mask<B: Backend>(
    batch_size: usize,
    seq_length: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    Tensor::<B, 2>::ones([seq_length, seq_length], device)
        .triu(1)
        .bool()
        .unsqueeze_dim(0)
        .repeat(0, batch_size)
}
