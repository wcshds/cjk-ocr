use burn::{
    config::Config,
    module::Module,
    nn::{
        conv::Conv2d, BatchNorm, BatchNormConfig, Dropout, DropoutConfig, LayerNorm,
        LayerNormConfig, PositionalEncoding, PositionalEncodingConfig, Relu,
    },
    tensor::{activation, backend::Backend, Tensor},
};

use crate::burn_ext::{
    activation::Relu6,
    sequential::Sequential,
    utils::{convolution, max_pool_2d},
};

#[derive(Module, Debug)]
pub struct InvertedResidualBlock<B: Backend> {
    conv1x1before: Option<Conv2d<B>>,
    batchnorm1: Option<BatchNorm<B, 2>>,
    depthwise: Conv2d<B>,
    conv1x1after: Conv2d<B>,
    batchnorm2: BatchNorm<B, 2>,
    batchnorm3: BatchNorm<B, 2>,
    relu6: Relu6,
    apply_residual: bool,
}

impl<B: Backend> InvertedResidualBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = match self.apply_residual {
            true => Some(input.clone()),
            false => None,
        };

        let out = match (&self.conv1x1before, &self.batchnorm1) {
            (Some(conv1x1before), Some(batchnorm1)) => {
                let out = conv1x1before.forward(input);
                let out = batchnorm1.forward(out);
                let out = self.relu6.forward(out);

                out
            }
            _ => input,
        };

        let out = self.depthwise.forward(out);
        let out = self.batchnorm2.forward(out);
        let out = self.relu6.forward(out);

        // linear activation here
        let out = self.conv1x1after.forward(out);
        let out = self.batchnorm3.forward(out);

        match residual {
            Some(residual) => out + residual,
            None => out,
        }
    }
}

#[derive(Config, Debug)]
pub struct InvertedResidualBlockConfig {
    in_channels: usize,
    out_channels: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    #[config(default = "[1, 1]")]
    padding: [usize; 2],
    #[config(default = "6")]
    expansion: usize,
    #[config(default = "false")]
    bias: bool,
}

impl InvertedResidualBlockConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> InvertedResidualBlock<B> {
        let apply_residual = self.stride == [1, 1] && self.in_channels == self.out_channels;
        let expanded_channels = self.in_channels * self.expansion;

        let (conv1x1before, batchnorm1) = match self.expansion {
            1 => (None, None),
            _ => {
                let conv = convolution(
                    device,
                    self.in_channels,
                    expanded_channels,
                    1,
                    [1, 1],
                    [1, 1],
                    [0, 0],
                    self.bias,
                );
                let batchnorm = BatchNormConfig::new(expanded_channels).init(device);

                (Some(conv), Some(batchnorm))
            }
        };
        let depthwise = convolution(
            device,
            expanded_channels,
            expanded_channels,
            expanded_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.bias,
        );
        let conv1x1after = convolution(
            device,
            expanded_channels,
            self.out_channels,
            1,
            [1, 1],
            [1, 1],
            [0, 0],
            self.bias,
        );

        InvertedResidualBlock {
            conv1x1before,
            batchnorm1,
            depthwise,
            conv1x1after,
            batchnorm2: BatchNormConfig::new(expanded_channels).init(device),
            batchnorm3: BatchNormConfig::new(self.out_channels).init(device),
            relu6: Relu6::new(),
            apply_residual,
        }
    }
}

#[derive(Module, Debug)]
struct ChannelConcatConv<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    layernorm: LayerNorm<B>,
    relu: Relu,
}

impl<B: Backend> ChannelConcatConv<B> {
    fn new(device: &B::Device, channels: usize, inner_channels: usize) -> Self {
        let conv1 = convolution(
            device,
            channels,
            inner_channels,
            1,
            [1, 1],
            [1, 1],
            [0, 0],
            true,
        );
        let conv2 = convolution(
            device,
            inner_channels,
            channels,
            1,
            [1, 1],
            [1, 1],
            [0, 0],
            true,
        );
        let layernorm = LayerNormConfig::new(inner_channels).init(device);
        let relu = Relu::new();

        Self {
            conv1,
            conv2,
            layernorm,
            relu,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let out = self.conv1.forward(input);
        let out = out.squeeze::<3>(3).squeeze::<2>(2);
        let out = self.layernorm.forward(out);
        let out = out.unsqueeze_dim::<3>(2).unsqueeze_dim(3);
        let out = self.relu.forward(out);
        let out = self.conv2.forward(out);

        out
    }
}

#[derive(Module, Debug)]
pub struct MultiAspectGCAttention<B: Backend> {
    conv_mask: Conv2d<B>,
    channel_concat_conv: ChannelConcatConv<B>,
    headers: usize,
    single_header_channels: usize,
    att_scale: bool,
}

impl<B: Backend> MultiAspectGCAttention<B> {
    fn spatial_pool(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _channel, height, width] = input.dims();
        // [N*headers, C', H , W], C = headers * C'
        let origin_input = input.reshape([
            batch * self.headers,
            self.single_header_channels,
            height,
            width,
        ]);
        let input = origin_input.clone();

        // [N*headers, C', H * W], C = headers * C'
        let input = input.reshape([
            batch * self.headers,
            self.single_header_channels,
            height * width,
        ]);

        // [N*headers, 1, C', H * W]
        let input: Tensor<B, 4> = input.unsqueeze_dim(1);
        // [N*headers, 1, H, W]
        let context_mask = self.conv_mask.forward(origin_input);
        let mut context_mask = context_mask.reshape([batch * self.headers, 1, height * width]);

        // scale variance
        if self.att_scale && self.headers > 1 {
            context_mask = context_mask.div_scalar((self.single_header_channels as f64).sqrt());
        }

        // [N*headers, 1, H * W]
        let context_mask = activation::softmax(context_mask, 2);

        // [N*headers, 1, H * W, 1]
        let context_mask = context_mask.unsqueeze_dim(3);
        // [N*headers, 1, C', 1] = [N*headers, 1, C', H * W] * [N*headers, 1, H * W, 1]
        let context = input.matmul(context_mask);

        // [N, headers * C', 1, 1]
        let context = context.reshape([batch, self.headers * self.single_header_channels, 1, 1]);

        context
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let context = self.spatial_pool(input.clone());

        // [N, C, 1, 1]
        let channel_concat_term = self.channel_concat_conv.forward(context);
        // The open source code uses cancat, but according to the original paper, add is used here
        input + channel_concat_term
    }
}

#[derive(Config, Debug)]
struct MultiAspectGCAttentionConfig {
    channels: usize,
    #[config(default = "1.0/16.0")]
    ratio: f64,
    #[config(default = "1")]
    headers: usize,
    #[config(default = "true")]
    att_scale: bool,
}

impl MultiAspectGCAttentionConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> MultiAspectGCAttention<B> {
        assert!(
            self.channels % self.headers == 0 && self.channels >= 8,
            "channels must be divided by headers evenly"
        );

        let inner_channels = (self.channels as f64 * self.ratio) as usize;
        let single_header_channels = self.channels / self.headers;

        MultiAspectGCAttention {
            conv_mask: convolution(
                device,
                single_header_channels,
                1,
                1,
                [1, 1],
                [1, 1],
                [0, 0],
                true,
            ),
            channel_concat_conv: ChannelConcatConv::new(device, self.channels, inner_channels),
            headers: self.headers,
            single_header_channels,
            att_scale: self.att_scale,
        }
    }
}

#[derive(Module, Debug)]
pub struct Encoder<B: Backend> {
    pub conv1: Sequential<B>,
    pub conv2: Sequential<B>,
    pub conv3: Sequential<B>,
    pub conv4: Sequential<B>,
    pub conv5: Sequential<B>,
    position: PositionalEncoding<B>,
    dropout: Dropout,
}

impl<B: Backend> Encoder<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv1 = Sequential::from(vec![
            convolution(device, 1, 32, 1, [3, 3], [1, 1], [1, 1], false).into(),
            BatchNormConfig::new(32).init(device).into(),
            Relu6::new().into(),
            InvertedResidualBlockConfig::new(32, 16, [3, 3], [1, 1])
                .with_padding([1, 1])
                .with_expansion(1)
                .with_bias(false)
                .init(device)
                .into(),
            max_pool_2d([2, 2], [2, 2], [0, 0]).into(),
        ]);
        let conv2 = Sequential::from(vec![
            InvertedResidualBlockConfig::new(16, 64, [3, 3], [1, 1])
                .with_padding([1, 1])
                .with_expansion(6)
                .with_bias(false)
                .init(device)
                .into(),
            InvertedResidualBlockConfig::new(64, 64, [3, 3], [1, 1])
                .with_padding([1, 1])
                .with_expansion(6)
                .with_bias(false)
                .init(device)
                .into(),
            MultiAspectGCAttentionConfig::new(64)
                .with_ratio(1.0 / 8.0)
                .with_headers(8)
                .with_att_scale(true)
                .init(device)
                .into(),
            max_pool_2d([2, 2], [2, 2], [0, 0]).into(),
        ]);
        let mut conv3 = Sequential::from(vec![InvertedResidualBlockConfig::new(
            64,
            128,
            [3, 3],
            [1, 1],
        )
        .with_padding([1, 1])
        .with_expansion(6)
        .with_bias(false)
        .init(device)
        .into()]);
        for _ in 0..2 {
            conv3.append(
                InvertedResidualBlockConfig::new(128, 128, [3, 3], [1, 1])
                    .with_padding([1, 1])
                    .with_expansion(6)
                    .with_bias(false)
                    .init(device)
                    .into(),
            )
        }
        conv3.append(
            MultiAspectGCAttentionConfig::new(128)
                .with_ratio(1.0 / 16.0)
                .with_headers(8)
                .with_att_scale(true)
                .init(device)
                .into(),
        );
        conv3.append(max_pool_2d([2, 1], [2, 1], [0, 0]).into());

        let conv4 = Sequential::from(vec![
            InvertedResidualBlockConfig::new(128, 256, [3, 3], [1, 1])
                .with_padding([1, 1])
                .with_expansion(6)
                .with_bias(false)
                .init(device)
                .into(),
            MultiAspectGCAttentionConfig::new(256)
                .with_ratio(1.0 / 16.0)
                .with_headers(8)
                .with_att_scale(true)
                .init(device)
                .into(),
        ]);
        let conv5 = Sequential::from(vec![
            InvertedResidualBlockConfig::new(256, 512, [3, 3], [1, 1])
                .with_padding([1, 1])
                .with_expansion(6)
                .with_bias(false)
                .init(device)
                .into(),
            MultiAspectGCAttentionConfig::new(512)
                .with_ratio(1.0 / 16.0)
                .with_headers(8)
                .with_att_scale(true)
                .init(device)
                .into(),
        ]);

        let position = PositionalEncodingConfig::new(512).init(device);

        Self {
            conv1,
            conv2,
            conv3,
            conv4,
            conv5,
            position,
            dropout: DropoutConfig::new(0.2).init(),
        }
    }

    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let feature = self.conv1.forward(input);
        let feature = self.conv2.forward(feature);
        let feature = self.conv3.forward(feature);
        let feature = self.conv4.forward(feature);
        let feature = self.conv5.forward(feature);

        let [batch, channels, height, width] = feature.dims(); // (B, C, H/8, W/4)
        let feature = feature.permute([0, 3, 2, 1]); // (B, H/8, W/4, C) different from the paper
        let feature = feature.reshape([batch, width * height, channels]);
        let feature = self.position.forward(feature);
        // The default implementation of PositionalEncoding in burn do not use dropout.
        // So, the dropout must be manually done here.
        let feature = self.dropout.forward(feature);

        return feature;
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{ndarray::NdArrayDevice, NdArray};

    use super::*;

    type MyBackend = NdArray;

    #[test]
    fn tt() {
        let device = NdArrayDevice::Cpu;
        let input: Tensor<MyBackend, 4> = Tensor::random(
            [2, 1, 48, 1000],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        );
        let encoder = Encoder::new(&device);
        let res = encoder.forward(input);

        println!("{}", res)
    }
}
