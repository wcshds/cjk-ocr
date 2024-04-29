use burn::{
    config::Config,
    module::Module,
    nn::loss::CrossEntropyLoss,
    tensor::{
        backend::{AutodiffBackend, Backend},
        Int, Tensor,
    },
    train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep},
};

use crate::dataset::TextImgBatch;

use super::{
    decoder::tiny::{Decoder, DecoderConfig},
    encoder::mobilenet_v2::Encoder,
};

#[derive(Config, Debug)]
pub struct OcrConfig {
    num_classes: usize,
    padding_idx: usize,
    #[config(default = 512)]
    dimensions: usize,
    #[config(default = 3)]
    stacks: usize,
    #[config(default = 8)]
    n_heads: usize,
    #[config(default = 0.2)]
    dropout: f64,
    #[config(default = true)]
    bias: bool,
    #[config(default = false)]
    share_parameter: bool,
    #[config(default = true)]
    use_feed_forward: bool,
    #[config(default = 2048)]
    feed_forward_size: usize,
}

impl OcrConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> OCR<B> {
        OCR {
            encoder: Encoder::new(device, self.dimensions),
            decoder: DecoderConfig::new(self.num_classes, self.dimensions)
                .with_padding_idx(self.padding_idx)
                .with_stacks(self.stacks)
                .with_n_heads(self.n_heads)
                .with_dropout(self.dropout)
                .with_bias(self.bias)
                .with_share_parameter(self.share_parameter)
                .with_use_feed_forward(self.use_feed_forward)
                .with_feed_forward_size(self.feed_forward_size)
                .init(device),
            padding_idx: self.padding_idx,
        }
    }
}

#[derive(Module, Debug)]
pub struct OCR<B: Backend> {
    pub encoder: Encoder<B>,
    pub decoder: Decoder<B>,
    padding_idx: usize,
}

impl<B: Backend> OCR<B> {
    pub fn new(num_classes: usize, padding_idx: usize, device: &B::Device) -> Self {
        Self {
            encoder: Encoder::new(device, 512),
            decoder: DecoderConfig::new(num_classes, 512)
                .with_padding_idx(padding_idx)
                .with_stacks(3)
                .with_n_heads(8)
                .with_dropout(0.2)
                .with_bias(true)
                .with_share_parameter(false)
                .with_use_feed_forward(true)
                .with_feed_forward_size(2048)
                .init(device),
            padding_idx,
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>, targets: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let encoded_res = self.encoder.forward(images);
        let result = self.decoder.forward(encoded_res, targets);

        result
    }

    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 2, Int>,
        ignore_padding_idx: usize,
    ) -> ClassificationOutput<B> {
        let device = &self.devices()[0];
        let images = images.to_device(device);
        let targets = targets.to_device(device);

        let [batch, time_steps] = targets.dims();
        let target_right_shifted = targets.clone().slice([0..batch, 0..(time_steps - 1)]);
        let target_left_shifted = targets.slice([0..batch, 1..time_steps]);

        let output = self.forward(images, target_right_shifted);
        let num_classess = output.dims()[2] as i32;
        let output_reshape = output.reshape([-1, num_classess]);
        let target_reshape = target_left_shifted.reshape([-1]);
        let loss = CrossEntropyLoss::new(Some(ignore_padding_idx), device)
            .forward(output_reshape.clone(), target_reshape.clone());

        ClassificationOutput::new(loss, output_reshape, target_reshape)
    }
}

impl<B: AutodiffBackend> TrainStep<TextImgBatch<B>, ClassificationOutput<B>> for OCR<B> {
    fn step(&self, batch: TextImgBatch<B>) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets, self.padding_idx);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<TextImgBatch<B>, ClassificationOutput<B>> for OCR<B> {
    fn step(&self, batch: TextImgBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets, self.padding_idx)
    }
}

#[cfg(test)]
mod test {
    use burn::backend::{libtorch::LibTorchDevice, LibTorch};

    use super::*;

    type Backend = LibTorch;

    #[test]
    fn test_model() {
        let device = LibTorchDevice::Cpu;
        let ocr = OCR::<Backend>::new(1003, 1000, &device);

        let input = Tensor::random(
            [5, 1, 48, 1000],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let target = Tensor::arange(1..101, &device).reshape([5, 20]);

        let res = ocr.forward(input, target);
        println!("{}", res);
    }
}
