use burn::backend::{libtorch::LibTorchDevice, Autodiff, LibTorch};
use cjk_ocr::{
    model::text_rec::OcrConfig,
    parse_config::OcrFullConfig,
    training::{self, TrainingConfig},
};

fn main() {
    let full_config = OcrFullConfig::from_yaml("./config.yaml");
    let ocr_config = OcrConfig::new(full_config.numbering + 3, full_config.numbering)
        .with_stacks(full_config.stacks)
        .with_n_heads(full_config.n_heads)
        .with_dropout(full_config.dropout)
        .with_bias(full_config.bias)
        .with_share_parameter(full_config.share_parameter)
        .with_use_feed_forward(full_config.use_feed_forward)
        .with_feed_forward_size(full_config.feed_forward_size);
    let training_config = TrainingConfig::new(
        ocr_config,
        full_config.num_workers,
        full_config.num_epochs,
        full_config.batch_size,
        full_config.seed,
        full_config.learning_rate,
        full_config.lexicon_path,
        full_config.numbering as u32,
        full_config.reserve_chars,
        full_config.data_root_path,
    );
    let devices: Vec<_> = (0..full_config.num_cuda)
        .map(|i| LibTorchDevice::Cuda(i))
        .collect();

    training::train::<Autodiff<LibTorch>>(&full_config.save_dir, training_config, devices);
}
