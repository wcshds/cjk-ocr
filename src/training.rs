use std::{fs, path::Path};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    module::Module,
    optim::AdamConfig,
    record::{BinFileRecorder, FullPrecisionSettings},
    tensor::backend::AutodiffBackend,
    train::{
        metric::{AccuracyMetric, LossMetric},
        LearnerBuilder,
    },
};

use crate::{
    dataset::{TextImgBatcher, TextImgDataset},
    model::text_rec::OcrConfig,
    utils::label_converter::LabelConverter,
};

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub ocr_config: OcrConfig,
    pub num_workers: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub lexicon_path: String,
    pub numbering: u32, // high surrogate number + low surrogate number
    pub reserve_chars: u32,
    pub data_root_path: String,
}

pub fn train<B: AutodiffBackend>(save_dir: &str, config: TrainingConfig, devices: Vec<B::Device>) {
    B::seed(config.seed);
    let main_device = devices[0].clone();

    let lexicon = fs::read_to_string(config.lexicon_path).unwrap();
    let converter = LabelConverter::new(&lexicon, config.numbering, config.reserve_chars);

    let batcher_train = TextImgBatcher::<B>::new(main_device.clone());
    let batcher_valid = TextImgBatcher::<B::InnerBackend>::new(main_device.clone());

    let data_root_path = Path::new(&config.data_root_path);
    let train_labels_path = data_root_path.join("./train-labels.txt");
    let valid_labels_path = data_root_path.join("./valid-labels.txt");
    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            train_labels_path,
            data_root_path.to_path_buf(),
            converter.clone(),
        ));

    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(TextImgDataset::new(
            valid_labels_path,
            data_root_path.to_path_buf(),
            converter,
        ));

    let learner = LearnerBuilder::new(save_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .devices(devices)
        .num_epochs(config.num_epochs)
        .build(
            config.ocr_config.init::<B>(&main_device),
            AdamConfig::new().init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(
            format!("{save_dir}/model"),
            &BinFileRecorder::<FullPrecisionSettings>::new(),
        )
        .expect("Trained model should be saved successfully");
}
