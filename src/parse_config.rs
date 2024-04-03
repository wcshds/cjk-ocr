use std::{fs, path::Path};

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
struct OcrYaml {
    numbering: usize,
    reserve_chars: u32,
    stacks: usize,
    n_heads: usize,
    dropout: f64,
    bias: bool,
    share_parameter: bool,
    use_feed_forward: bool,
    feed_forward_size: usize,
}

#[derive(Serialize, Deserialize, Debug)]
struct TrainingYaml {
    num_cuda: usize,
    num_workers: usize,
    batch_size: usize,
    num_epochs: usize,
    seed: u64,
    learning_rate: f64,
    lexicon_path: String,
    data_root_path: String,
    save_dir: String,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "UPPERCASE")]
struct OcrTrainingConfigYaml {
    model: OcrYaml,
    training: TrainingYaml,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct OcrFullConfig {
    pub numbering: usize,
    pub reserve_chars: u32,
    pub stacks: usize,
    pub n_heads: usize,
    pub dropout: f64,
    pub bias: bool,
    pub share_parameter: bool,
    pub use_feed_forward: bool,
    pub feed_forward_size: usize,
    pub num_cuda: usize,
    pub num_workers: usize,
    pub num_epochs: usize,
    pub batch_size: usize,
    pub seed: u64,
    pub learning_rate: f64,
    pub lexicon_path: String,
    pub data_root_path: String,
    pub save_dir: String,
}

impl OcrFullConfig {
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Self {
        let path = fs::read_to_string(path).expect("training config does not exist");
        let yaml: OcrTrainingConfigYaml =
            serde_yaml::from_str(&path).expect("fail to read training config");

        Self {
            numbering: yaml.model.numbering,
            reserve_chars: yaml.model.reserve_chars,
            stacks: yaml.model.stacks,
            n_heads: yaml.model.n_heads,
            dropout: yaml.model.dropout,
            bias: yaml.model.bias,
            share_parameter: yaml.model.share_parameter,
            use_feed_forward: yaml.model.use_feed_forward,
            feed_forward_size: yaml.model.feed_forward_size,
            num_cuda: yaml.training.num_cuda,
            num_workers: yaml.training.num_workers,
            num_epochs: yaml.training.num_epochs,
            batch_size: yaml.training.batch_size,
            seed: yaml.training.seed,
            learning_rate: yaml.training.learning_rate,
            lexicon_path: yaml.training.lexicon_path,
            data_root_path: yaml.training.data_root_path,
            save_dir: yaml.training.save_dir,
        }
    }
}
