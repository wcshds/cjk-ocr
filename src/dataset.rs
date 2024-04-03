use std::{fs, path::Path};

use crate::utils::label_converter::LabelConverter;
use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    tensor::{backend::Backend, Data, Int, Shape, Tensor},
};
use image::imageops;
use serde::{Deserialize, Serialize};

pub struct TextImgBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> TextImgBatcher<B> {
    pub fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Clone, Debug)]
pub struct TextImgBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<TextImgItem, TextImgBatch<B>> for TextImgBatcher<B> {
    fn batch(&self, items: Vec<TextImgItem>) -> TextImgBatch<B> {
        let batch_size = items.len();
        let mut images = Vec::with_capacity(batch_size);
        let mut targets = Vec::with_capacity(batch_size);

        for item in items {
            let data_img = Data::new(
                item.image_raw,
                Shape::new([1, 1, item.image_height, item.image_width]),
            );
            let tensor_img =
                Tensor::<B, 4, Int>::from_data(data_img.convert(), &self.device).float();
            // range: [-1.0, 1.0]
            let tensor_img = ((tensor_img / 255) - 0.5) / 0.5;

            let length = item.target.len();
            let data_target = Data::new(item.target, Shape::new([1, length]));
            let tensor_target = Tensor::<B, 2, Int>::from_data(data_target.convert(), &self.device);

            images.push(tensor_img);
            targets.push(tensor_target);
        }

        let images = Tensor::cat(images, 0).to_device(&self.device);
        let targets = Tensor::cat(targets, 0).to_device(&self.device);

        TextImgBatch { images, targets }
    }
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct TextImgItem {
    // The raw vec of the image is passed here because GrayImage
    // does not directly implement the Serialize trait.
    pub image_raw: Vec<u8>,
    pub image_height: usize,
    pub image_width: usize,
    pub target: Vec<u32>,
}

pub struct TextImgDataset {
    root_path: String,
    path_and_label: Vec<(String, String)>,
    converter: LabelConverter,
}

impl TextImgDataset {
    pub fn new<P: AsRef<Path>>(
        label_file_path: P,
        root_path: P,
        converter: LabelConverter,
    ) -> Self {
        let data = fs::read_to_string(label_file_path).unwrap();
        let path_and_label = data
            .trim()
            .split("\n")
            .map(|row| {
                let mut tmp = row.trim().split("\t");
                let path = tmp.next().unwrap().replace("./data/", "./");
                let label = tmp.next().unwrap();

                (path, label.to_string())
            })
            .collect();

        Self {
            root_path: root_path.as_ref().to_str().unwrap().to_string(),
            path_and_label,
            converter,
        }
    }
}

impl Dataset<TextImgItem> for TextImgDataset {
    fn get(&self, index: usize) -> Option<TextImgItem> {
        if index >= self.len() {
            return None;
        }
        let (path, label) = &self.path_and_label[index];
        let path = Path::new(&self.root_path).join(path);

        let img = image::open(path).unwrap();
        let gray = imageops::grayscale(&img);
        let image_height = gray.height() as usize;
        let image_width = gray.width() as usize;
        let image_raw = gray.into_vec();
        let target = self.converter.encode_single(label, true, Some(102));

        Some(TextImgItem {
            image_raw,
            image_height,
            image_width,
            target,
        })
    }

    fn len(&self) -> usize {
        self.path_and_label.len()
    }
}
