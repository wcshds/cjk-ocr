use std::{cmp, path::Path};

use burn::tensor::{backend::Backend, Data, Shape, Tensor};
use image::{GrayImage, Luma};

#[derive(Debug)]
pub struct ImageFactory {
    imgs: Vec<u8>,
    batch: usize,
    height: usize,
    width: usize,
}

impl ImageFactory {
    pub fn read_images<P: AsRef<Path>>(paths: &[P], height: u32, width: u32) -> ImageFactory {
        let batch = paths.len();
        let height_usize = height as usize;
        let width_usize = width as usize;
        let mut total_img_vec = Vec::with_capacity(batch * height_usize * width_usize);
        for path in paths {
            let img = GrayImage::from(image::open(path).unwrap());
            let [origin_height, origin_width] = [img.height(), img.width()];
            let img = image::imageops::resize(
                &img,
                cmp::min(
                    width,
                    (height as f64 * (origin_width as f64) / (origin_height as f64)) as u32,
                ),
                height,
                image::imageops::FilterType::Lanczos3,
            );
            let new_width = img.width();
            let padded = if new_width < width {
                let mut padded = image::GrayImage::from_pixel(width, height, Luma([0]));
                image::imageops::overlay(&mut padded, &img, 0, 0);
                padded
            } else {
                img
            };

            let mut img_vec = padded.into_vec();
            total_img_vec.append(&mut img_vec);
        }

        Self {
            imgs: total_img_vec,
            height: height_usize,
            width: width_usize,
            batch,
        }
    }

    pub fn to_tensor<B: Backend>(self, device: &B::Device) -> Tensor<B, 4> {
        let data = Data::new(
            self.imgs,
            Shape::new([self.batch, 1, self.height, self.width]),
        );
        let input = Tensor::<B, 4>::from_data(data.convert(), device);
        let input = (input - 127.5) / 127.5;

        input
    }
}
