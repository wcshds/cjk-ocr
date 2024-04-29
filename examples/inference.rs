use std::{fs, time::Instant};

use burn::{
    backend::{libtorch::LibTorchDevice, LibTorch},
    module::Module,
    record::{
        BinFileRecorder, DoublePrecisionSettings, FullPrecisionSettings, PrettyJsonFileRecorder,
    },
    tensor::{activation, Data, Int, Tensor},
};
use cjk_ocr::{
    model::text_rec::OcrConfig,
    utils::{image_reader::ImageReader, label_converter::LabelConverter},
};

fn main() {
    type MyBackend = LibTorch;
    let device = LibTorchDevice::Cuda(0);

    // prepare data
    let img_factory = ImageReader::read_images(
        &[
            "images/temp.png",
            "images/temp2.png",
            "images/temp3.png",
            "images/temp4.png",
            "images/temp5.png",
            "images/temp6.png",
            "images/temp7.png",
            "images/temp8.png",
            "images/temp9.png",
            "images/temp11.png",
            "images/temp12.png",
        ],
        48,
        1000,
    );
    let images: Tensor<MyBackend, 4> = img_factory.to_tensor(&device);

    let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
    let converter = LabelConverter::new(&lexicon, 1000, 200000);

    // model
    let model = OcrConfig::new(1003, 1000)
        .with_dimensions(384)
        .with_stacks(3)
        .with_share_parameter(false)
        .with_use_feed_forward(true)
        .init::<MyBackend>(&device);
    let pjr = PrettyJsonFileRecorder::<DoublePrecisionSettings>::new();
    let model = model
        .load_file("./build/model_modified.json", &pjr, &device)
        .unwrap();

    let start = Instant::now();
    let batch_size = images.dims()[0];
    let max_text_length = 35;
    let to_return_label: Tensor<MyBackend, 2, Int> = Tensor::full(
        [batch_size, max_text_length * 2 + 2],
        converter.additional_symbols.pad_numbering,
        &device,
    );
    let mut probabilities: Tensor<MyBackend, 2> =
        Tensor::ones([batch_size, max_text_length * 2 + 2], &device);
    let mut to_return_label = to_return_label.slice_assign(
        [0..batch_size, 0..1],
        Tensor::from_data(
            Data::from([converter.additional_symbols.sos_numbering]).convert(),
            &device,
        )
        .unsqueeze_dim::<2>(0)
        .expand([batch_size as i32, -1]),
    );

    let encoded_res = model.encoder.forward(images);
    for i in 0..(max_text_length * 2 + 1) {
        let m_label = model
            .decoder
            .forward(encoded_res.clone(), to_return_label.clone());
        let m_probability = activation::softmax(m_label, 2);
        let (m_max_probs, m_next_word) = m_probability.max_dim_with_indices(2);
        let m_max_probs = m_max_probs.squeeze(2);
        let m_next_word = m_next_word.squeeze(2);
        to_return_label = to_return_label.slice_assign(
            [0..batch_size, (i + 1)..(i + 2)],
            m_next_word.slice([0..batch_size, i..(i + 1)]),
        );
        probabilities = probabilities.slice_assign(
            [0..batch_size, (i + 1)..(i + 2)],
            m_max_probs.slice([0..batch_size, i..(i + 1)]),
        );
    }
    println!("{}", probabilities);
    let to_return_label_vec: Vec<_> = to_return_label
        .iter_dim(0)
        .map(|each| each.to_data().value)
        .collect();
    println!("{:#?}", converter.decode(&to_return_label_vec));
    println!("time: {:.5}", start.elapsed().as_secs_f64());
}
