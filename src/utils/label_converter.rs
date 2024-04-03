use std::collections::HashMap;

use burn::tensor::{Data, Shape};
use num_traits::PrimInt;

use super::string_utils::StringUtil;

#[derive(Clone, Debug)]
pub struct AdditionalSymbols {
    pub pad_numbering: u32,
    pub sos_numbering: u32,
    pub eos_numbering: u32,
    pub unk: u32, // treat it as if it is a normal character
}

#[derive(Clone, Debug)]
pub struct LabelConverter {
    pub num_classes: usize,
    pub additional_symbols: AdditionalSymbols,
    numbering_half: u32,
    alphabet_mapper: HashMap<String, u32>,
    alphabet_inverse_mapper: HashMap<u32, String>,
}

impl LabelConverter {
    pub fn new(lexicon: &str, numbering: u32, reserver_chars: u32) -> Self {
        // now this tool only support two position encoding
        let mut position_count = 0;
        let mut accumulate = 1;
        loop {
            position_count += 1;
            accumulate *= numbering / 2;
            if accumulate > reserver_chars {
                break;
            }
        }

        assert!(
            position_count == 2,
            "2 surrogate number should be enough to represent `reserve_chars_rest` characters."
        );

        let mut alphabet_mapper = HashMap::new();
        alphabet_mapper.insert("<UNK>".to_string(), 0);
        for (idx, ch) in lexicon.strs().enumerate() {
            alphabet_mapper.insert(ch.to_string(), (idx + 1) as u32);
        }
        let alphabet_inverse_mapper: HashMap<u32, String> = alphabet_mapper
            .iter()
            .map(|(k, &v)| (v, k.to_string()))
            .collect();

        let pad_numbering = numbering;
        let sos_numbering = numbering + 1;
        let eos_numbering = numbering + 2;
        let unk = *alphabet_mapper.get("<UNK>").unwrap();

        let num_classes = alphabet_mapper.len() + 3;

        Self {
            num_classes,
            numbering_half: numbering / 2,
            alphabet_mapper,
            alphabet_inverse_mapper,
            additional_symbols: AdditionalSymbols {
                pad_numbering,
                sos_numbering,
                eos_numbering,
                unk,
            },
        }
    }

    pub fn encode_single(
        &self,
        text: &str,
        add_sos_eos: bool,
        real_max_length: Option<usize>,
    ) -> Vec<u32> {
        let preserved_length = match real_max_length {
            None => {
                if add_sos_eos {
                    (text.len() / 3 + 1) * 2 + 2
                } else {
                    (text.len() / 3 + 1) * 2
                }
            }
            Some(num) => num,
        };
        let mut res = Vec::with_capacity(preserved_length);
        if add_sos_eos {
            res.push(self.additional_symbols.sos_numbering as u32);
        }
        for item in text.strs() {
            let tmp = match self.alphabet_mapper.get(item) {
                Some(&idx) => self.convert_numbering(idx),
                None => self.convert_numbering(self.additional_symbols.unk),
            };
            res.extend(tmp);
        }
        if add_sos_eos {
            res.push(self.additional_symbols.eos_numbering as u32);
        }
        if let Some(_) = real_max_length {
            let remain_len = preserved_length - res.len();
            res.extend((0..remain_len).map(|_| self.additional_symbols.pad_numbering));
        }

        res
    }

    fn encode_internal<Output, F>(
        &self,
        texts: &[&str],
        add_sos_eos: bool,
        max_text_length: Option<usize>,
        deal_texts: F,
    ) -> Output
    where
        F: Fn(usize, usize, &[&str]) -> Output,
    {
        let additional_symbol_len = if add_sos_eos { 2 } else { 0 };
        let real_max_length = match max_text_length {
            Some(num) => num * 2 + additional_symbol_len,
            None => {
                texts.iter().map(|each| each.strs().count()).max().unwrap() * 2
                    + additional_symbol_len
            }
        };

        let batch = texts.len();
        deal_texts(batch, real_max_length, &texts)
    }

    pub fn encode(
        &self,
        texts: &Vec<&str>,
        add_sos_eos: bool,
        max_text_length: Option<usize>,
    ) -> Vec<Vec<u32>> {
        self.encode_internal(
            texts,
            add_sos_eos,
            max_text_length,
            |batch, real_max_length, texts| {
                Vec::from_iter(
                    (0..batch)
                        .map(|i| self.encode_single(texts[i], add_sos_eos, Some(real_max_length))),
                )
            },
        )
    }

    pub fn encode_to_1d_vec_with_shape(
        &self,
        texts: &[&str],
        add_sos_eos: bool,
        max_text_length: Option<usize>,
    ) -> (Vec<u32>, [usize; 2]) {
        let batch = texts.len();
        let deal_texts = |batch: usize, real_max_length: usize, texts: &[&str]| {
            let mut res = Vec::with_capacity(batch * real_max_length);
            for i in 0..batch {
                let tmp = self.encode_single(texts[i], add_sos_eos, Some(real_max_length));
                res.extend(tmp.iter().map(|&each| each as u32));
            }

            res
        };
        let res_vec = self.encode_internal(texts, add_sos_eos, max_text_length, deal_texts);

        let real_max_length = res_vec.len() / batch;
        (res_vec, [batch, real_max_length])
    }

    pub fn encode_to_tensor_data(
        &self,
        texts: &[&str],
        add_sos_eos: bool,
        max_text_length: Option<usize>,
    ) -> Data<u32, 2> {
        let (res_vec, shape) =
            self.encode_to_1d_vec_with_shape(texts, add_sos_eos, max_text_length);
        let data = Data::new(res_vec, Shape::new(shape));

        data
    }

    pub fn decode<INT: PrimInt>(&self, encoded_texts: &[Vec<INT>]) -> Vec<Vec<&str>> {
        let pad = self.additional_symbols.pad_numbering as u32;
        let sos = self.additional_symbols.sos_numbering as u32;
        let eos = self.additional_symbols.eos_numbering as u32;
        let mut res = Vec::with_capacity(encoded_texts.len());

        for text in encoded_texts {
            let length = text.len();

            let mut idx = 0;
            let mut tmp = Vec::with_capacity(length);
            while idx < length {
                let each = text[idx].to_u32().unwrap();
                match each {
                    _ if each == pad => tmp.push("<PAD>"),
                    _ if each == sos => tmp.push("<SOS>"),
                    _ if each == eos => tmp.push("<EOS>"),
                    _ => {
                        if idx == length - 1 {
                            tmp.push("<PAD>");
                            break;
                        } else {
                            let real_map_idx =
                                self.numbering_to_index([each, text[idx + 1].to_u32().unwrap()]);
                            match real_map_idx {
                                None => {
                                    tmp.push("<UNK>");
                                    idx += 1;
                                }
                                Some(num) => {
                                    let tmp_char = match self.alphabet_inverse_mapper.get(&num) {
                                        None => "<UNK>",
                                        Some(ch) => ch,
                                    };
                                    tmp.push(tmp_char);
                                    idx += 2;
                                }
                            }
                            continue;
                        }
                    }
                }

                idx += 1;
            }

            res.push(tmp);
        }

        res
    }

    #[inline]
    fn convert_numbering(&self, num: u32) -> [u32; 2] {
        [
            num / self.numbering_half + self.numbering_half,
            num % self.numbering_half,
        ]
    }

    #[inline]
    fn numbering_to_index(&self, numberings: [u32; 2]) -> Option<u32> {
        if numberings[0] < self.numbering_half || numberings[1] >= self.numbering_half {
            return None;
        }

        Some((numberings[0] - self.numbering_half) * self.numbering_half + numberings[1])
    }
}

#[cfg(test)]
mod test {
    use std::fs;

    use burn::{
        backend::{libtorch::LibTorchDevice, LibTorch},
        tensor::{Int, Tensor},
    };

    use super::*;
    use crate::utils::string_utils::StringUtil;

    #[test]
    fn test_str_iter() {
        let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
        for ch in lexicon.as_str().strs().skip(70000).take(500) {
            println!("{}", ch);
        }
    }

    #[test]
    fn test_converter_new() {
        let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
        let converter = LabelConverter::new(&lexicon, 1000, 200000);
        println!("{}", converter.alphabet_inverse_mapper[&0]);
        println!("{}", converter.alphabet_inverse_mapper[&1]);
        println!("{}", converter.alphabet_inverse_mapper[&2]);
        println!("{}", converter.alphabet_inverse_mapper[&3]);
        println!("{}", converter.alphabet_inverse_mapper[&4]);
        println!("{}", converter.alphabet_inverse_mapper[&5]);
        println!("{}", converter.alphabet_inverse_mapper[&1000]);
        println!("{}", converter.additional_symbols.pad_numbering);
        println!("{}", converter.additional_symbols.sos_numbering);
        println!("{}", converter.additional_symbols.eos_numbering);
        println!("{}", converter.additional_symbols.unk);
    }

    #[test]
    fn test_encode_single() {
        let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
        let converter = LabelConverter::new(&lexicon, 1000, 200000);

        let res = converter.encode_single("天氣😀", true, None);
        println!("{:#?}", "天氣😀".len());
        println!("{:#?}", res.capacity());
        println!("{:#?}", res);
    }

    #[test]
    fn test_encode() {
        let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
        let converter = LabelConverter::new(&lexicon, 1000, 200000);

        let res = converter.encode_to_tensor_data(&vec!["我是誰啊", "天氣😀"], true, Some(10));
        let tensor: Tensor<LibTorch, 2, Int> =
            Tensor::from_data(res.convert(), &LibTorchDevice::Cpu);
        println!("{}", tensor);
    }

    #[test]
    fn test_decode() {
        let lexicon = fs::read_to_string("./lexicon.txt").unwrap();
        let converter = LabelConverter::new(&lexicon, 1000, 200000);

        let res = converter.encode(&vec!["我是誰啊", "天氣😀"], true, Some(10));
        let de = converter.decode(&res);
        println!("{:#?}", de);
    }
}
