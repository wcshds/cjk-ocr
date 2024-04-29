use burn::{
    config::Config,
    module::Module,
    nn::{
        attention::{
            generate_autoregressive_mask, MhaInput, MultiHeadAttention, MultiHeadAttentionConfig,
        },
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear,
        LinearConfig, PositionalEncoding, PositionalEncodingConfig,
    },
    tensor::{backend::Backend, Int, Tensor},
};

use crate::burn_ext::position_wise_feed_forward::{
    PositionWiseFeedForward, PositionWiseFeedForwardConfig,
};

#[derive(Module, Debug)]
pub struct Decoder<B: Backend> {
    pub embedding: Embedding<B>,
    pub position: PositionalEncoding<B>,
    pos_dropout: Dropout,
    pub masked_attention: Vec<MultiHeadAttention<B>>,
    pub attention: Vec<MultiHeadAttention<B>>,
    pub layernorm: LayerNorm<B>,
    dropout: Dropout,
    pub position_feed_forward: Option<Vec<PositionWiseFeedForward<B>>>,
    pub generator: Linear<B>,
    padding_idx: usize,
    sqrt_model_size: f64,
    stacks: usize,
}

impl<B: Backend> Decoder<B> {
    pub fn forward(&self, encoded_res: Tensor<B, 3>, target: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward_with_iteration(encoded_res, target, self.stacks)
    }

    pub fn forward_with_iteration(
        &self,
        encoded_res: Tensor<B, 3>,
        target: Tensor<B, 2, Int>,
        iteration: usize,
    ) -> Tensor<B, 3> {
        let device = encoded_res.device();
        let target = target.to_device(&device);
        let [batch, target_length] = target.dims();
        let target_embed = self
            .embedding
            .forward(target.clone())
            .mul_scalar(self.sqrt_model_size);
        let target_embed = self.position.forward(target_embed);
        let target_embed = self.pos_dropout.forward(target_embed);

        let mut output = target_embed;
        for i in 0..iteration {
            let (masked_attention, attention, position_feed_forward) = match (
                self.masked_attention.len(),
                self.attention.len(),
                &self.position_feed_forward,
            ) {
                (1, 1, None) => (&self.masked_attention[0], &self.attention[0], None),
                (1, 1, Some(ff)) if ff.len() == 1 => {
                    (&self.masked_attention[0], &self.attention[0], Some(&ff[0]))
                }
                (_, _, None) => (&self.masked_attention[i], &self.attention[i], None),
                (_, _, Some(ff)) => (&self.masked_attention[i], &self.attention[i], Some(&ff[i])),
            };
            let normed_output = self.layernorm.forward(output.clone());
            output = output.clone()
                + self.dropout.forward(
                    masked_attention
                        .forward(
                            MhaInput::new(
                                normed_output.clone(),
                                normed_output.clone(),
                                normed_output,
                            )
                            .mask_pad(target.clone().equal_elem(self.padding_idx as i64))
                            .mask_attn(generate_autoregressive_mask(batch, target_length, &device)),
                        )
                        .context,
                );
            let normed_output = self.layernorm.forward(output.clone());
            output = output
                + self.dropout.forward(
                    attention
                        .forward(MhaInput::new(
                            normed_output,
                            encoded_res.clone(),
                            encoded_res.clone(),
                        ))
                        .context,
                );
            if let Some(position_feed_forward) = position_feed_forward {
                let normed_output = self.layernorm.forward(output.clone());
                output = output
                    + self
                        .dropout
                        .forward(position_feed_forward.forward(normed_output))
            }
        }

        let feature = self.layernorm.forward(output);
        let output = self.generator.forward(feature);

        output
    }
}

#[derive(Config, Debug)]
pub struct DecoderConfig {
    n_classes: usize,
    dimensions: usize,
    #[config(default = "true")]
    use_feed_forward: bool,
    #[config(default = "2048")]
    feed_forward_size: usize,
    #[config(default = "0")]
    padding_idx: usize,
    #[config(default = "3")]
    stacks: usize,
    #[config(default = "8")]
    n_heads: usize,
    #[config(default = "0.2")]
    dropout: f64,
    #[config(default = "true")]
    bias: bool,
    #[config(default = "false")]
    share_parameter: bool,
}

impl DecoderConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Decoder<B> {
        let embedding = EmbeddingConfig::new(self.n_classes, self.dimensions).init(device);
        let position = PositionalEncodingConfig::new(self.dimensions).init(device);
        let pos_dropout = DropoutConfig::new(0.2).init();

        let get_mha = |dimensions, n_heads, dropout| {
            MultiHeadAttentionConfig::new(dimensions, n_heads)
                .with_dropout(dropout)
                .init(device)
        };
        let get_position_wise_feed_forward = || {
            PositionWiseFeedForwardConfig::new(self.dimensions, self.feed_forward_size)
                .with_dropout(self.dropout)
                .init(device)
        };

        let stacks = if self.stacks > 1 { self.stacks } else { 1 };
        let mut masked_attention = vec![get_mha(self.dimensions, self.n_heads, self.dropout)];
        let mut attention = vec![get_mha(self.dimensions, self.n_heads, self.dropout)];
        let mut position_feed_forward = if self.use_feed_forward {
            Some(vec![get_position_wise_feed_forward()])
        } else {
            None
        };
        if !self.share_parameter {
            for _ in 1..self.stacks {
                masked_attention.push(get_mha(self.dimensions, self.n_heads, self.dropout));
                attention.push(get_mha(self.dimensions, self.n_heads, self.dropout));
                if let Some(ref mut position_feed_forward) = position_feed_forward {
                    position_feed_forward.push(get_position_wise_feed_forward());
                }
            }
        }

        let layernorm = LayerNormConfig::new(self.dimensions)
            .with_epsilon(1e-6)
            .init(device);
        let dropout = DropoutConfig::new(self.dropout).init();
        let generator = LinearConfig::new(self.dimensions, self.n_classes).init(device);

        Decoder {
            embedding,
            position,
            pos_dropout,
            masked_attention,
            attention,
            layernorm,
            dropout,
            generator,
            position_feed_forward,
            padding_idx: self.padding_idx,
            sqrt_model_size: (self.dimensions as f64).sqrt(),
            stacks,
        }
    }
}
