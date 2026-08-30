"""Paddle implementation of the HuBERT teacher used for Conan distillation."""

from pathlib import Path

import paddle
import paddle.nn as nn
import paddle.nn.functional as F


class HubertFeatureExtractor(nn.Layer):
    def __init__(self):
        super().__init__()
        specs = (
            (1, 512, 10, 5),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 3, 2),
            (512, 512, 2, 2),
            (512, 512, 2, 2),
        )
        blocks = []
        for index, (in_channels, out_channels, kernel_size, stride) in enumerate(specs):
            layers = [nn.Conv1D(in_channels, out_channels, kernel_size, stride=stride, bias_attr=False)]
            if index == 0:
                layers.extend([nn.Identity(), nn.GroupNorm(out_channels, out_channels)])
            blocks.append(nn.Sequential(*layers))
        self.conv_layers = nn.LayerList(blocks)

    def forward(self, source: paddle.Tensor) -> paddle.Tensor:
        x = source
        for layer in self.conv_layers:
            x = layer(x)
            x = F.gelu(x)
        return x


class HubertSelfAttention(nn.Layer):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: paddle.Tensor, padding_mask: paddle.Tensor | None = None
    ) -> paddle.Tensor:
        batch_size, length, _ = x.shape
        q = self.q_proj(x) * self.scaling
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape([batch_size, length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        k = k.reshape([batch_size, length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        v = v.reshape([batch_size, length, self.num_heads, self.head_dim]).transpose([0, 2, 1, 3])
        scores = paddle.matmul(q, k, transpose_y=True)
        if padding_mask is not None:
            scores = paddle.where(
                padding_mask.unsqueeze(1).unsqueeze(1),
                paddle.full_like(scores, -1e9),
                scores,
            )
        weights = F.softmax(scores, axis=-1)
        values = paddle.matmul(weights, v).transpose([0, 2, 1, 3])
        output = self.out_proj(values.reshape([batch_size, length, self.embed_dim]))
        if padding_mask is not None:
            output = paddle.where(padding_mask.unsqueeze(-1), paddle.zeros_like(output), output)
        return output


class HubertEncoderLayer(nn.Layer):
    def __init__(self, embed_dim: int = 768, ffn_dim: int = 3072, num_heads: int = 12):
        super().__init__()
        self.self_attn = HubertSelfAttention(embed_dim, num_heads)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

    def forward(
        self, x: paddle.Tensor, padding_mask: paddle.Tensor | None = None
    ) -> paddle.Tensor:
        x = self.self_attn_layer_norm(x + self.self_attn(x, padding_mask))
        if padding_mask is not None:
            x = paddle.where(padding_mask.unsqueeze(-1), paddle.zeros_like(x), x)
        x = self.final_layer_norm(x + self.fc2(F.gelu(self.fc1(x))))
        if padding_mask is not None:
            x = paddle.where(padding_mask.unsqueeze(-1), paddle.zeros_like(x), x)
        return x


class HubertPositionalConv(nn.Layer):
    def __init__(self, embed_dim: int = 768, kernel_size: int = 128, groups: int = 16):
        super().__init__()
        self.conv = nn.Layer()
        self.conv.add_parameter("weight_v", self.create_parameter([embed_dim, embed_dim // groups, kernel_size]))
        self.conv.add_parameter("weight_g", self.create_parameter([embed_dim]))
        self.conv.add_parameter("bias", self.create_parameter([embed_dim], is_bias=True))

    def forward(self, x: paddle.Tensor) -> paddle.Tensor:
        scale = paddle.rsqrt(paddle.sum(paddle.square(self.conv.weight_v), axis=[1, 2], keepdim=True))
        weight = self.conv.weight_v * self.conv.weight_g.reshape([-1, 1, 1]) * scale
        x = F.conv1d(x.transpose([0, 2, 1]), weight, self.conv.bias, padding=64, groups=16)
        return F.gelu(x[:, :, :-1].transpose([0, 2, 1]))


class HubertEncoder(nn.Layer):
    def __init__(self, num_layers: int = 9):
        super().__init__()
        self.pos_conv = HubertPositionalConv()
        self.layer_norm = nn.LayerNorm(768)
        self.layers = nn.LayerList([HubertEncoderLayer() for _ in range(num_layers)])

    def forward(
        self, x: paddle.Tensor, padding_mask: paddle.Tensor | None = None
    ) -> paddle.Tensor:
        if padding_mask is None:
            x = self.layer_norm(x + self.pos_conv(x))
        else:
            lengths = (~padding_mask).astype("int64").sum(axis=-1).numpy().tolist()
            features = x
            positioned = []
            for index, length in enumerate(lengths):
                value = self.pos_conv(features[index:index + 1, :int(length), :])
                positioned.append(value)
            x = paddle.zeros_like(features)
            for index, value in enumerate(positioned):
                x[index, :value.shape[1], :] = self.layer_norm(
                    features[index, :value.shape[1], :] + value[0]
                )
            x = paddle.where(padding_mask.unsqueeze(-1), paddle.zeros_like(x), x)
        for layer in self.layers:
            x = layer(x, padding_mask)
        return x


class HubertTeacher(nn.Layer):
    def __init__(self):
        super().__init__()
        self.feature_extractor = HubertFeatureExtractor()
        self.layer_norm = nn.LayerNorm(512)
        self.post_extract_proj = nn.Linear(512, 768)
        self.encoder = HubertEncoder(num_layers=9)
        self.final_proj = nn.Linear(768, 256)

    def encode_features(
        self, features: paddle.Tensor, padding_mask: paddle.Tensor | None = None
    ) -> paddle.Tensor:
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)
        features = self.encoder(features, padding_mask)
        output = self.final_proj(features)
        if padding_mask is not None:
            output = paddle.where(padding_mask.unsqueeze(-1), paddle.zeros_like(output), output)
        return output

    def forward(self, source: paddle.Tensor) -> paddle.Tensor:
        features = self.feature_extractor(source).transpose([0, 2, 1])
        return self.encode_features(features)

    def load_pretrained(self, path: str | Path) -> None:
        state_dict = paddle.load(str(path))
        missing, unexpected = self.set_state_dict(state_dict, use_structured_name=True)
        if missing or unexpected:
            raise ValueError(f"HuBERT checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def hubert_frame_count(n_samples: int) -> int:
    if n_samples < 400:
        return 0
    return (n_samples - 400) // 320 + 1
