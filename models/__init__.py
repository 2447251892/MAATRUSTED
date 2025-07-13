# 文件名: models/__init__.py

from .model_with_clsdisc import (
    Generator_gru,
    Generator_lstm,
    Generator_transformer,
    Generator_transformer_deep,
    Generator_rnn,
    Discriminator3,
    Generator_dct,
    Generator_mpd,
    Generator_bigru,  # <-- 新增
    Generator_bilstm  # <-- 新增
)

__all__ = [
    "Generator_gru",
    "Generator_lstm",
    "Generator_transformer",
    "Generator_transformer_deep",
    "Generator_rnn",
    "Discriminator3",
    "Generator_dct",
    "Generator_mpd",
    "Generator_bigru",  # <-- 新增导出
    "Generator_bilstm"  # <-- 新增导出
]