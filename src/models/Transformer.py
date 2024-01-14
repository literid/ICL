import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=None,  # if None then dim_feedforward = d_model * 4
        activation="relu",
        batch_first=True,
        norm_first=False,
    ) -> None:
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
        self.dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
            batch_first=batch_first,
            norm_first=norm_first,
        )

    def forward(self, x):
        bs, seq, feat = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(seq)
        return self.dec_layer(x, mask, is_causal=True)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=256, dropout_p=0.1):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.randn(max_seq_len, d_model), requires_grad=True
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        """
        `B` - batch size, `N` - sequence length (number of examples), `D` - `d_model`,
        Args: x: torch.Tensor shape `(B, N, D)`
        Returns: torch.Tensor shape `(B, N, D)`
        """
        _, N, *_ = x.shape
        return self.dropout(self.embeddings[:N]) + x


class ClassificationHead(nn.Module):
    def __init__(self, d_model, nclasses) -> None:
        super().__init__()
        self.d_model = d_model
        self.nclasses = nclasses

        self.fc1 = nn.Linear(d_model, nclasses)

    def forward(self, x):
        """
        Return logits
        `D` - `d_model`, `C` - `nclasses`
        Args: x: torch.Tensor shape `(*, D)`
        Returns: torch.Tensor shape `(*, C)`
        """
        return self.fc1(x)


class MetaLearningTransformer(nn.Module):
    def __init__(
        self,
        nlayers,
        d_model,
        nhead,
        nclasses,
        input_dim,
        dim_feedforward=None,
        activation="relu",
        batch_first=True,
        norm_first=False,
        max_seq_len=None,
    ) -> None:
        super().__init__()
        if max_seq_len is None:
            max_seq_len = 256
        self.max_seq_len = 256
        self.nclasses = nclasses

        self.projection_layer = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.layers = nn.ModuleList(
            DecoderLayer(
                d_model,
                nhead,
                dim_feedforward,
                activation=activation,
                batch_first=batch_first,
                norm_first=norm_first,
            )
            for _ in range(nlayers)
        )
        self.classification_head = ClassificationHead(d_model, nclasses)

    def forward(self, x):
        """
        `B` - batch size, `N` - number of examples, `D` - `d_model`, `C` - `nclasses`
        Args: x: torch.Tensor shape `(B, N, D)`
        Returns: torch.Tensor shape `(B, C, N)`
        """
        x = self.projection_layer(x)
        x = self.positional_encoding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.classification_head(x)
        return x.permute(0, 2, 1)
