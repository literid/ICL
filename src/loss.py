import torch
import torch.nn as nn


class MetaLearningLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        r"""
        Computes the sum of the cross entropy losses
        for each prediction computed by prefix of data

        B - batch size, N - number of examples (size of support set + 1), C - number of classes
        Args:
            y_pred: torch.Tensor shape `(B, C, N)`, contain logits
                    e.g `y_pred[i, :, j]` are logits for class `y_true[i, j]`
            y_true: torch.Tensor shape `(B, N)`, contain true class labels in range `[0, C)`
        Returns:
            L: torch.Tensor mean (over `B * N`) loss
        """
        loss = self.ce(y_pred, y_true)
        return loss
