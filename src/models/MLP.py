import torch.nn as nn


class MetaLearningMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        `B` - batch size, `N` - number of examples, `D` - `input_dim`, `C` - `output_dim`
        Args: x: torch.Tensor shape `(B, N, D)`
        Returns: torch.Tensor shape `(B, C, N)`
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x.permute(0, 2, 1)
