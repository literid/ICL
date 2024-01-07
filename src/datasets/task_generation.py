import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.datasets import MNIST


class ProjectionTransform(nn.Module):
    def __init__(self, ndim, seed=None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.ndim = ndim
        mean = 0.0
        std = (1 / self.ndim) ** 1 / 2
        self.projection_matrix = torch.normal(mean, std, (self.ndim, self.ndim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        `B` - batch size, `N` - `ndim`
        Args: x: torch.Tensor shape `(B, N)`
        Returns: torch.Tensor shape `(B, N)`"""
        return torch.mm(self.projection_matrix, x).squeeze(-1)


class LabelPermuteTransform(nn.Module):
    def __init__(self, nclasses, seed=None) -> None:
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        self.nclasses = nclasses
        self.label_permutation = torch.randperm(self.nclasses)

    def forward(self, y: int):
        assert 0 <= y < self.nclasses
        return self.label_permutation[y]


class TaskDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        input_transform,
        label_transform,
    ) -> None:
        self.base_dataset = base_dataset
        self.input_transform = input_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.base_dataset)  # type: ignore

    def __getitem__(self, idx: int):
        x, y = self.base_dataset[idx]
        x = self.input_transform(x)
        y = self.label_transform(y)

        return x, y


def read_mnist(mnist_path):
    input_transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.1307,), (0.3081,)),
            T.Lambda(lambda x: x.reshape(-1, 1)),
        ]
    )
    mnist = MNIST(mnist_path, download=True, transform=input_transform)
    data = []
    dl = DataLoader(mnist, 1)

    for x, y in dl:
        x = x.squeeze(0)
        y = y.squeeze(0)
        data.append((x, y))

    return data


class MNISTTaskGenerator:
    def __init__(self, mnist_path, seed=None) -> None:
        self.mnist_path = mnist_path
        self.seed = seed

    def generate_tasks(self, num_tasks):
        base_dataset = MNIST(self.mnist_path, download=True)

        for i in range(num_tasks):
            if self.seed is not None:
                seed = i + self.seed
            else:
                seed = None
            input_transform = T.Compose(
                [
                    T.ToTensor(),
                    T.Normalize((0.1307,), (0.3081,)),
                    T.Lambda(lambda x: x.reshape(-1, 1)),
                    ProjectionTransform(28 * 28, seed),
                ]
            )
            label_transform = LabelPermuteTransform(10, seed)
            yield TaskDataset(base_dataset, input_transform, label_transform)


class PermuteProjectTaskGenerator:
    def __init__(self, data, projection_ndim, n_classes, seed=None) -> None:
        self.data = data
        self.projection_ndim = projection_ndim
        self.n_classes = n_classes
        self.seed = seed

    def generate_tasks(self, num_tasks):
        for i in range(num_tasks):
            if self.seed is not None:
                seed = i + self.seed
            else:
                seed = None

            input_transform = ProjectionTransform(self.projection_ndim, seed)
            label_transform = LabelPermuteTransform(self.n_classes, seed)
            yield TaskDataset(self.data, input_transform, label_transform)
