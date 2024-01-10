import torch
from torch.utils.data import DataLoader

from datasets.task_generation import TaskDataset


def eval_accuracy(dataset: TaskDataset, model, bs=100):
    correct = 0
    dataloader = DataLoader(dataset, batch_size=bs)

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x = x.unsqueeze(0)
            outputs = model(x).squeeze(0)  # (1, C, bs) -> (C, bs)
            preds = outputs.argmax(dim=0)
            correct += (preds == y).sum().item()

    return correct / len(dataset)
