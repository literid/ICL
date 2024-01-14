import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets.dataset import MetaLearningDataset
from datasets.task_generation import TaskDataset


def eval_accuracy_on_taskdataset(
    dataset: TaskDataset, model, device, num_examples, nclasses, bs=100
):
    correct = 0
    dataloader = DataLoader(dataset, batch_size=bs * num_examples)
    model.to(device)
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            if bs * num_examples != x.shape[0]:
                # in case of incomplete batch
                # we skip this batch
                continue
            else:
                x = x.reshape(bs, num_examples, -1)  # (B * N, D) -> (B, N ,D)
                y = y.reshape(bs, num_examples)  # (B * N) -> (B, N)

            # prepare input for transformer
            y_oh = F.one_hot(y, num_classes=nclasses)  # (B, N, C)
            zeros = torch.zeros(y_oh.shape[0], 1, y_oh.shape[-1])  # (B, 1, C)
            y_oh_shifted = torch.cat([zeros, y_oh[:, :-1, :]], dim=1)
            x = torch.cat([x, y_oh_shifted], dim=-1)  # (B, N, D + C)

            outputs = model(x)  # (B,C,N)
            preds = outputs.argmax(dim=1)  # (B, N)
            correct += (preds == y).sum().item()

    return correct / len(dataset)


def eval_accuracy_on_metalearningdataset(
    dataset: MetaLearningDataset, model, device, tasks_num=None, taskdataset_bs=100
) -> dict[int, float]:
    if tasks_num is None:
        tasks_num = len(dataset)
    task_to_acc = {}
    num_examples = dataset.num_examples
    nclasses = dataset.nclasses
    for i in range(tasks_num):
        acc = eval_accuracy_on_taskdataset(
            dataset.task_datasets[i],
            model,
            device,
            num_examples,
            nclasses,
            bs=taskdataset_bs,
        )
        task_to_acc[i] = acc
    return task_to_acc
