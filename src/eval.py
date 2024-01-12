import torch
from torch.utils.data import DataLoader

from datasets.dataset import MetaLearningDataset
from datasets.task_generation import TaskDataset


def eval_accuracy_on_taskdataset(dataset: TaskDataset, model, bs=100):
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


def eval_accuracy_on_metalearningdataset(
    dataset: MetaLearningDataset, model, tasks_num=None, taskdataset_bs=1000
) -> dict[int, float]:
    if tasks_num is None:
        tasks_num = len(dataset)
    task_to_acc = {}
    for i in range(tasks_num):
        acc = eval_accuracy_on_taskdataset(
            dataset.task_datasets[i], model, bs=taskdataset_bs
        )
        task_to_acc[i] = acc
    return task_to_acc
