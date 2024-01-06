import torch
from torch.utils.data import DataLoader, Dataset


class MetaLearningDataset(Dataset):
    def __init__(self, tasks_num, tasks_generator, num_examples=100, seed=None):
        self.tasks_num = tasks_num
        self.task_datasets = [
            task_dataset
            for task_dataset in tasks_generator.generate_tasks(self.tasks_num)
        ]
        self.task_dataloaders = [
            DataLoader(dataset, batch_size=num_examples)
            for dataset in self.task_datasets
        ]
        self.task_iterators = [iter(dl) for dl in self.task_dataloaders]
        self.num_examples = num_examples

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def __len__(self):
        return self.tasks_num

    def __getitem__(self, idx):
        try:
            x, y = next(self.task_iterators[idx])
        except StopIteration:
            self.task_iterators[idx] = iter(self.task_dataloaders[idx])
            x, y = next(self.task_iterators[idx])

        return x, y
