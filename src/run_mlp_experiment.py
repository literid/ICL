import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets.dataset import MetaLearningDataset
from datasets.task_generation import PermuteProjectTaskGenerator, read_mnist
from eval import eval_accuracy_on_taskdataset
from loss import MetaLearningLoss
from model_trainer import Trainer
from models.MLP import MetaLearningMLP

mnist_path = "fake"
train_mnist_data_all = read_mnist(mnist_path, train=True)
test_mnist_data = read_mnist(mnist_path, train=False)
train_mnist_data, val_mnist_data = train_test_split(
    train_mnist_data_all, test_size=0.2, random_state=0
)

ndim = 28 * 28
n_classes = 10
train_tasks_generator = PermuteProjectTaskGenerator(
    train_mnist_data, ndim, n_classes, seed=0
)
val_tasks_generator = PermuteProjectTaskGenerator(
    val_mnist_data, ndim, n_classes, seed=0
)
test_tasks_generator = PermuteProjectTaskGenerator(
    test_mnist_data, ndim, n_classes, seed=0
)

tasks_num = 2**4
num_examples = 1
batch_size = 2**3

train_dataset = MetaLearningDataset(
    tasks_num, train_tasks_generator, num_examples=num_examples, seed=0
)
val_dataset = MetaLearningDataset(
    tasks_num, val_tasks_generator, num_examples=num_examples, seed=0
)
test_dataset = MetaLearningDataset(
    tasks_num, test_tasks_generator, num_examples=num_examples, seed=0
)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)


torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaLearningMLP(784, 1024, 10)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
trainer = Trainer(
    model, optimizer, MetaLearningLoss(), DEVICE, train_dataloader, val_dataloader, None
)
trainer.train(30000)

for i in range(tasks_num):
    train_acc = eval_accuracy_on_taskdataset(
        train_dataset.task_datasets[i], model, bs=1000
    )
    val_acc = eval_accuracy_on_taskdataset(val_dataset.task_datasets[i], model, bs=1000)
    test_acc = eval_accuracy_on_taskdataset(
        test_dataset.task_datasets[i], model, bs=1000
    )
    print(f"{train_acc=}, {val_acc=}, {test_acc=}")
# torch.save(model.state_dict(), "model0.pt")
