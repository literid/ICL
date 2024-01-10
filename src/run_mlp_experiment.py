import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.dataset import MetaLearningDataset
from datasets.task_generation import PermuteProjectTaskGenerator, read_mnist
from eval import eval_accuracy
from loss import MetaLearningLoss
from model_trainer import Trainer
from models.MLP import MetaLearningMLP

mnist_path = "fake"
mnist_data = read_mnist(mnist_path)
tasks_generator = PermuteProjectTaskGenerator(mnist_data, 28 * 28, 10, seed=0)

tasks_num = 2**4
num_examples = 10
batch_size = 2**3

train_dataset = MetaLearningDataset(
    tasks_num, tasks_generator, num_examples=num_examples, seed=0
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
val_dataloader = None

torch.manual_seed(0)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MetaLearningMLP(784, 1024, 10)
model.to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
trainer = Trainer(
    model, optimizer, MetaLearningLoss(), DEVICE, train_dataloader, val_dataloader
)
trainer.train(500)

acc = eval_accuracy(train_dataset.task_datasets[0], model, bs=1000)
print(acc)
# torch.save(model.state_dict(), "model0.pt")
