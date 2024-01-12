import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
from datasets.dataset import MetaLearningDataset
from datasets.task_generation import PermuteProjectTaskGenerator, read_mnist
from loss import MetaLearningLoss
from model_trainer import Trainer
from models.MLP import MetaLearningMLP


def create_datasets(mnist_path, ndim, n_classes, tasks_num, num_examples, val_size=0.2):
    train_mnist_data_all = read_mnist(mnist_path, train=True)
    test_mnist_data = read_mnist(mnist_path, train=False)
    train_mnist_data, val_mnist_data = train_test_split(
        train_mnist_data_all, test_size=val_size, random_state=0
    )

    train_tasks_generator = PermuteProjectTaskGenerator(
        train_mnist_data, ndim, n_classes, seed=0
    )
    val_tasks_generator = PermuteProjectTaskGenerator(
        val_mnist_data, ndim, n_classes, seed=0
    )
    test_tasks_generator = PermuteProjectTaskGenerator(
        test_mnist_data, ndim, n_classes, seed=0
    )

    train_dataset = MetaLearningDataset(
        tasks_num, train_tasks_generator, num_examples=num_examples, seed=0
    )
    val_dataset = MetaLearningDataset(
        tasks_num, val_tasks_generator, num_examples=num_examples, seed=0
    )
    test_dataset = MetaLearningDataset(
        tasks_num, test_tasks_generator, num_examples=num_examples, seed=0
    )

    return train_dataset, val_dataset, test_dataset


def train_model(
    model,
    optimizer,
    loss_function,
    device,
    train_dataloader,
    val_dataloader,
    num_epochs,
    early_stopping_patience,
    save_path,
    eval_tasks_num,
):
    trainer = Trainer(
        model,
        optimizer,
        loss_function,
        device,
        train_dataloader,
        val_dataloader,
        early_stopping_patience,
    )

    trainer.train(num_epochs, eval_tasks_num=eval_tasks_num)
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(save_path))
    wandb.log_artifact(artifact)


def main(args):
    ndim = 28 * 28
    n_classes = 10

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            "architecture": args.architecture_name,
            "dataset": args.dataset_name,
            **vars(args),
        },
    )

    train_dataset, val_dataset, test_dataset = create_datasets(
        args.mnist_path,
        ndim,
        n_classes,
        args.tasks_num,
        args.num_examples,
        args.val_size,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.tasks_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.tasks_batch_size)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaLearningMLP(ndim, args.hidden_dim, n_classes)
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    train_model(
        model,
        optimizer,
        MetaLearningLoss(),
        DEVICE,
        train_dataloader,
        val_dataloader,
        num_epochs=args.num_epochs,
        early_stopping_patience=args.early_stopping_patience,
        save_path=args.save_path,
        eval_tasks_num=args.eval_tasks_num,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP MetaLearningModel Training")
    parser.add_argument(
        "--wandb_project",
        type=str,
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        help="Wandb run name",
    )
    parser.add_argument(
        "--architecture_name",
        type=str,
        help="Logging architecture name in wandb",
        default="MLP",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Logging dataset name in wandb",
        default="MNIST",
    )
    parser.add_argument(
        "--mnist_path",
        type=str,
        help="Path to MNIST dataset",
    )
    parser.add_argument("--tasks_num", type=int, default=4, help="Number of tasks")
    parser.add_argument(
        "--num_examples", type=int, default=1, help="Number of examples per task"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation set size"
    )
    parser.add_argument(
        "--tasks_batch_size", type=int, default=2, help="Tasks count per batch"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=32, help="Hidden dimension of the model"
    )
    parser.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    parser.add_argument(
        "--num_epochs", type=int, default=5000, help="Number of training epochs"
    )
    parser.add_argument(
        "--eval_tasks_num", type=int, default=2, help="Number of evaluation tasks"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Patience for early stopping",
    )

    args = parser.parse_args()
    main(args)
