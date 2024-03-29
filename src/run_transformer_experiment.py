import argparse

import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
from datasets.dataset import MetaLearningDatasetForTransformers
from datasets.task_generation import PermuteProjectTaskGenerator, read_mnist
from eval import eval_accuracy_on_metalearningdataset
from loss import MetaLearningLoss
from model_trainer import Trainer
from models.Transformer import MetaLearningTransformer


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

    train_dataset = MetaLearningDatasetForTransformers(
        tasks_num, train_tasks_generator, n_classes, num_examples=num_examples, seed=0
    )
    val_dataset = MetaLearningDatasetForTransformers(
        tasks_num, val_tasks_generator, n_classes, num_examples=num_examples, seed=0
    )
    test_dataset = MetaLearningDatasetForTransformers(
        tasks_num, test_tasks_generator, n_classes, num_examples=num_examples, seed=0
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
    eval_every_n_epoch,
    min_delta,
):
    trainer = Trainer(
        model,
        optimizer,
        loss_function,
        device,
        train_dataloader,
        val_dataloader,
        early_stopping_patience,
        min_delta,
    )

    trainer.train(
        num_epochs, eval_tasks_num=eval_tasks_num, eval_every_n_epoch=eval_every_n_epoch
    )
    torch.save(model.state_dict(), save_path)
    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(save_path))
    wandb.log_artifact(artifact)


def main(args):
    ndim = 28 * 28
    nclasses = 10
    input_dim = ndim + nclasses

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
        nclasses,
        args.tasks_num,
        args.num_examples,
        args.val_size,
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.tasks_batch_size)
    val_dataloader = DataLoader(val_dataset, batch_size=args.tasks_batch_size)

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MetaLearningTransformer(
        args.nlayers, args.d_model, args.nhead, nclasses, input_dim
    )
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
        eval_every_n_epoch=args.eval_every_n_epoch,
        min_delta=args.min_delta,
    )

    test_metrics = eval_accuracy_on_metalearningdataset(
        test_dataset, model, DEVICE, args.eval_tasks_num
    )
    wandb.log({"test": {"acc": test_metrics}})
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
        default="Transformer",
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
        "--num_examples", type=int, default=100, help="Number of examples per task"
    )
    parser.add_argument(
        "--val_size", type=float, default=0.2, help="Validation set size"
    )
    parser.add_argument(
        "--tasks_batch_size", type=int, default=8, help="Tasks count per batch"
    )
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
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
    parser.add_argument(
        "--eval_every_n_epoch",
        type=int,
        default=100,
        help="Eval every n epochs",
    )
    parser.add_argument(
        "--min_delta", type=float, default=0.0, help="Minimum delta for early stopping"
    )
    parser.add_argument(
        "--nlayers", type=int, default=4, help="Number of layers in the model"
    )
    parser.add_argument(
        "--nhead", type=int, default=2, help="Number of head in the model"
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=32,
        help="Dimension of the model, must be divisible by nhead",
    )
    args = parser.parse_args()
    main(args)
