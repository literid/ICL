import numpy as np
import torch

import wandb
from datasets.dataset import MetaLearningDataset
from eval import eval_accuracy_on_metalearningdataset


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        train_loader,
        val_loader,
        early_stopping_patience=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        if early_stopping_patience is not None:
            self.early_stopping = EarlyStopping(early_stopping_patience)
        else:
            self.early_stopping = None

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for inputs, targets in self.train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / num_batches
        return average_loss

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / num_batches
        return average_loss

    def train(self, num_epochs, eval_tasks_num=None, eval_every_n_epoch=100):
        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            log_dict = {
                "epoch": epoch,
                "train": {"loss": train_loss},
                "val": {"loss": val_loss},
            }

            if epoch % eval_every_n_epoch == 0:
                train_metrics, val_metrics = self.eval_metrics(eval_tasks_num)
                log_dict["train"].update(train_metrics)
                log_dict["val"].update(val_metrics)
                if self.early_stopping is not None:
                    val_metric = np.mean(list(val_metrics["acc"].values()))
                    if self.early_stopping.should_stop(val_metric):
                        print(f"Early stopping triggered at epoch {epoch}")
                        wandb.log(log_dict)
                        break

            wandb.log(log_dict)

    def eval_metrics(self, tasks_num=None):
        train_metrics = {}
        val_metrics = {}
        train_metrics["acc"] = eval_accuracy_on_metalearningdataset(
            self.train_loader.dataset, self.model, self.device, tasks_num
        )
        val_metrics["acc"] = eval_accuracy_on_metalearningdataset(
            self.val_loader.dataset, self.model, self.device, tasks_num
        )
        return train_metrics, val_metrics


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.max_metric = float("-inf")

    def should_stop(self, val_metric):
        if val_metric - self.max_metric > self.min_delta:
            self.max_metric = val_metric
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience
