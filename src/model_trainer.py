import torch


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device,
        train_loader,
        val_loader=None,
        early_stopping_patience=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

        if early_stopping_patience is not None:
            assert self.val_loader is not None, "Validation data not provided"
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
        if self.val_loader is None:
            raise ValueError("Validation data not provided")

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

    def train(
        self,
        num_epochs,
    ):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            if self.val_loader is not None:
                val_loss = self.validate()
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}"
                )
                if self.early_stopping is not None and self.early_stopping.should_stop(
                    val_loss
                ):
                    print("Early stopping triggered")
                    break
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")


class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.count = 0
        self.min_val_loss = float("inf")

    def should_stop(self, val_loss):
        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.count = 0
        else:
            self.count += 1
        return self.count >= self.patience
