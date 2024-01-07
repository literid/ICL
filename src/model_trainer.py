import torch


class Trainer:
    def __init__(
        self, model, optimizer, loss_fn, device, train_loader, val_loader=None
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
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
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                total_loss += loss.item()

        average_loss = total_loss / num_batches
        return average_loss

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            if self.val_loader is not None:
                val_loss = self.validate()
                print(
                    f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Validation Loss: {val_loss:.4f}"
                )
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f}")
