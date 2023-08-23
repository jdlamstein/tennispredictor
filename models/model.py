import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy
from torchvision import transforms
__author__='Josh Lamstein'

class Model(LightningModule):
    def __init__(self, learning_rate=2e-4):

        super().__init__()
        self.save_hyperparameters()
        # Set our init args as class attributes
        self.learning_rate = learning_rate

        # Hardcode some dataset specific attributes
        self.features = 36
        self.num_classes = 2
        # self.transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.1307,), (0.3081,)),
        #     ]
        # )

        # Define PyTorch model
        self.model = nn.Sequential(
            nn.Linear(self.features, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

        self.val_accuracy = Accuracy(task='binary')
        self.test_accuracy = Accuracy(task='binary')

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.argmax(logits, dim=1)
        gt = torch.argmax(y, dim=1)
        self.val_accuracy.update(preds, gt)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())
        preds = torch.argmax(logits, dim=1)
        gt = torch.argmax(y, dim=1)
        self.test_accuracy.update(preds, gt)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################
    #
    # def prepare_data(self):
    #     # download
    #     MNIST(self.data_dir, train=True, download=True)
    #     MNIST(self.data_dir, train=False, download=True)
    #
    # def setup(self, stage=None):
    #
    #     # Assign train/val datasets for use in dataloaders
    #     if stage == "fit" or stage is None:
    #         mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
    #         self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])
    #
    #     # Assign test dataset for use in dataloader(s)
    #     if stage == "test" or stage is None:
    #         self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform)
    #
    # def train_dataloader(self):
    #     return DataLoader(self.mnist_train, batch_size=BATCH_SIZE)
    #
    # def val_dataloader(self):
    #     return DataLoader(self.mnist_val, batch_size=BATCH_SIZE)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=BATCH_SIZE)