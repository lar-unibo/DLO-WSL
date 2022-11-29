import torch
import pytorch_lightning as pl
import torchvision


class Normalize(object):
    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        tensor = tensor.clone()
        if len(tensor.shape) == 4:
            for b in tensor:
                for t, m, s in zip(b, self.mean, self.std):
                    t.sub_(m).div_(s)
        elif len(tensor.shape) == 3:
            for t, m, s in zip(tensor, self.mean, self.std):
                t.sub_(m).div_(s)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor



class ResNetGaussian(pl.LightningModule):

    def __init__(self, lr = 0.01, output_dim=None, hidden_dim=None):
        super().__init__()
        self.encoder = torchvision.models.resnet18(pretrained=True)
        self.encoder.fc = torch.nn.Linear(512, 256)
        self.lin_out = torch.nn.Linear(256, output_dim)
        self.normalize = Normalize()

        self.lr = lr


    def forward(self, x):
        embedding = self.encoder(self.normalize(x)).relu()
        y = self.lin_out(embedding)
        return y

    def training_step(self, batch, batch_idx):

        image = batch['image']
        label = batch['label']

        output = self(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)

        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        image = batch['image']
        label = batch['label']

        output = self(image)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(output, label)
        self.log("val_loss", loss)
        return {"val_loss": loss}


    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_val_loss', avg_loss)
        return {'avg_val_loss': avg_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return {"optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2),
                    "monitor": "val_loss",
                    "frequency": 1
                }}