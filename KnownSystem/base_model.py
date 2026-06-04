import inspect
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import sys

import collections
import inspect
from IPython import display

from matplotlib_inline.backend_inline import set_matplotlib_formats





"""
Based model: HyperParameters, Module, Regressor, 

"""
class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
            """Save function arguments into class attributes.
        
            Defined in :numref:`sec_utils`"""
            frame = inspect.currentframe().f_back
            _, _, _, local_vars = inspect.getargvalues(frame)
            self.hparams = {k:v for k, v in local_vars.items()
                            if k not in set(ignore+['self']) and not k.startswith('_')}
            for k, v in self.hparams.items():
                setattr(self, k, v)

#Change to make [TODO]: be Wandb compliant
class Module(nn.Module, HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""
    #Should change constructor here [TODO] make wandb compliant
    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()

    def loss(self, y_hat, y, pastwindow):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, 'net'), 'Neural network is defined'
        return self.net(X)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, 'trainer'), 'Trainer is not inited'
        self.board.xlabel = 'epoch'
        if train:
            x = self.trainer.train_batch_idx / \
                self.trainer.num_train_batches
            n = self.trainer.num_train_batches / \
                self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / \
                self.plot_valid_per_epoch
        y = value.detach().cpu().numpy()
        self.board.draw(x, y,
                        ('train_' if train else 'val_') + key,
                        every_n=int(n))

    def training_step(self, batch, pastwindow):
        l = self.loss(self(*batch[:-1]), batch[-1], pastwindow)
       
        return l


    def validation_step(self, batch, pastwindow):
        l = self.loss(self(*batch[:-1]), batch[-1], pastwindow)
       
        return l

    def configure_optimizers(self):
        """Defined in :numref:`sec_classification`"""
        return torch.optim.SGD(self.parameters(), lr=self.lr)

    def apply_init(self, inputs, init=None):
        """Defined in :numref:`sec_lazy_init`"""
        #Dry init. Usefull if custom init needed.
        self.forward(*inputs)
        if init is not None:
            self.net.apply(init)
    
    def count_parameters(self, trainable_only=False):
        if trainable_only:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())

class Regressor(Module):
    """The base class of regression models.
    """
    def validation_step(self, batch, pastwindow):
        Y_hat = self(*batch[:-1])
        loss = self.loss(Y_hat, batch[-1], pastwindow)

        #Need to be changed to Wandb plot okay
        #self.plot('loss', loss, train=False)
        #self.plot('acc', self.loss(Y_hat, batch[-1]), train=False)#Need change here [TODO]
        return loss

    def THD(self, Y_hat, Y, averaged=True): #[TODO]
        raise NotImplementedError

    def loss(self, Y_hat, Y, pastwindow, averaged=True):
        #Return MSE over a batch
        #Y_hat shape (batch_size, past_window + 400, 1)
        #Y shape (batch_size, 400, 1)

        return F.mse_loss(Y_hat, Y, reduction='mean' if averaged else 'none')

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = torch.randn(*X_shape)
        for layer in self.net:
            X = layer(X)
            print(layer.__class__.__name__, 'output shape:\t', X.shape)

class Seq2Seq(Regressor):

    def __init__(self, lr= 1e-3):
        super().__init__()
        self.lr = lr

    def configure_optimizers(self):
        # Adam with weight decay optimizer
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def predict_step(self, batch, *args):
        raise NotImplementedError
