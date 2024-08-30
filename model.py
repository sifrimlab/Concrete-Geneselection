import os
import torch
from torch import nn, optim
import torch.nn.functional as F

from scipy import sparse
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import datetime
from tqdm import tqdm
import math
from CAE_pytorch.utils import *
import argparse


class mmpEarlyStopping:
    """
    Early stops the training if training/validation loss doesn't improve after a given patience.
    """
    def __init__(self):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.early_stop = False
        self.mmd_max =  1
    def __call__(self, mmd_value):
        if mmd_value >= self.mmd_max:
            self.early_stop = True

class ConcreteSelectLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_epochs, start_temp = 10.0, min_temp = 0.05, alpha = 0.99999, dev='cuda'):
        super(ConcreteSelectLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.start_temp = start_temp
        self.n_epochs = n_epochs
        self.min_temp = torch.tensor(min_temp)
        self.alpha = alpha
        self.epsilon = 1e-7
        self.logits = nn.init.xavier_normal_(nn.Parameter(torch.empty(self.input_dim, self.output_dim), requires_grad = True))
        self.dev = dev

    def get_temp(self, epoch, mode='exp'):
        schedules = {
            'exp': torch.max(self.min_temp, self.start_temp * ((self.min_temp / self.start_temp) ** (epoch / self.n_epochs))),
            'lin': torch.max(self.min_temp, self.start_temp - (self.start_temp - self.min_temp) * (epoch / self.n_epochs)),
            'cos': self.min_temp + 0.5 * (self.start_temp - self.min_temp) * (1. + np.cos(epoch * math.pi / self.n_epochs))
        }
        return schedules[mode]

    def forward(self, X, epoch=None, training = True):
#         uniform = torch.distributions.uniform.Uniform(low=1e-7, high=1.0).sample(self.logits.size())
#         gumbel = -torch.log(-torch.log(uniform)).to(self.dev)

        if training==True:
            temp = self.get_temp(epoch)
#             noisy_logits = (self.logits + gumbel) / temp
#             selection = F.softmax(noisy_logits, dim=0)
#             print(self.logits.shape)
            # Changed hard to True to test, check differences in results
            selection = F.gumbel_softmax(self.logits, tau=temp, hard=False, dim=0)
#             selection = F.gumbel_softmax(self.logits, tau=temp, hard=True, dim=0)
        else:
            selection = F.one_hot(torch.argmax(self.logits, dim=0), self.input_dim).float().T.to(self.dev)
            #selection = F.gumbel_softmax(self.logits, tau=temp, hard=True)

        Y = X @ selection
        
        return Y, selection

    def get_weights(self, epoch):
        temp = self.get_temp(epoch)
        return F.softmax(self.logits / temp, dim=0)

    def get_selected_feats(self):
        feats = torch.argmax(self.logits, dim=0)
        return feats.cpu().numpy()

class CAE(nn.Module):
    def __init__(self,
                 adata,
                 k=100, # number of selected features
                 preselected_genes = None,
                 decoder_model = None,
                 n_epochs = 300,
                 batch_size = None,
                 lr_selection = 0.01,
                 lr_decoder = 0.0001,
                 start_temp = 10.0,
                 min_temp = 0.05,
                 use_cuda = True,
                 **kwargs):
        super(CAE, self).__init__()
        self.input_dim=adata.shape[1]
        self.k = k
        self.preselected_genes = preselected_genes
        self.var_names = adata.var_names
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr_selection = lr_selection
        self.lr_decoder =  lr_decoder
        self.start_temp = start_temp
        self.min_temp = min_temp
        self.steps_per_epochs =  (adata.shape[0] + self.batch_size - 1) // self.batch_size
        self.alpha = np.exp(np.log(self.min_temp / self.start_temp) / (n_epochs * self.steps_per_epochs))
        if use_cuda:
            self.dev = 'cuda'
        else:
            self.dev = 'cpu'
        self.decoder = decoder_model.to(self.dev)

        if self.preselected_genes is not None:
            in_dim_cl = self.input_dim - len(self.var_names[self.var_names.isin(self.preselected_genes)])
            self.k = k - len(self.var_names[self.var_names.isin(self.preselected_genes)])
        else:
            in_dim_cl = self.input_dim
            self.k = k
        self.selector_layer = ConcreteSelectLayer(input_dim=in_dim_cl , output_dim=self.k, n_epochs=self.n_epochs, start_temp = self.start_temp, min_temp = self.min_temp, alpha = self.alpha, dev=self.dev).to(self.dev)

    def forward(self, x, epoch, training=True):
        if self.preselected_genes is not None:
            x_preselected = x[:, self.var_names.isin(self.preselected_genes)]
            x = x[:, ~self.var_names.isin(self.preselected_genes)]

        x_sel, selection = self.selector_layer(x, epoch, training)

        if self.preselected_genes is not None:
            x = self.decoder(torch.cat((x_sel, x_preselected), dim=-1))
        else:
            x = self.decoder(x_sel)
        return x, x_sel

    def coranking(self, y_true, y_pred):
        return rnx_auc_crm(compute_coranking_matrix(y_pred, data_hd=y_true))

    def losses(self, pred, real):
        return F.mse_loss(pred, real, reduction='mean')

    def train_cae(self, adata, train_loader, test_loader):
        self.to(self.dev)
        
        # Initialize Tensorboard summary writer
        writer = SummaryWriter('logs/cae_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

        epoch_hist = {}
        epoch_hist['train_loss'] = []
        epoch_hist['valid_loss'] = []
        epoch_hist['mean_max_prob'] = []
        epoch_hist['temperature'] = []
#         optimizer = optim.Adam([{'params': self.decoder.parameters(), 'lr': self.lr_decoder},
#                                 {'params': self.selector_layer.parameters(), 'lr': self.lr_selection}], lr=self.lr_decoder, weight_decay=5e-4)
        optimizer = optim.Adam(self.parameters(), lr = self.lr_selection)

        ES_callback = mmpEarlyStopping()

        it = 0
        for epoch in range(self.n_epochs):
            # Train
            self.train()
            mean_max_probs = torch.mean(torch.max(F.softmax(self.selector_layer.logits, dim=0), dim = 0).values)
            ES_callback(mean_max_probs)
            epoch_hist['mean_max_prob'].append(mean_max_probs.item())
            epoch_hist['temperature'].append(self.selector_layer.get_temp(epoch).item())

            for x in train_loader:
                optimizer.zero_grad()
#                 x = x.to(self.dev)
                loss_v = 0
                x_pred, _ = self.forward(x, epoch)
                loss = self.losses(x_pred, x)
                loss_v += loss.item()

                loss.backward()
                optimizer.step()
                writer.add_scalar('Loss', loss_v, it)
                it += 1

            # Get epoch loss
            epoch_loss = loss_v / len(train_loader)
            epoch_hist['train_loss'].append(epoch_loss)
            print('Epoch: ', epoch,'loss:',epoch_loss, ' - mean max of probabilities:', mean_max_probs.item(), '- temperature', self.selector_layer.get_temp(epoch).item())


            # Eval
            if test_loader:
                self.eval()
                torch.save(self.state_dict(), 'cae_params.pt')
                test_dict = self._test_model(test_loader, epoch)
                test_loss = test_dict['loss']
                epoch_hist['valid_loss'].append(test_loss)
                writer.add_scalar('Test/loss', test_loss, epoch+1)
                
                
            if ES_callback.early_stop:
                print('[Epoch %d] Early stopping (reached max_mean_prob target value)' % (epoch+1), flush=True)
                break

        return epoch_hist

    def _test_model(self, loader, epoch):
        """
        Test model on input loader.
        """
        self.eval()
        test_dict = {}
        loss = 0
        i = 0
        with torch.no_grad():
            for x in loader:
#                 x = x.to(self.dev)
                x_pred, x_sel = self.forward(x, epoch, training=False)
                loss += self.losses(x_pred, x).item()
                i += 1
        test_dict['loss'] = loss/i
        return test_dict
