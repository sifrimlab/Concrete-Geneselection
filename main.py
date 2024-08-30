import os
import torch
from torch import nn
import numpy as np
from scipy import sparse
import string
from utils import *
from model import CAE
import argparse

parser = argparse.ArgumentParser(description='CAE for gene panel selection.')
parser.add_argument('--infile', type=str, help='Input Scanpy object.')
parser.add_argument('--layer',default=None , help='Which adata expression layer to use')
parser.add_argument('--preselection_file',default='None' ,type=str, help='File with preselected genes to include in the panel.')
parser.add_argument('--outdir', type=str, help='Output directory.')
parser.add_argument('--outfile', type=str, help='Output file name.')
parser.add_argument('--nepochs', type=int, default=250, help='Number of epochs.')
parser.add_argument('-k', type=int, default=100, help='Number of genes to selected (preselected included).')
parser.add_argument('--lr_selection',type=float, default="0.005", help="learning rate for the concrete_selector_layer")
parser.add_argument('--start_temp', type=float, default=10.0, help='Starting temperature for CAE.')
parser.add_argument('--nhiddens', type=int, default=300, help='Number of hidden nodes for CAE decoder layers.')
parser.add_argument('--runnumber',type=int, nargs='?', default=1, help="runnumber")
parser.add_argument('--device',type=str, nargs='?', default="0", help="Device to train on")
parser.add_argument('--platform', default="None", help="Which spatial platform is being used")



args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.device

class Decoder(torch.nn.Module):
    def __init__(self, k, output_dim, dropout_rate=0.1, n_hiddens=args.nhiddens):
        super(Decoder, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(k, n_hiddens),
                   nn.ReLU(),
                   nn.Dropout(0.1))
        self.fc2 = nn.Sequential(nn.Linear(n_hiddens, n_hiddens),
                   nn.ReLU(),
                   nn.Dropout(0.1)) 
        self.fc3 = nn.Sequential(nn.Linear(n_hiddens, output_dim))
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    


epochs = args.nepochs
lr_selection = args.lr_selection
number_features = args.k
start_temp  = args.start_temp
input_file = args.infile
outfile_name = args.outfile
preselection_file = args.preselection_file
runnumber = args.runnumber
layer = args.layer
platform = args.platform

try:
    with open(preselection_file) as f:
        preselected_genes = f.read().splitlines()
except:
        preselected_genes=None
        print("Preselected Genes is NONE")
        print(f'Preselection file is {preselection_file}')


out_dir = args.outdir
os.makedirs(out_dir,exist_ok=True)
output_file = os.path.join(out_dir,F"{outfile_name}_{epochs}_Genes{number_features}_lr{lr_selection}_run{runnumber}")


adata, train_loader, test_loader, batch_size ,test_adata = read_data(input_file,layer=layer,platform=platform)

model = CAE(adata, k=number_features,preselected_genes=preselected_genes, decoder_model=Decoder(k=number_features, output_dim=adata.shape[1]), 
        n_epochs=epochs, batch_size=batch_size, start_temp=start_temp, lr_selection=0.005,)

epoch_hist = model.train_cae(adata, train_loader, test_loader)

save_selected_genes(model, output_file+".txt")
plot_loss(epoch_hist, output_file+".png")
print('Training complete')
