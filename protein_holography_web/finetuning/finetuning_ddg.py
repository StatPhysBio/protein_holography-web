

import os, sys
import json
import gzip, pickle

from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import torch
from copy import deepcopy

from torch.utils.data import DataLoader

import argparse

from protein_holography_web.models import CGNet, SO3_ConvNet
from protein_holography_web.cg_coefficients import get_w3j_coefficients

from protein_holography_web.inference.hcnn_inference import get_channels, get_data_irreps

from sklearn.metrics import classification_report, roc_auc_score
from scipy.special import softmax

from protein_holography_web.utils.data import put_dict_on_device
from protein_holography_web.utils.protein_naming import ol_to_ind_size

from e3nn import o3
from torch.utils.data import Dataset
from typing import *


def hcnn_model_init(model_dir):

    with open(f'{model_dir}/hparams.json', 'r') as f:
        hparams = json.load(f)

    data_irreps, ls_indices = get_data_irreps(hparams)

    # setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Running on %s.' % (device), flush=True)

    # load w3j coefficients
    w3j_matrices = get_w3j_coefficients(hparams['lmax'])
    for key in w3j_matrices:
        # if key[0] <= hparams['net_lmax'] and key[1] <= hparams['net_lmax'] and key[2] <= hparams['net_lmax']:
        if device is not None:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float().to(device)
        else:
            w3j_matrices[key] = torch.tensor(w3j_matrices[key]).float()
        w3j_matrices[key].requires_grad = False
    
    # load model and pre-trained weights
    if hparams['model_type'] == 'cgnet':
        model = CGNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    elif hparams['model_type'] == 'so3_convnet':
        model = SO3_ConvNet(data_irreps, w3j_matrices, hparams['model_hparams'], normalize_input_at_runtime=hparams['normalize_input']).to(device)
    else:
        raise NotImplementedError()
    model.load_state_dict(torch.load(os.path.join(model_dir, 'lowest_valid_loss_model.pt'), map_location=device))
    
    num_params = 0
    for param in model.parameters():
        num_params += torch.flatten(param.data).shape[0]
    print('There are %d parameters' % (num_params), flush=True)

    return hparams, data_irreps, model, device


def fermi_transform(ddg: np.ndarray, beta: float = 0.4, alpha: float = 3.0) -> np.ndarray:
    return 1 / (1 + np.exp(-beta*(ddg - alpha)))


def finetune(config_file):
    






