#%%
import os
import time
import torch
import torchvision
import torch as th
import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from schedulers import CosineSchedulerWithWarmupStart
import matplotlib.pyplot as plt
import numpy as np
from vit import VIT
from transforms import vit_val_transform as val_transform
from transforms import vit_train_transform as train_transform
from utils import load_model, save_model, seed_everything
import argparse
from torch.utils.tensorboard import SummaryWriter
from communicators import LastPass, WeightedPass, DensePass, AttentionPass



def get_weighted_matrix(model):
    padded_length = len(model.communicators)
    matrix = [
        comm.weighter[0, 0].detach().cpu().numpy().tolist() + [0]*(padded_length-i) 
        for i, comm in enumerate(model.communicators)
    ]
    matrix = torch.tensor(matrix).numpy().T
    matrix = np.flip(matrix, axis=0)
    matrix = matrix / np.sum(matrix, axis=0, keepdims=True)
    # Plot the matrix
    plt.imshow(matrix, cmap='Greys', aspect='auto')

    # Add the x-axis label
    plt.ylabel('what they value most, normalised')
    ylen = matrix.shape[0]
    ticks = ['input'] + [str(i) + ' layer' for i in range(0, ylen-1)]
    plt.yticks(range(ylen), reversed(ticks))

    # Add the y-axis label
    plt.xlabel('Layers')
    xlen = matrix.shape[1]
    plt.xticks(range(xlen), range(xlen))

    # Add the title
    plt.title('How layers weight their inputs')

    # Display the plot
    plt.show()
    print(matrix)




# %%



# Main start #################################################################################

if __name__ == '__main__':
    # parse arguments
    device = 'cpu'
    print("using", device)
    model_name = 'tiny'
    communicators = 'weighted'

    size2params = {

        'tiny':{
            'd_model' : 64,
            'n_layers' : 4,
            'n_heads' : 4,
            'patch_size' : 4,
        },
        'small':{
            'd_model' : 128,
            'n_layers' :  8,
            'n_heads' : 8,
            'patch_size' : 4,
        },
        'base':{
            'd_model' : 192,
            'n_layers' : 12,
            'n_heads' : 12,
            'patch_size' : 4,
        },
        'large':{
            'd_model' : 256,
            'n_layers' : 12,
            'n_heads' : 16,
            'patch_size' : 4,
        },
        'huge':{
            'd_model' : 512,
            'n_layers' : 12,
            'n_heads' : 16,
            'patch_size' : 4,
        },
    }

    communicators = {
        'normal':LastPass,
        'weighted':WeightedPass,
        'dense':DensePass,
        # 'attention':AttentionPass
    }


    # %%
    # Load everything #################################################################################
    model_path = 'model_checkpoints/huge_weighted_1679187281.pt'
    model_path = 'model_checkpoints/tiny_weighted_1679172096.pt'
    communicators = communicators['weighted']
    model = VIT(**size2params[model_name], layer_communicators=communicators, n_heads_communicator=4)
    load_model(model, model_path)

    print(f'{model_name} {communicators} has, {model.get_number_of_parameters()/10**6} M parameters')
    get_weighted_matrix(model)


# %%
for size, kwargs in size2params.items():
    for comm_name, comm in communicators.items():
        model = VIT(**kwargs, layer_communicators=comm, n_heads_communicator=4)
        print(f'Vit {size} {comm_name} has, {model.get_number_of_parameters()/10**6} M parameters')

# %%

import matplotlib.pyplot as plt

# Define the data for the scatter plot
normal_data = [(0.208, 0.4599), (1.602, 0.5734), (5.362, 0.6414), (9.509, 0.6557), (37.892, 0.6645)]
weighted_data = [(0.208, 0.4574), (1.602, 0.5578), (5.362, 0.6127), (9.509, 0.625), (37.892, 0.6430)]
dense_data = [(0.249, 0.4740), (2.192, 0.5901), (8.240, 0.6384), (14.624, 0.6449), (58.345, 0.6430)]

# Create a new figure
fig, ax = plt.subplots()

# Plot the scatter plot for each group with lines connecting the points
ax.plot([x[0] for x in normal_data], [x[1] for x in normal_data], 'r-o', label='Normal')
ax.plot([x[0] for x in weighted_data], [x[1] for x in weighted_data], 'g-o', label='Weighted')
ax.plot([x[0] for x in dense_data], [x[1] for x in dense_data], 'b-o', label='Dense')

# Set the x and y axis labels and the plot title
ax.set_xlabel('Number of Parameters (M)')
ax.set_ylabel('CIFAR10 Accuracy')
ax.set_title('Accuracy by Method and Number of Parameters')

# Add a legend to the plot
ax.legend()

# Show the plot
plt.show()
