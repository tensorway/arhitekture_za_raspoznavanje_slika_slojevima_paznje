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
from communicators import LastPass, WeightedPass, DensePass, AttentionPass, FeatureWiseWeightedPass, BottleneckDensePass



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
        'feature_wise_weighted':FeatureWiseWeightedPass,
        'bottlenecked_dense':BottleneckDensePass,
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


#%%
import pandas as pd
import matplotlib.pyplot as plt

# load the CSV file into a pandas DataFrame
df = pd.read_csv('tmp.csv', header=0, skip_blank_lines=True, skipinitialspace=True)
# print(df)

# set the communicators column as categorical
df['communicators'] = pd.Categorical(df['communicators'])

# initialize the plot
fig, ax = plt.subplots(figsize=(10, 6))

# iterate over the unique communicators
for communicator in df['communicators'].unique():
    if str(communicator) == 'nan':
        continue
    # filter the DataFrame for the current communicator
    df_comm = df[df['communicators'] == communicator]
    df_comm = df_comm.sort_values(by='n_params/M', ascending=False)
    # plot the model parameters against accuracy, connecting the points with a line
    ax.plot(df_comm['n_params/M'], df_comm['acc'], label=communicator, marker='o')
    
# add a legend to the plot
ax.legend()
    
# add labels and title to the plot
ax.set_xlabel('Model Parameters (in millions)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs Model Parameters')
    
# show the plot
plt.show()

#%%
model_path = 'model_checkpoints/custom_feature_wise_weighted_1682267244.pt'

communicators = FeatureWiseWeightedPass
model = VIT(d_model=128, n_heads=8, n_layers=8, patch_size=4, layer_communicators=communicators, n_heads_communicator=4, use_classifier_communicator=False)
load_model(model, model_path)

print(f'{model_path} {communicators} has, {model.get_number_of_parameters()/10**6} M parameters')

#%%
import torch
import matplotlib.pyplot as plt

for i in range(6):
    # Example 1D torch tensor
    tensor = model.communicators[5].weighter[0, :, i].detach()

    # Convert tensor to a NumPy array for plotting
    values = tensor.numpy()

    # Plotting the distribution]
    plt.figure(figsize=(10, 6)) 
    plt.hist(values, bins='auto', color='blue', alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.5)
    plt.xlabel('Vrijednost')
    plt.ylabel('Frekvencija')
    plt.title('Distribucija težina prema sloju ' + str(i))
    plt.savefig(f'output/feature_wise_frequency_layer{i}_prvimodel.png')


#%%
model_path = 'model_checkpoints/huge_weighted_1679187281.pt'
model_path = 'model_checkpoints/tiny_weighted_1679172096.pt'
model_path = 'model_checkpoints/large_dense_1679177503.pt'
model_name = 'large'
model = VIT(**size2params[model_name], layer_communicators=communicators['dense'], n_heads_communicator=4, use_classifier_communicator=False)
load_model(model, model_path)

#%%
M = model.communicators[3].lin.weight.detach().cpu().numpy()
plt.figure(figsize=(10, 6)) 
plt.title('Težinska matrica četvrtog sloja')
plt.imshow(M)
plt.savefig("output/Težinska matrica četvrtog sloja.png")

#%%
U, S, Vt = np.linalg.svd(M)
plt.figure(figsize=(10, 6)) 
plt.ylabel('Singularna vrijednost')
plt.title('Spektar težinske matrice četvrtog sloja')
plt.plot(S)
plt.savefig("output/Spektar težinske matrice četvrtog sloja.png")



#%%
####### ############# ############## ############## Create table
import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv("tmp.csv")

# Trim whitespace from column names
df.columns = df.columns.str.strip()


# Remove rows with NaN in batch_size column
df = df.dropna(subset=["batch_size"])

# Filter out unwanted columns
unwanted_columns = [
    "pretrained_model_path",
    "results_path",
    "save_delta_all",
    "save_delta_revert",
    "checkpoints_path",
    "n_heads_communicators",
    "model_name",
]
df = df.drop(columns=unwanted_columns)

# Convert all columns to string type and trim whitespace
df = df.astype(str).apply(lambda x: x.str.strip())

df["communicators"] = df["communicators"].apply(lambda s: {
    "normal":"obični",
    "dense":"gusti",
    "weighted":"težinski",
    "feature_wise_weighted": "po značajkama",
    "bottlenecked_dense":"niskog ranga",
    "nan":"nan"
}[s])

# Round values in "acc" column to 4 decimal places
df["acc"] = df["acc"].astype(float).round(4)

# Sort rows by accuracy in descending order
df = df.sort_values(by="acc", ascending=False)

# Rename column headers to Croatian
croatian_headers = {
    "acc": "Točnost",
    "batch_size": "Veličina grupe",
    # "checkpoints_path": "Putanja do kontrolnih točaka",
    "communicators": "Komunikatori",
    "d_model": "d_model",
    "label_smoothing": "Izlizavanje oznaka",
    "lr": "Stopa učenja",
    # "model_name": "Naziv modela",
    "n_heads": "Broj glava",
    # "n_heads_communicators": "Broj glava komunikatora",
    "n_layers": "Broj slojeva",
    "n_params/M": "Broj parametara/M",
    "n_total_epochs": "n_epoha",
    "n_warmup_epochs": "n_epoha_zagrijavanja",
    "patch_size": "Veličina pločica",
    # "pretrained_model_path": "Putanja do prethodno obučenog modela",
    "seed": "Sjeme",
    "use_scheduler": "Koristi raspoređivač",
    "weight_decay": "Raspad težine"
}
df = df.rename(columns=croatian_headers)


for i in range(2):
    tdf = df[i*len(df)//1: (i+1)*len(df)//1]
    # Create a table plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    table = ax.table(cellText=tdf.values, colLabels=tdf.columns, cellLoc="center", loc="center", colWidths=[0.15]*len(tdf.columns))
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.1)

    # Save the table as an image
    plt.savefig(f"output/results_table_{i}.png", bbox_inches="tight", dpi=300)

# %%
