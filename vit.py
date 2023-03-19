#%%
import torch
import torch as th
from torch import nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch.nn.functional as F
from communicators import LastPass

class VIT(nn.Module):
    def __init__(self, 
            d_model=192, 
            n_layers=12, 
            n_heads=3, 
            mlp_ratio=4,
            patch_size=4,
            img_size=(32, 32),
            n_classes=10,
            layer_communicators=LastPass,
            n_heads_communicator=4
        ):

        '''
        Args:
            layer_communicators: list of classes or 
        '''
        super().__init__()
        if type(layer_communicators) != list:
            layer_communicators = [layer_communicators]*n_layers
        else:
            assert n_layers is None
            n_layers = len(layer_communicators)

        self.layers = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead = n_heads,
                    dim_feedforward=mlp_ratio*d_model,
                    dropout=0,
                    activation=nn.GELU(),
                    batch_first=True,
                    norm_first= True
                ) for _ in range(n_layers)
            ]
        )
        self.communicators = nn.ModuleList([
            comm(d_model, i+1, n_heads=n_heads_communicator) for i, comm in enumerate(layer_communicators)
        ])

        n_tokens_in_image= img_size[0] * img_size[1] // patch_size**2
        self.positional_encoding_encoder = torch.nn.parameter.Parameter(
            torch.randn(size=(1, n_tokens_in_image, d_model))
        )
        self.cls_token_embedding = torch.nn.parameter.Parameter(
            torch.randn(size=(1, 1, d_model))
        )
        self.classifier = torch.nn.Linear(d_model, n_classes)
        numel_of_patch=3 * patch_size**2
        self.patch_projector_in = nn.Linear(numel_of_patch, d_model)
        self.d_model = d_model
        self.patch_size = patch_size

    def forward(self, imgs):
        patches = imgs_to_patches(imgs, patch_size=self.patch_size)
        source = self.patches_to_tokens(patches)
        source = self.add_cls_token(source)
        outputs = [source]
        for transformer_layer, comm_layer in zip(self.layers, self.communicators):
            x = comm_layer(outputs)
            x = transformer_layer(x)
            outputs.append(x)
        cls_token = x[:, 0]
        classes = self.classifier(cls_token)
        return torch.softmax(classes, dim=-1)

    def patches_to_tokens(self, patches):
        x = self.patch_projector_in(patches)
        return x + self.positional_encoding_encoder

    def add_cls_token(self, x):
        cls_token = self.cls_token_embedding.expand(len(x), -1, -1)
        return torch.cat((cls_token, x), dim=1)

    def get_output_dim(self):
        return self.d_model
    
    def get_number_of_parameters(self):
        nump = 0
        for p in self.parameters():
            nump += p.numel()
        return nump
    



def _get_conv_weight(h, w):
    zeros = []
    for i in range(3*h*w):
        z = th.zeros((3*h*w,))
        z[i] = 1
        zeros.append(z.reshape(1, 3, h, w))
    return th.cat(zeros)

def imgs_to_patches(imgs, patch_size):
    '''
    returns b, -1, 3 x patch_size x patch_size
    '''
    weight = _get_conv_weight(patch_size, patch_size).to(imgs.device)
    convolved = F.conv2d(imgs, weight, None, patch_size)
    return convolved.view(imgs.shape[0], 3*patch_size*patch_size, -1).permute(0, 2, 1)

def visualize_patches(patches, n_patches_in_row, patch_size):
    assert len(patches.shape) == 2, 'give only one patch'
    patches = patches.view(-1, 3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1)
    patches_in_column = patches.shape[0] // n_patches_in_row

    for i, patch in enumerate(patches):
        plt.subplot(n_patches_in_row, patches_in_column, i+1)
        plt.imshow(patch.numpy())

    return patches

def patches_to_imgs_chw_slow(patches, n_patches_in_row, patch_size):
    # patches.shape = b, n, 3*patch_size**2
    b, n, _ = patches.shape
    patches = patches.view(b, n//n_patches_in_row, n_patches_in_row, 3, patch_size, patch_size)
    imgs = []
    for i in range(patches.shape[1]):
        row = []
        for j in range(patches.shape[2]):
            row.append( patches[:, i, j])
        row = torch.cat(row, dim=-1)
        imgs.append(row)
    imgs = torch.cat(imgs, dim=-2)
    return imgs

def patches_to_imgs_hwc_slow(patches, n_patches_in_row, patch_size):
    return patches_to_imgs_chw_slow(patches, n_patches_in_row, patch_size).permute(0, 2, 3, 1)


# %% 
if __name__ == '__main__':
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10('data', True, transform=T.ToTensor(), download=True)
    patch_size = 4
    imgs = dataset[100][0][None]
    _, _, h, w = imgs.shape
    n_patches_in_row = w // patch_size
    patches = imgs_to_patches(imgs, patch_size)
    print(patches.shape)

    
    # visualize_patches(patches[0], n_patches_in_row, patch_size)
    imgs2 = patches_to_imgs_hwc_slow(patches, n_patches_in_row, patch_size)

    # run cell with MAEModel in it for this line to work
    model = VIT()
    model(imgs)