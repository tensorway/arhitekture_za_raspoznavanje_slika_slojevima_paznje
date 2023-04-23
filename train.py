#%%
import os
import time
import torch
import torchvision
import torch as th
import torch.nn as nn
from tqdm import tqdm
from pathlib import Path
from schedulers import CosineSchedulerWithWarmupStart
from vit import VIT
from transforms import vit_val_transform as val_transform
from transforms import vit_train_transform as train_transform
from utils import load_model, save_model, seed_everything
import argparse
from torch.utils.tensorboard import SummaryWriter
from communicators import LastPass, WeightedPass, DensePass, AttentionPass



# Argument parsing #################################################################################
def add_standard_arguments(parser):

    # model parameters
    parser.add_argument("-n", "--model_name", type=str, default='tiny', help="Model size parameters, default 'custom'.")
    parser.add_argument("-c", "--communicators", type=str, default='normal', choices=['normal', 'weighted', 'dense', 'attention'] ,help="Model size parameters, default 'custom'.")
    parser.add_argument("-d", "--d_model", type=int, default=192, help="Model dimension, default 192.")
    parser.add_argument("-nl", "--n_layers", type=int, default=12, help="Number of layers in the transformer encoder. Default: 12.")
    parser.add_argument("-nh", "--n_heads", type=int, default=12, help="Number of heads in the transformer encoder. Default: 12.")
    parser.add_argument("-nhc", "--n_heads_communicators", type=int, default=4, help="Number of heads in the attention communicator. Default: 4.")
    parser.add_argument("-ps", "--patch_size", type=int, default=4, help="pixel width and height of a patch, total number of pixels in a patch is patch_size**2")

    # optimization parameters
    parser.add_argument("-lr", "--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=128, help="train and valid batch size")
    parser.add_argument("-e", "--n_total_epochs", type=int, default=50, help="total number of epochs")
    parser.add_argument("-w", "--n_warmup_epochs", type=int, default=5, help="number of warmup epochs")
    parser.add_argument("-wd", "--weight_decay", type=float, default=5e-5, help="weight decay parameter")
    parser.add_argument('-us', '--use_scheduler', action='store_true')
    parser.add_argument('-ls', '--label_smoothing', type=float, default=0.1)

    # saving delta times
    parser.add_argument("-sda", "--save_delta_all", type=int, default=15000, help="in seconds, the model that is stored and overwritten to save space")
    parser.add_argument("-sdr", "--save_delta_revert", type=int, default=30000, help="in seconds, checkpoint models saved rarely to save storage")
    
    # paths so save/load
    parser.add_argument("-chp", "--checkpoints_path", type=str, default='model_checkpoints/', help="folder where to save the checkpoints")
    parser.add_argument("-ptr", "--pretrained_model_path", default=None, type=str, help="pretrained model path from which to train")
    parser.add_argument("-r", "--results_path", type=str, default='results.csv', help="file where to save the results")

    # seed
    parser.add_argument("-s", "--seed", type=int, default=42, help="RNG seed. Default: 42.")



def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# Main start #################################################################################

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Finetuning script')
    add_standard_arguments(parser)
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(args)
    print("using", device)
    seed_everything(args.seed)

    size2params = {
        'custom':{
            'd_model' : args.d_model,
            'n_layers' : args.n_layers,
            'n_heads' : args.n_heads,
            'patch_size' : args.patch_size,
        },
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
    args.d_model = size2params[args.model_name]['d_model']
    args.n_layers = size2params[args.model_name]['n_layers']
    args.n_heads = size2params[args.model_name]['n_heads']
    args.patch_size = size2params[args.model_name]['patch_size']

    communicators = {
        'normal':LastPass,
        'weighted':WeightedPass,
        'dense':DensePass,
        'attention':AttentionPass
    }[args.communicators]

    import pprint
    pprint.pprint(size2params[args.model_name])


    # Load everything #################################################################################
    model_pretrained_str = 'pretrain_'+args.__str__()
    model_str = f'{args.model_name}_{args.communicators}_{int(time.time())}'
    model_path = Path(args.checkpoints_path)/('model_'+model_str+'.pt')
    optimizer_path =  Path(args.checkpoints_path)/('optimizer_'+model_str+'.pt')

    model = VIT(**size2params[args.model_name], layer_communicators=communicators, n_heads_communicator=args.n_heads_communicators)
    opt = th.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    model.to(device)
    if args.pretrained_model_path is not None:
        load_model(model, args.pretrained_model_path)

    print('*'*30)
    print(f'{args.model_name} {args.communicators} has, {model.get_number_of_parameters()/10**6} M parameters')
    print('*'*30)


    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=val_transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=6)
    valid_dataloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=6)


    writer = SummaryWriter('tensorboard/'+model_str)
    if args.use_scheduler:
        scheduler = CosineSchedulerWithWarmupStart(opt, n_warmup_epochs=args.n_warmup_epochs, n_total_epochs=args.n_total_epochs)
        print('using scheduler')
    else:
        scheduler = torch.optim.lr_scheduler.ConstantLR(opt, factor=1, total_iters=1)


    step = 0
    t_last_save_revert = time.time()
    t_last_save_all = time.time()
    scaler = torch.cuda.amp.GradScaler()

    def whole_dataset_eval(ep):
        model.eval()
        cumacc = 0
        cumloss = 0
        for ibatch, (imgs, labels) in enumerate(valid_dataloader):
            with th.no_grad():
                with torch.cuda.amp.autocast():
                    imgs, labels = imgs.to(device), labels.to(device) 
                    outputs = model(imgs)
                    cumloss += criterion(outputs, labels)
                    cumacc += (outputs.argmax(dim=-1) == labels).float().mean().item()
        acc, loss = cumacc/(ibatch+1), cumloss/(ibatch+1)
        # print('valid', loss.item(), acc, ibatch)
        writer.add_scalar("loss/valid", loss, ep)
        writer.add_scalar("acc/valid", acc, ep)
        model.train()
        return acc


    # Main loop #################################################################################
    for ep in tqdm(range(args.n_total_epochs)):
        model.eval()
        scheduler.step()
        for ibatch, (imgs, labels) in enumerate(train_dataloader):

            opt.zero_grad()
            with torch.cuda.amp.autocast():
                imgs, labels = imgs.to(device), labels.to(device) 
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                acc = (outputs.argmax(dim=-1) == labels).float().mean().item()

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()


            if ibatch%30 == 0:
                writer.add_scalar("loss/train", loss, step)
                writer.add_scalar("acc/train", acc, step)
                writer.add_scalar("lr", get_lr(opt), step)
                # print(ep, step, loss.item(), acc)

            if ibatch % 300 == 0:
                whole_dataset_eval(ep)

            if time.time() - t_last_save_all > args.save_delta_all:
                save_model(model, os.path.join(args.checkpoints_path, 'model_slow.pt'))
                save_model(opt, os.path.join(args.checkpoints_path,'opt_slow.pt'))
                t_last_save_all = time.time()

            if time.time() - t_last_save_revert > args.save_delta_revert:
                save_model(model, os.path.join(args.checkpoints_path, 'model' + str(step) + '.pt'))
                save_model(opt, os.path.join(args.checkpoints_path,'opt' + str(step) + '.pt'))
                t_last_save_revert = time.time()
            

            step += 1

    save_model(model, os.path.join(args.checkpoints_path, model_str + '.pt'))
    acc = whole_dataset_eval(ep)

    # Write results #################################################################################
    with open(args.results_path, 'a') as fout:
        dic = vars(args)
        dic['acc'] = acc
        dic['n_params/M'] = model.get_number_of_parameters()/10**6
        fout.write(','.join([str(dic[k]) for k in sorted(dic.keys())]) + '\n')
        # with open('tmp.txt') as fin:
        #     lines = fin.readlines()[0].strip()
        #     elements = lines.split(',')
        #     while len(elements)>0:
        #         minidic = {k:'nan' for k in dic.keys()}
        #         curr, elements = elements[:len(dic)], elements[len(dic)+1:]
        #         for el in curr:
        #             name, val = el.split('=')
        #             minidic[name] = val

        #         fout.write(','.join(
        #             [
        #                 minidic[k] for k in sorted(dic.keys())
        #             ]
        #         )+'\n')
        #         l = {}



        # for k in sorted(dic.keys()):
            # fout.write(','+str(k)+'='+str(dic[k]))
            # fout.write(','+str(k)+'='+str(dic[k]))