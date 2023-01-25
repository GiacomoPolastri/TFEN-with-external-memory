"""

Last training in whitch every 50 epochs I change the lr parameter 

"""

import torch
import torch.nn as nn
import torch.optim as optim
import warnings

from Models.mobilenetv2 import mobilenetv2
from Models.utils import get_number_model, change_number_model, save_model, save_plots
from Train.utils import training, valid
from utils import get_device, model_to_device
from Data.dataset import LoadDataset
from Data.classes import classes
from runtime_args import args

n_model = get_number_model()
root = './Train/' + str(n_model) + '.pth'

# eliminate warnings
warnings.filterwarnings("ignore", category=UserWarning) 

train_loss, val_loss = [], []
train_acc, val_acc = [], []

lr = args.lr

train_loader = LoadDataset(args).trainloader
val_loader = LoadDataset(args).validloader

model = mobilenetv2()
device = get_device()
model = model_to_device(model,device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), 
                      lr, 
                      momentum = args.momentum,
                      weight_decay = args.weight_decay)

if args.resume == True:
    print ('=> loading checkpoint')
    checkpoint = torch.load(root, map_location=device)                          # device
    args.startepoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])                
    print('=> loading checkpoint: {} /n (epoch: {})/n'.format(root, checkpoint['epoch']))

for epoch in range(args.startepoch, args.epochs):
    if ((epoch) % 50 == 0): 
        lr = lr/10
        optimizer = optim.SGD(model.parameters(), 
                                           lr = lr, 
                                           weight_decay = args.weight_decay, 
                                           momentum = args.momentum)
    train_epoch_loss, train_epoch_acc = training(device, 
                                              model, 
                                              epoch, 
                                              train_loader, 
                                              optimizer,
                                              criterion)
    val_epoch_loss, val_epoch_acc = valid(device,
                                           model,
                                           epoch,
                                           val_loader,
                                           optimizer,
                                           criterion,
                                           n_model)
    #scheduler.step()
    train_loss.append(train_epoch_loss)
    val_loss.append(val_epoch_loss)
    train_acc.append(train_epoch_acc)
    val_acc.append(val_epoch_acc)
    if (epoch % args.mem == 0) : save_plots(train_acc, val_acc, train_loss, val_loss, n_model)

a = change_number_model(n_model)
final_n = get_number_model()
aa = save_model(args.epochs, model, optimizer, criterion, val_acc, final_n)
a = change_number_model(final_n)