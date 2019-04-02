"""
how to run: python src/torch_net.py --pickle-file <PICKLE DATA> --epochs 5 --lr 1e-4 --weight-decay  1e-6 --momentum 0.9 --nesterov --batch-size 64 --saveprefix /home/franck/TRASH/cnn-music-structure/src/checkpoints --tensorboard-saveprefix /home/franck/TRASH/cnn-music-structure/src/tensorboard
"""


# Usual imports
import time
import math
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import pickle
from glob import glob

import sys
sys.path.append('./util')

# My modules
import generate_data as d
from dataset import DictDataset

def rel_error(x, y):
    """ Returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8. np.abs(x) + np.abs(y))))

import generate_data # My data function
import util.evaluation as ev

#Torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torch.backends import cudnn

#tensorboard
from tensorboardX import SummaryWriter





class SegmentationNet(nn.Module):
    def __init__(self):
        super(SegmentationNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        #The paper actually has 128 neurons in the 1st fully connected layer
        self.fc1 = nn.Linear(16 * 32 * 32, 256)
        
        self.fc2 = nn.Linear(256, 1)
        self.dropout = nn.Dropout(p=0.25)
        self.batch_norm = nn.BatchNorm2d(16)
        
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        #print(x.shape)
        x = self.batch_norm(F.relu(self.conv1(x)))
        #print(x.shape)
        x = self.batch_norm(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.dropout(self.pool(x))
        #print(x.shape)
        x = self.batch_norm(F.relu(self.conv3(x)))
        #print(x.shape)
        x = self.batch_norm(F.relu(self.conv2(x)))
        #print(x.shape)
        x = self.dropout(self.pool(x))
        #print(x.shape)
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.fc2(x)
        return x


# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
cudnn.benchmark = True

def main():

    with open(cmd_args.pickle_file, 'rb') as f:
        data_dict = pickle.load(f)

    run_training(data_dict, cmd_args.batch_size, cmd_args.lr, cmd_args.weight_decay, cmd_args.momentum, cmd_args.nesterov, cmd_args.epochs)

def run_training(data_dict, batch_size, lr, weight_decay, momentum, nesterov, epochs):

    runs = glob(cmd_args.saveprefix+'/*')
    it = len(runs) + 1
    writer = SummaryWriter(os.path.join(cmd_args.tensorboard_saveprefix, str(it)))
    writer.add_text('Metadata', 'Run {} metadata :\n{}'.format(it, cmd_args,))
    # Parameters
    params = {'batch_size': batch_size,
              'shuffle': True,
              'num_workers': 0}

    # Generators
    training_set = DictDataset(data_dict, split='train')
    training_generator = data.DataLoader(training_set, **params)

    validation_set = DictDataset(data_dict, split='val')
    validation_generator = data.DataLoader(validation_set, **params)

    #create model
    model = SegmentationNet()

    model = torch.nn.DataParallel(model)
    if use_cuda:
        model.cuda()

    # Loss and Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)  
    #mean squarred logarithmic error
    def msle(output, target):
        loss = torch.mean((torch.log(output + 1) - torch.log(target+1))**2)
        return loss

    best_val_loss = np.inf
    for ind, epoch in enumerate(range(epochs)):
        val_loss = train(training_generator, validation_generator, model, msle, optimizer, epoch, writer)

        
        is_best = val_loss > best_val_loss
        if is_best:
            best_val_loss = val_loss
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, is_best, os.path.join(cmd_args.saveprefix, str(it), f'ep_{epoch+1}_map_{best_val_loss:.3}'))
    writer.close()

def train(train_loader, val_loader, model, criterion, optimizer, epoch, writer):

    # switch to train mode
    model.train()

    # Training
    running_loss = 0.0
    loader_len = len(train_loader)
    for i, data in enumerate(train_loader, 0):
        local_batch, local_labels = data
        
        # Transfer to GPU
        #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
        local_batch, local_labels = local_batch.cuda(), local_labels.cuda()

        # Model computations
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(local_batch)
        loss = criterion(outputs, local_labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % cmd_args.print_offset == cmd_args.print_offset -1:    # print every 10 mini-batches
            print('[%d, %5d] loss: %.6f' %
                  (epoch + 1, i + 1, running_loss /cmd_args.print_offset ))
            writer.add_scalar('data/losses/train_loss', running_loss/cmd_args.print_offset, i + 1 + epoch*loader_len)
            running_loss = 0
    

    # Validation
    avg_val_loss = eval(val_loader, model, criterion, epoch)
    writer.add_scalar('data/losses/train_loss', running_loss/cmd_args.print_offset, (epoch+1)*loader_len)
    

    return avg_val_loss

def eval(val_loader, model, criterion, epoch=0):

    # switch to eval mode
    model.eval()

    with torch.set_grad_enabled(False):
        val_loss = 0.0
        for cpt, data in enumerate(val_loader, 0):
            local_batch, local_labels = data

            # Transfer to GPU
            #local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
            
            # Model computations
            # forward + backward + optimize
            outputs = model(local_batch)
            loss = criterion(outputs, local_labels)
            val_loss += loss.item()

    #cpt here is the last cpt in the loop, len(validator_generator) -1
    print(f'Epoch {epoch + 1} validation loss: {val_loss / (cpt+1)}')

    return val_loss / (cpt+1)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '_latest.pth.tar')
    if is_best:
        shutil.copyfile(filename + '_latest.pth.tar', filename + '_best.pth.tar')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CNN Music Structure training')
    parser.add_argument('--pickle-file', type=str,
                        help='file from which to load the dictionnary containing the training data info')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-6,
                        help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum')
    parser.add_argument('--nesterov', action='store_true',
                        help='Nesterov')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size to use for training')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of epochs for training')
    parser.add_argument('--saveprefix', type=str,
                        help='folder where to save the checkpoint files')
    parser.add_argument('--tensorboard-saveprefix', type=str,
                        help='folder where to save the tensorboardX  files')
    parser.add_argument('--print-offset', type=int, default=10,
                        help='how often to print in minibatches')
    cmd_args = parser.parse_args()
    main()
