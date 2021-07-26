from torchvision import transforms, models
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from get_data import CommaCalib
from poseNet import poseNet, poseNetCriterion
from utils import AverageMeter, train, save_checkpoint
import time
import numpy as np
from utils import get_mse_error

# dataset and dataloader
calib_dataset = CommaCalib(csv_file = 'labels.csv',data_path='frames_labeled', transforms = transforms)
train_data = DataLoader(calib_dataset,batch_size = 16, shuffle = True)

# primary device
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("gpu detected")
else :
    device = torch.device('cpu')
    print("gpu undetected")

# feature extractor
feature_extractor = models.resnet34(pretrained=True)

# num of features for last layer before pose regressor
num_features = 2048
# model
model = poseNet(feature_extractor, num_features=num_features,dropout=0.5)
model = model.to(device)


criterion = poseNetCriterion(learn_beta = True)
criterion = criterion.to(device)

# adding parameters for optimization
param_list = [{'params' : model.parameters()}]
if criterion.learn_beta:
    param_list.append({'params':criterion.parameters()}) # learning sq from loss function

# create optimizer
optimizer = optim.Adam(params = param_list, lr=1e-5, weight_decay = 0.0005)

epochs = 200

# write code here for training
loss_stats = []
for curr_epoch in range(0,epochs):
    train(train_data, model, criterion, optimizer, curr_epoch, epochs, log_freq = 1, print_sum=True, device = device)


    save_checkpoint(model, optimizer, criterion, experiment_name='quats', epoch = curr_epoch) # save model for every epoch

# save checkpoint only once at the end of training
#save_checkpoint(model, optimizer, criterion, 'quats',epochs)





































