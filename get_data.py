from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch
import transforms3d.euler as txe
import pandas as pd
from skimage import io
import os
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

class CommaCalib(Dataset):

    def __init__(self, csv_file, data_path, transforms = transforms):

        self.transforms = transforms
        self.csv_file = pd.read_csv(csv_file)
        self.data_path = data_path
        self.resize_image = transforms.Resize((582,437))
        self.center_crop = transforms.CenterCrop(280)
        #self.mean = [0.485, 0.456, 0.406]
        #self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.data_path,self.csv_file.iloc[idx,0])
        image = io.imread(img_name)
        image = torch.from_numpy(image.transpose((2,0,1)))

        # h*w*x -> c*h*w , numpy image to torch image
        image = image / 255

        if self.transforms is not None:
            image = self.resize_image(image)
            image = self.center_crop(image)
            #image[0] = (image[0] - self.mean[0])/ self.std[0]
            #image[1] = (image[1] - self.mean[1]) / self.std[1]
            #image[2] = (image[2] - self.mean[2]) / self.std[2]


        label = self.csv_file.iloc[idx,1:]
        label = label.values.tolist()
        # convert label in euler to quaternions
        # roll = 0 always, appending roll before converting to quaternions
        label.insert(0,0) # roll pitch yaw
        label = txe.euler2quat(label[0],label[1],label[2])
        label = torch.Tensor(label)

        return image,label

    def __len__(self):
        return len(self.csv_file)

#calib_dataset = CommaCalib(csv_file = 'labels.csv',data_path='frames_labeled', transforms = transforms)
#print(calib_dataset[0][1])
#plt.imshow(calib_dataset[0][0].permute(1, 2, 0))
#plt.show()


# show an image