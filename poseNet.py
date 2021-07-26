# poseNet architecture
import torch
import torch.nn.functional as F
from transforms3d.euler import quat2euler

class poseNet(torch.nn.Module):

    def __init__(self, feature_extractor, num_features = 128, dropout = 0.5):

        super(poseNet, self).__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        fc_in_features = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = torch.nn.Linear(fc_in_features,num_features)
        self.dropout = dropout

        # rotation in quaternions
        self.fc_quat = torch.nn.Linear(num_features,4)

        # initializing weights and biases
        init_modules = [self.feature_extractor, self.fc_quat]
        for m in init_modules :
            if isinstance(m,torch.nn.Conv2d) or isinstance(m,torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight.data)
                torch.nn.init.constant_(m.bias.data,0)



    def extract_features(self,x):

        x_features = self.feature_extractor(x)
        x_features = F.relu(x_features)
        if self.dropout > 0 :
            x_features = F.dropout(x_features, p = self.dropout, training = self.training)
        return x_features

    def forward(self,x):

        x_features = self.extract_features(x)
        x_rotations = self.fc_quat(x_features)
        return x_rotations # returns quaternions


class poseNetCriterion(torch.nn.Module):

    def __init__(self, beta = 512.0, learn_beta = True, sq = 0):
        super(poseNetCriterion,self).__init__()
        self.loss_fn = torch.nn.MSELoss()
        self.learn_beta = learn_beta

        if not learn_beta:
            self.beta = beta
        else :
            self.beta = 1.0

        # neglecting sq for now
        #self.sq = torch.nn.Parameter(torch.Tensor([sq]), requires_grad= learn_beta)

    def forward(self,x,y):

        """
        args :
        x : predicted N*4
        y : gt N*4
        """

        # convert x and y to euler's angles
        x_euler = []
        y_euler = []

        x_np = x.numpy()
        y_np = y.numpy()

        # convert tensors to numpy array
        for c in x_np:
            x_euler.append(quat2euler())



        loss = self.loss_fn(x,y)
        return loss

