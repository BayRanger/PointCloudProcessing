# %%
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
#import pretty_errors
import numpy as np

# %% 
#继承 官网 的nn.Module
class PointNet(nn.Module):
  def __init__(self):
    super(PointNet, self).__init__()
    #Applies a 1D convolution over an input signal composed of several input planes.
    # 3 input pcl channel, 64 output channels, 1x1 square convolution
    self.conv1 = nn.Conv1d(3, 64, 1)
    # 64 input pcl channel, 128 output channels, 1x1 square convolution
    self.conv2 = nn.Conv1d(64, 128, 1)
    # 128 input pcl channel, 1024 output channels, 1x1 square convolution
    self.conv3 = nn.Conv1d(128, 1024, 1)
 
    self.fc1 = nn.Linear(1024, 512)
    self.fc2 = nn.Linear(512, 256)
    self.fc3 = nn.Linear(256, 40)

    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(128)
    self.bn3 = nn.BatchNorm1d(1024)
    self.bn4 = nn.BatchNorm1d(512)
    self.bn5 = nn.BatchNorm1d(256)

    self.relu = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.3)

  def forward(self, x):
    # TODO: use functions in __init__ to build network
    """
        In the forward function we accept a Tensor of input data and we must return
        a Tensor of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Tensors.
    """
    batchsize = x.size()[0]
    x = F.relu(self.bn1(self.conv1(x)))
    x = F.relu(self.bn2(self.conv2(x)))
    x = F.relu(self.bn3(self.conv3(x))) 
    #x = F.relu((self.conv1(x)))
    #x = F.relu((self.conv2(x)))
    #x = F.relu((self.conv3(x)))
    # 至此唯独上升到1024，开始寻找全局特征 ， 3,1024,10000
    x = torch.max(x, 2, keepdim=True)[0] #3，1024,1
    x = x.view(-1, 1024) ##3，1024
    x = F.relu(self.bn4(self.fc1(x)))
    x = F.relu(self.bn5(self.fc2(x)))
    #x = F.relu((self.fc1(x)))
    #x = F.relu(self.dropout(self.fc2(x)))
    x = self.fc3(x)

    #x = F.softmax(x, dim=1)
     
    return x


if __name__ == "__main__":
  net = PointNet()
  sim_data = Variable(torch.rand(3, 3, 10000))
  out = net(sim_data)
  print('gfn', out.size(),out)
  

 