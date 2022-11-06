import torch.nn as nn
import torch.nn.functional as F

class NetBodyParts(nn.Module):
    def __init__(self,resolution=1024):
        super(NetBodyParts,self).__init__()
        #sees 1024x1024x3 image tensor 
        self.conv1 = nn.Conv2d(3,16,7,padding=3)
        #sees 512x512x16 tensor      /4
        self.conv2 = nn.Conv2d(16,32,7,padding=3)
        #sees 256x256x32 tensor      /16
        self.conv3 = nn.Conv2d(32,64,3,padding=1)
        #sees 128x128X128 tensor    /64
#        self.conv4 = nn.Conv2d(64,128,3,padding=1)
        #                           /256
        self.pool  = nn.MaxPool2d(4,4)
        self.resolution=resolution
#       only 3 convolutional layers
        self.sizefc1=int(64*self.resolution*self.resolution/(64*64))
 
        self.fc1 = nn.Linear(self.sizefc1,int(self.sizefc1/2))
        self.fc2 = nn.Linear(int(self.sizefc1/2),6)
        self.dropout = nn.Dropout(0.5)

    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        x=self.pool(F.relu(self.conv3(x)))
#        x=self.pool(F.softmax(self.conv4(x),dim=3))
#        x=self.pool(F.relu(self.conv4(x)))        
        #flatten image input
        x=x.view(-1,self.sizefc1)
#        x=x.view(-1,64*128*128)
        #add dropout
        x=self.dropout(x)
        #first fc layer with relu
        x=F.relu(self.fc1(x))
        x=self.dropout(x)
        #second fc layer with softmax
        x=F.softmax(self.fc2(x),dim=1)
#        x=self.fc2(x) 
        return x
class NetMelanoma(nn.Module):
    def __init__(self,modelA,resolution):
        super(NetMelanoma,self).__init__()
        self.modelBP =modelA
        self.resolution=resolution
        self.classifier = nn.Linear(6,2)
    def forward(self,x):
        x=self.modelBP(x)
        x=F.softmax(self.classifier(x),dim=1)
        return x
