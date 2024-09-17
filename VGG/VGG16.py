import torch as th
import torch.nn as nn
from vgg_network_prune import Network
from  vgg_pruningmethod import PruningMethod
#from torchsummary import summary
import math
from collections import OrderedDict

class VGG16(nn.Module,Network,PruningMethod):
    def __init__(self, n_c, a_type):
        super(VGG16, self).__init__()

        self.a_type = a_type

        if a_type == 'relu':
            self.activation = nn.ReLU()
        elif a_type == 'tanh':
            self.activation = nn.Tanh()
        elif a_type == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif a_type == 'leaky_relu':
            self.activation = nn.LeakyReLU()
        else:
            print('Not implemented')
            raise

        # First encoder
        self.layer1 = nn.Sequential(
                *([nn.Conv2d(3, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   self.activation]))
        self.layer2 = nn.Sequential(
                *([nn.Conv2d(64, 64, kernel_size=3, padding=1),
                   nn.BatchNorm2d(64),
                   self.activation]))
        # Second encoder
        self.layer3 = nn.Sequential(
                *([nn.Conv2d(64, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   self.activation]))
        self.layer4 = nn.Sequential(
                *([nn.Conv2d(128, 128, kernel_size=3, padding=1),
                   nn.BatchNorm2d(128),
                   self.activation]))

        # Third encoder
        self.layer5 = nn.Sequential(
                *([nn.Conv2d(128, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))
        self.layer6 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))
        self.layer7 = nn.Sequential(
                *([nn.Conv2d(256, 256, kernel_size=3, padding=1),
                   nn.BatchNorm2d(256),
                   self.activation]))

        # Fourth encoder
        self.layer8 = nn.Sequential(
                *([nn.Conv2d(256, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer9 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer10 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))

        # Fifth encoder
        self.layer11 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer12 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))
        self.layer13 = nn.Sequential(
                *([nn.Conv2d(512, 512, kernel_size=3, padding=1),
                   nn.BatchNorm2d(512),
                   self.activation]))

        # Classifier
        self.fc1 = nn.Sequential(*([
                nn.Linear(512, 512),
                nn.BatchNorm1d(512),
                self.activation]))

        self.classifier = nn.Sequential(
                *([nn.Linear(512, n_c),]))

        for m in self.modules():
            self.weight_init(m)

        self.pool = nn.MaxPool2d(2, 2)
        self.softmax = nn.Softmax(dim=1)


        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}

        self.remaining_filters_each_epoch=[]

       

    def forward(self, x):

        # Encoder 2
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        pool1 = self.pool(layer2)

        # Encoder 2
        layer3 = self.layer3(pool1)
        layer4 = self.layer4(layer3)
        pool2 = self.pool(layer4)

        # Encoder 3
        layer5 = self.layer5(pool2)
        layer6 = self.layer6(layer5)
        layer7 = self.layer7(layer6)
        pool3 = self.pool(layer7)

        # Encoder 4
        layer8 = self.layer8(pool3)
        layer9 = self.layer9(layer8)
        layer10 = self.layer10(layer9)
        pool4 = self.pool(layer10)

        # Encoder 5
        layer11 = self.layer11(pool4)
        layer12 = self.layer12(layer11)
        
        layer13 = self.layer13(layer12)
        #pool5 = self.pool(layer13)
        #print(layer13.shape)

        #avgpool 
        avg_x=nn.AvgPool2d(2)(layer13)
        #print(avg_x.shape)

        # Classifier
        fc1 = self.fc1(avg_x.view(avg_x.size(0), -1))
        #print(fc1.shape)
        
        classifier = self.classifier(fc1)
        return classifier


'''device = th.device("cuda" if th.cuda.is_available() else "cpu") # PyTorch v0.4.0
model_tupry = VGG16(10,'relu',100).to(device)

a=model_tupry(th.randn(1, 3, 32, 32).cuda())
for i in range(15):
	print('layer ',str(i),' ',a[i].shape)
#for name, layer_module in model_tupry.named_modules():
#  if(isinstance(layer_module, th.nn.Conv2d)):
#    a.append(layer_module.shape)
#---------intialize layer numbers and names dictionary------
model_tupry.intialize_layer_name_num(model_tupry)

#---------intialize pruned and remaning filters-----------
model_tupry.filters_in_each_layer(model_tupry)
 
#--------calculate remaining filters in each epoch--------
model_tupry.remaining_filters_per_epoch(model=model_tupry,initial=True)
model_tupry.calculate_total_flops(model_tupry)
import sys
sys.exit()
for layer_name, layer_module in model_tupry.named_modules():
           if(isinstance(layer_module, th.nn.Conv2d) or  isinstance(layer_module,th.nn.Linear)):
              print(layer_name,'-------',layer_module,'---------',layer_module.weight.size()[0])'''

