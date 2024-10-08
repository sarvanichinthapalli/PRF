
import torch.nn as nn
import torch.nn.functional as F
import torch as th
from resnet_network_prune import Network
from resnet_pruningmethod import PruningMethod
norm_mean, norm_var = 0.0, 1.0


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ResBasicBlock(nn.Module,Network,PruningMethod):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super(ResBasicBlock, self).__init__()
        self.inplanes = inplanes
        self.planes = planes
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.stride = stride
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
        #    print('called')
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes-inplanes-(planes//4)), "constant", 0))

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        ##print('@@@@@@@@@@@',self.planes,self.inplanes)
        
        
        ##print('x=',x.shape,'out=',out.shape,'shortcut=',self.shortcut(x).shape)
        #if(x.shape[1]!=out.shape[1]):
        #  planes=out.shape[1]
        #  inplanes=x.shape[1]
          ##print(inplanes,'---------',planes)
        #  self.shortcut = LambdaLayer(
        #        lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes // 4, planes-inplanes-(planes//4)), "constant", 0))

        out += self.shortcut(x)
        out = self.relu2(out)

        return out


class ResNet(nn.Module,Network,PruningMethod):
    def __init__(self, block, num_layers, covcfg,num_classes=10):
        super(ResNet, self).__init__()
        assert (num_layers - 2) % 6 == 0, 'depth should be 6n+2'
        n = (num_layers - 2) // 6
        self.covcfg = covcfg
        self.num_layers = num_layers

        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)


        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(1,block, 16, blocks=n, stride=1)
        self.layer2 = self._make_layer(2,block, 32, blocks=n, stride=2)
        self.layer3 = self._make_layer(3,block, 64, blocks=n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        if num_layers == 110:
            self.linear = nn.Linear(64 * block.expansion, num_classes)
        else:
            self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.initialize()
        self.layer_name_num={}
        self.pruned_filters={}
        self.remaining_filters={}

        self.remaining_filters_each_epoch=[]

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,a, block, planes, blocks, stride):
        layers = [] 

        layers.append(block(self.inplanes, planes, stride))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        #print(x.shape,'enter this logicaly')
        x = self.avgpool(x)
        #print(x.shape,'changed')
        x = x.view(x.size(0), -1)

        if self.num_layers == 110:
            x = self.linear(x)
        else:
            x = self.fc(x)

        return x


def resnet_56():
    cov_cfg = [(3 * i + 2) for i in range(9 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 56, cov_cfg)
def resnet_110(n_iterations):
    cov_cfg = [(3 * i + 2) for i in range(18 * 3 * 2 + 1)]
    return ResNet(ResBasicBlock, 110, cov_cfg)
