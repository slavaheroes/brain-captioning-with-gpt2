import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class BrainConv3DNet(nn.Module):
    def __init__(self, in_chans, planes=32, num_classes=768):
        super(BrainConv3DNet, self).__init__()
        self.in_planes = planes

        self.conv1 = nn.Conv3d(in_chans, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        
        # Assuming 'BasicBlock' is repeated twice for a small network
        self.layer1 = self._make_layer(BasicBlock, planes*2, 2, stride=2)
        self.layer2 = self._make_layer(BasicBlock, planes*4, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, planes*8, 2, stride=4)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(planes*8, num_classes)
        
        print("Num classes: ", num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    # Instantiate the network
    x = torch.randn(2, 1, 81, 104, 83)
    net = BrainConv3DNet(in_chans=1, planes=32,
        num_classes=768)

    # Print 
    print(net(x).shape)
    
    print(net)
