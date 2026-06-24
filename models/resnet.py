import torch
import torch.nn as nn 


class CIFARResNetStem(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
    
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride = stride, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, stride = 1, padding = 1, kernel_size=3, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        nn.init.constant_(self.bn2.weight, 0)

        
        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential( nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)

        return out

class CIFARResNet(nn.Module):
    def __init__(self, block, layers, num_classes=100):
        super().__init__()

        self.stem = CIFARResNetStem()
        self.in_channels = 64

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1 )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,block, out_channels, blocks, stride):

        layers = []
        
        layers.append(block(self.in_channels, out_channels, stride))
        
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CIFARResNet(BasicBlock, [2,2,2,2], num_classes = 100)

    dummy = torch.randn(1, 3, 32, 32)

    model = model.to(device)
    dummy = dummy.to(device)
    output = model(dummy)

    print(output.shape)

        



