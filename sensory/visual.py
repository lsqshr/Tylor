from torch import nn
from torchvision.models.resnet import resnet50
import torch.nn.functional as F


# Visual
class VisualNetwork(nn.Module):
    def __init__(self):
        super(VisualNetwork, self).__init__()
        self.base = resnet50(pretrained=True)

    def forward(self, x):
        x = self.base.conv1(x)
        x = self.base.bn1(x)
        x = self.base.relu(x)
        x = self.base.maxpool(x)

        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        f4 = self.base.layer4(x)

        fpool = F.adaptive_avg_pool2d(f4, 1)
        return fpool, f4
