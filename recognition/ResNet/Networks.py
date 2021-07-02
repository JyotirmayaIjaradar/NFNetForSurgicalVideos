import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet50


class FeatureResNet(nn.Module):
    def __init__(self, num_features=7, pretrained=True):
        super(FeatureResNet, self).__init__()
        self.resnet = resnet50(pretrained=pretrained)
        self.resnet.fc = Identity()
        self.fc = nn.Sequential(nn.Linear(2048, num_features), nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)


class InstrumentResNet(nn.Module):
    def __init__(self, num_classes, pretrain=None):
        super(InstrumentResNet, self).__init__()

        if pretrain is not None:
            self.featureNet = FeatureResNet(pretrained=False)
            self.featureNet.load(pretrain)
        else:
            self.featureNet = FeatureResNet()

        # self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.featureNet.forward(x)
        # x = self.classifier(x)

        return x

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def instantiate(self, model):
        params2 = model.named_parameters()
        params1 = self.named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(param1.data)


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
