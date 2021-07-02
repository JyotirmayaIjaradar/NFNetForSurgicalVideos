import torch
from torch import nn
from torchvision.models import resnet50


class BayesFeatureResNet(nn.Module):
    def __init__(self, num_features=4096):
        super(BayesFeatureResNet, self).__init__()

        self.mode = "VARIATIONAL"

        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = Identity()
        self.fc = nn.Sequential(nn.Linear(2048, num_features), nn.ReLU(inplace=True))

    def forward(self, x, dropout_mask=(1, 1, 1, 1, 1)):
        x = self.resnet(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def init_dropout_mask(self, prob=0.2):
        return (
            nn.Dropout(p=prob)(torch.ones(64, 53, 95)).cuda(),
            nn.Dropout(p=prob)(torch.ones(192, 26, 47)).cuda(),
            nn.Dropout(p=prob)(torch.ones(384, 12, 23)).cuda(),
            nn.Dropout(p=prob)(torch.ones(256, 12, 23)).cuda(),
            nn.Dropout(p=prob)(torch.ones(256, 12, 23)).cuda(),
        )

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def set_mode(self, mode="VARIATIONAL"):
        if mode in ["VARIATIONAL", "DETERMINISTIC"]:
            self.mode = mode
        else:
            print(
                "Mode unchanged since {} is not a valid mode. Possible modes are VARIATIONAL and DETERMINISTIC".format(
                    mode
                )
            )


class BayesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BayesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = "VARIATIONAL"

        self.cell = nn.LSTMCell(input_size, hidden_size)

    # requires sequence dimensions (SEQ_LENGTH x BATCH_SIZE x INPUT_SIZE)
    # output has dimensions (SEQ_LENGTH x BATCH_SIZE x HIDDEN_SIZE)
    def forward(self, sequence, hidden_state, dropout_mask=(1, 1)):
        h, c = hidden_state
        mask_x, mask_h = dropout_mask
        output = []
        for x in sequence:
            if self.mode == "VARIATIONAL":
                h, c = self.cell(x * mask_x, (h * mask_h, c))
            else:
                h, c = self.cell(x, (h, c))
            output.append(h.view(1, h.size(0), -1))
        output = torch.cat(output, 0)
        return output, (h, c)

    def init_hidden(self):
        return (
            torch.zeros(1, self.hidden_size).cuda(),
            torch.zeros(1, self.hidden_size).cuda(),
        )

    def init_dropout_mask(self, prob=0.2, batch_size=1):
        return (
            nn.Dropout(p=prob)(torch.ones(batch_size, self.input_size)).cuda(),
            nn.Dropout(p=prob)(torch.ones(batch_size, self.hidden_size)).cuda(),
        )

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def set_mode(self, mode="VARIATIONAL"):
        if mode in ["VARIATIONAL", "DETERMINISTIC"]:
            self.mode = mode
        else:
            print(
                "Mode unchanged since {} is not a valid mode. Possible modes are VARIATIONAL and DETERMINISTIC".format(
                    mode
                )
            )


class BayesLSTMResNet(nn.Module):
    def __init__(self, num_classes, lstm_size=512, lstm_input_size=4096, pretrain=None):
        super(BayesLSTMResNet, self).__init__()

        self.featureNet = BayesFeatureResNet(num_features=lstm_input_size)
        self.lstm = BayesLSTM(lstm_input_size, lstm_size)
        self.classifier = nn.Linear(lstm_size, num_classes)

        if pretrain is not None:
            self.load(pretrain)

        self.feature_dropout_mask = (1, 1, 1, 1, 1)
        self.lstm_dropout_mask = (1, 1)

    def forward(self, x, hidden_state):
        x = self.featureNet.forward(x, self.feature_dropout_mask)
        x = x.view(x.size(0), 1, -1)
        x, hidden_state = self.lstm(x, hidden_state, self.lstm_dropout_mask)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x, hidden_state

    def init_hidden(self):
        return self.lstm.init_hidden()

    def init_dropout_mask(self, prob=0.2):
        self.feature_dropout_mask = self.featureNet.init_dropout_mask(prob=prob)
        self.lstm_dropout_mask = self.lstm.init_dropout_mask(prob=prob)

    def load(self, model_file):
        self.load_state_dict(torch.load(model_file))

    def save(self, model_file):
        torch.save(self.state_dict(), model_file)

    def set_mode(self, mode="VARIATIONAL"):
        if mode in ["VARIATIONAL", "DETERMINISTIC"]:
            self.featureNet.mode = mode
            self.lstm.mode = mode
        else:
            print(
                "Mode unchanged since {} is not a valid mode. Possible modes are VARIATIONAL and DETERMINISTIC".format(
                    mode
                )
            )


#### Identity Layer ####
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x
