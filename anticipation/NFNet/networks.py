import torch
from torch import nn

from nfnets import NFNet

Stochdepth_rate = 0.25
Se_ratio = 0.5
Activation = "gelu"
Alpha = 0.2
Variant = "F0"


class BayesFeatureNFNet(nn.Module):
    def __init__(self, num_features=4096):
        super(BayesFeatureNFNet, self).__init__()

        self.mode = "VARIATIONAL"

        self.nfnet = NFNet(
            variant=Variant,
            num_classes=num_features,
            alpha=Alpha,
            stochdepth_rate=Stochdepth_rate,
            se_ratio=Se_ratio,
            activation=Activation,
        )

    def forward(self, x, dropout_mask=(1, 1, 1, 1, 1)):
        # if self.mode == "VARIATIONAL":
        #     mask1, mask2, mask3, mask4, mask5 = dropout_mask
        #     x = self.alexnet.features[0](x)  # Conv
        #     x = x * mask1
        #     x = self.alexnet.features[1](x)  # ReLu
        #     x = self.alexnet.features[2](x)  # MaxPool
        #     x = self.alexnet.features[3](x)  # Conv
        #     x = x * mask2
        #     x = self.alexnet.features[4](x)  # ReLU
        #     x = self.alexnet.features[5](x)  # MaxPool
        #     x = self.alexnet.features[6](x)  # Conv
        #     x = x * mask3
        #     x = self.alexnet.features[7](x)  # ReLU
        #     x = self.alexnet.features[8](x)  # Conv
        #     x = x * mask4
        #     x = self.alexnet.features[9](x)  # ReLU
        #     x = self.alexnet.features[10](x)  # Conv
        #     x = x * mask5
        #     x = self.alexnet.features[11](x)  # ReLU
        #     x = self.alexnet.features[12](x)  # MaxPool
        #     x = self.avgpool(x)

        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)  # FC

        # else:
        #     x = self.alexnet.features(x)
        #     x = self.avgpool(x)
        #     x = x.view(x.size(0), -1)
        #     x = self.fc(x)

        x = self.nfnet(x)
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


class BayesLSTMNFNet(nn.Module):
    def __init__(self, num_classes, lstm_size=512, lstm_input_size=4096, pretrain=None):
        super(BayesLSTMNFNet, self).__init__()

        self.featureNet = BayesFeatureNFNet(num_features=lstm_input_size)
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
