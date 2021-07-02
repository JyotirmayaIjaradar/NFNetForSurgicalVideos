import torch
from torch import nn, optim
from networks import BayesLSTMResNet
import util
import os
import numpy as np


class AnticipationModel:
    def __init__(self, opts, train=True, pretrain=None):

        self.opts = opts
        if train:
            (
                self.result_folder,
                self.model_folder,
                self.log_path,
            ) = util.prepare_output_folders(opts)

        output_size = opts.num_ins * opts.num_class + opts.num_ins
        self.net = BayesLSTMResNet(output_size, pretrain=pretrain).cuda()

        # freezing layer
        for param in self.net.featureNet.resnet.parameters():
            param.requires_grad = False
        for param in self.net.featureNet.resnet.layer4.parameters():
            param.requires_grad = True

        if train:
            self.criterion_reg = nn.SmoothL1Loss(reduction="mean")
            self.criterion_cls = nn.CrossEntropyLoss(reduction="mean")
            self.eval_metric = nn.L1Loss(reduction="sum")
            self.optimizer = optim.Adam(
                self.net.parameters(), lr=opts.lr, weight_decay=opts.weight_decay
            )

    def init_op(self, mode):

        self.net.init_dropout_mask(prob=self.opts.drop_prob)
        hidden_state = self.net.init_hidden()

        if mode not in ["train", "test"]:
            raise NotImplementedError(
                '<mode> has to be either "train" or "test". got "{}".'.format(mode)
            )

        if mode == "train":
            self.optimizer.zero_grad()
            self.optimized = False

        return hidden_state

    def forward(self, data, hidden_state):

        output, hidden_state = self.net(data, hidden_state)
        output_reg = output[:, -self.opts.num_ins :]
        output_cls = output[:, : -self.opts.num_ins].view(
            -1, self.opts.num_class, self.opts.num_ins
        )

        return output_reg, output_cls, hidden_state

    def compute_loss(self, output_reg, output_cls, target_reg, target_cls):

        return (
            self.criterion_reg(output_reg, target_reg)
            + self.opts.loss_scale * self.criterion_cls(output_cls, target_cls)
        ) * self.opts.num_ins

    def backward(self, loss, batch):

        if batch % 3 == 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.optimized = True
        else:
            loss.backward(retain_graph=True)
            self.optimized = False

    def reset_stats(self):
        self.train_loss, self.train_reg, self.train_cls, self.train_count = 0, 0, 0, 0
        self.test_loss, self.test_reg, self.test_cls, self.test_count = 0, 0, 0, 0

    def update_stats(self, loss, output_reg, output_cls, target_reg, target_cls, mode):

        if mode == "train":
            self.train_loss += loss * output_reg.size(0)
            self.train_reg += (
                self.eval_metric(output_reg, target_reg) / self.opts.num_ins
            ).item()
            _, predicted = torch.max(output_cls, dim=1)
            self.train_cls += (predicted == target_cls).sum().item() / self.opts.num_ins
            self.train_count += target_reg.size(0)
        elif mode == "test":
            self.test_loss += loss * output_reg.size(0)
            self.test_reg += (
                self.eval_metric(output_reg, target_reg) / self.opts.num_ins
            ).item()
            _, predicted = torch.max(output_cls, dim=1)
            self.test_cls += (predicted == target_cls).sum().item() / self.opts.num_ins
            self.test_count += target_reg.size(0)
        else:
            raise NotImplementedError(
                '<mode> has to be either "train" or "test". got "{}".'.format(mode)
            )

    def summary(self, log_file, epoch):
        log_message = "Epoche {:3d}: Train (loss {:.3f}, L1 {:.3f}, Acc {:.3f}) test (loss {:.3f}, L1 {:.3f}, Acc {:.3f})".format(
            epoch,
            self.train_loss / self.train_count,
            self.train_reg / self.train_count,
            self.train_cls / self.train_count,
            self.test_loss / self.test_count,
            self.test_reg / self.test_count,
            self.test_cls / self.test_count,
        )

        print(log_message)
        log_file.write(log_message + "\n")

        if epoch % self.opts.model_save_freq == 0:
            self.net.save(
                os.path.join(self.model_folder, "model{:3d}.pkl".format(epoch))
            )

    def sample_op_predictions(self, op, num_samples):

        samples_cls, samples_reg = [], []

        for _ in range(num_samples):

            hidden_state = self.init_op("test")

            pred_cls, gt_cls = [], []
            pred_reg, gt_reg = [], []

            for data, target_reg, target_cls in op:

                data, target_reg, target_cls = (
                    data.cuda(),
                    target_reg.cuda(),
                    target_cls.cuda(),
                )

                output_reg, output_cls, hidden_state = self.forward(data, hidden_state)

                pred_reg.append(output_reg)
                gt_reg.append(target_reg)
                pred_cls.append(output_cls)
                gt_cls.append(target_cls)

            pred_reg, gt_reg = torch.cat(pred_reg), torch.cat(gt_reg)
            samples_reg.append(pred_reg)
            pred_cls, gt_cls = torch.cat(pred_cls), torch.cat(gt_cls)
            samples_cls.append(pred_cls)

        samples_reg = torch.stack(samples_reg).permute(2, 1, 0)
        gt_reg = gt_reg.permute(1, 0)
        samples_cls = torch.stack(samples_cls).permute(3, 1, 2, 0)
        gt_cls = gt_cls.permute(1, 0)

        return samples_reg, gt_reg, samples_cls, gt_cls

    def save_samples(self, samples, ID=0, epoch=0, result_folder=None):

        if result_folder is None:
            result_folder = self.result_folder

        samples_reg, gt_reg, samples_cls, gt_cls = samples

        np.save(
            "{}/sample_epoch_{}_op_{}_pred_reg.npy".format(result_folder, epoch, ID),
            samples_reg.cpu().numpy(),
        )
        np.save(
            "{}/sample_epoch_{}_op_{}_gt_reg.npy".format(result_folder, epoch, ID),
            gt_reg.cpu().numpy(),
        )
        np.save(
            "{}/sample_epoch_{}_op_{}_pred_cls.npy".format(result_folder, epoch, ID),
            samples_cls.cpu().numpy(),
        )
        np.save(
            "{}/sample_epoch_{}_op_{}_gt_cls.npy".format(result_folder, epoch, ID),
            gt_cls.cpu().numpy(),
        )
