import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet
import Networks
import Dataloader_Cholec
import argparse
import os.path

torch.multiprocessing.set_sharing_strategy("file_system")
import datetime
from shutil import copy2
from sklearn.metrics import f1_score, precision_score, recall_score
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, default="/mnt/g27prist/TCO/TCO-Studenten/jyotirmaya/ins_ant-master/data/images/", help="Folder containing the different frames for training")
    parser.add_argument("--output_folder", type=str, default="./resnet_output", help="Folder to store the outputs of the training process")
    parser.add_argument("--epochs", type=int, default=100, help="epochs to train")

    opt = parser.parse_args()

    data_folder = opt.data_folder
    output_folder = opt.output_folder
    epochs = opt.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device_cpu = torch.device()

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    output_folder += "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M") + "/"
    os.makedirs(output_folder, exist_ok=True)
    copy2(os.path.realpath(__file__), output_folder)
    num_class = 7

    width = 384
    height = 216
    batch_size = 128

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([transforms.ToTensor(), normalize])

    train_sets = []

    for dI in os.listdir(data_folder):
        op_path = os.path.join(data_folder, dI)

        if os.path.isdir(op_path):
            print(op_path)
            set = Dataloader_Cholec.InstrumentDataCholec80(
                op_path, width, height, transform_train, True
            )
            train_sets.append(
                torch.utils.data.DataLoader(
                    set, batch_size=batch_size, shuffle=False, num_workers=0
                )
            )

    t0 = time.time()
    net = Networks.InstrumentResNet(num_class)

    for param in net.featureNet.resnet.parameters():
        param.requires_grad = False
    for param in net.featureNet.resnet.layer4.parameters():
        param.requires_grad = True

    sig = nn.Sigmoid()

    net.to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=1e-4)
    f_log = open(output_folder + "log.txt", "w")
    f_log.write("Data folder= " + data_folder + "\n")
    f_log.write("Epochs= " + str(epochs) + "\n")
    for epoch in range(epochs):

        train_loss = 0
        train_count = 0

        outputs_train = []
        labels_train = []

        for op in train_sets:
            for data in op:
                images, labels_cpu = data

                images = images.to(device)
                labels = labels_cpu.to(device)

                outputs = net(images)

                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss += loss.item()

                predicted = sig(outputs)
                predicted = torch.round(predicted.data).cpu().numpy()
                outputs_train.append(predicted)
                labels_train.append(labels_cpu.numpy())

                train_count += labels.size(0)

        outputs_train = np.concatenate(outputs_train).astype(np.int64)
        labels_train = np.concatenate(labels_train).astype(np.int64)

        train_f1 = f1_score(labels_train, outputs_train, average="macro")
        train_recall = recall_score(labels_train, outputs_train, average="macro")
        train_precision = precision_score(labels_train, outputs_train, average="macro")

        print(
            "Epoche %3d: Train (loss %.3f, F1 %.3f)"
            % (epoch + 1, train_loss / train_count, train_f1)
        )
        f_log.write(
            "Epoche %3d: Train (loss %.3f, F1 %.3f)\n"
            % (epoch + 1, train_loss / train_count, train_f1)
        )

        if (epoch + 1) % 25 == 0:
            net.save(output_folder + "model%d.pkl" % (epoch + 1))
    f_log.close()

    net.save(output_folder + "model_final.pkl")
    t1 = time.time()
    total_n = t1-t0
    print("total training time: ", total_n)
