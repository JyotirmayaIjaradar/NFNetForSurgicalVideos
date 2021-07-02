import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models.resnet

import Dataloader_Cholec
import argparse
import os.path
torch.multiprocessing.set_sharing_strategy('file_system')
import datetime
from shutil import copy2
from sklearn.metrics import f1_score
from nfnets import NFNet, SGD_AGC, pretrained_nfnet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Folder containing the different frames for training")
    parser.add_argument("--output_folder", type=str, default="./nfnet_output", help="Folder to store the outputs of the training process")
    parser.add_argument("--epochs", type=int, default=100, help="epochs to train")

    opt = parser.parse_args()

    data_folder = str(opt.data_folder)
    output_folder = str(opt.output_folder)
    epochs = opt.epochs

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    #print(device)


    os.makedirs(output_folder, exist_ok=True)
    copy2(os.path.realpath(__file__),output_folder)
    num_class = 7


    batch_size = 128


    transforms =  transforms.Compose([
        transforms.Resize((192, 192)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_sets = []
    data_folder=r"/mnt/g27prist/TCO/TCO-Studenten/jyotirmaya/ins_ant-master/data/images/"
    for dI in os.listdir(data_folder):
        op_path = os.path.join(data_folder, dI)

        if os.path.isdir(op_path):
            #print(op_path)
            set = Dataloader_Cholec.InstrumentDataCholec80(op_path, width=192, height=192,transform=transforms, preload=True)
            train_sets.append(torch.utils.data.DataLoader(set, batch_size=64, shuffle=False, num_workers=0))

    net = NFNet(num_classes=num_class,
            variant='F0',
            stochdepth_rate=0.25,
            alpha=0.2,
            se_ratio=0.5,
            activation='gelu'
)
    sig = nn.Sigmoid()



    net.to(device)
    print(net)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = SGD_AGC(
        named_params=net.named_parameters(),
        lr=0.1,
        momentum=0.9,
        clipping=0.1,
        weight_decay=0.00002,
        nesterov=True
    )
    for params in optimizer.param_groups:
        name = params['name']

        if net.exclude_from_weight_decay(name):
            params['weight_decay'] = 0
        if net.exclude_from_clipping(name):
            params['clipping'] = None

    f_log = open(output_folder + "log.txt", "w")
    f_log.write("Data folder= " + data_folder + "\n")
    f_log.write("Epochs= " + str(epochs) + "\n")
    for epoch in range(100):

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

        print("Epoche %3d: Train (loss %.3f, F1 %.3f)" % (epoch + 1, train_loss/train_count, train_f1))
        f_log.write("Epoche %3d: Train (loss %.3f, F1 %.3f)\n"  % (epoch + 1, train_loss/train_count, train_f1))

        if (epoch + 1)%25 == 0:
            torch.save(net.state_dict(),output_folder + "model%d.pkl" % (epoch+1))
    f_log.close()

    torch.save(net.state_dict(), output_folder + "model%d.pkl" % (epoch + 1))


