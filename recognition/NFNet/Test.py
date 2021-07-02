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
torch.multiprocessing.set_sharing_strategy('file_system')
import datetime
from shutil import copy2
from sklearn.metrics import f1_score

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_folder", type=str, help="Folder containing the different frames for testing")
    #parser.add_argument("--output_folder", type=str, help="Folder to load the outputs of the training process")
    parser.add_argument("--model", type=str, help="Path to trained model")

    opt = parser.parse_args()

    data_folder =r"/mnt/g27prist/TCO/TCO-Studenten/jyotirmaya/test"
    #output_folder = opt.output_folder
    model = r"/home/ljaradajy/Downloads/tool_presence_NFnet/nfnet_outputmodel100.pkl"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_cpu = torch.device("cpu")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [transforms.ToTensor(),
        normalize])

    width = 384
    height = 216
    batch_size = 64

    num_class = 7

    net = Networks.InstrumentAlexNet(num_class)
    sig = nn.Sigmoid()

    #net.load(output_folder + "/model_final.pkl")
    net.load(model)
    net.to(device)
    net.eval()

    test_sets = []

    for dI in os.listdir(data_folder):
        op_path = os.path.join(data_folder, dI)

        if os.path.isdir(op_path):
            print(op_path)
            set = Dataloader_Cholec.InstrumentDataCholec80(op_path, width, height, transform, False)
            test_sets.append(torch.utils.data.DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=6))

    outputs_test = []
    labels_test = []

    with torch.no_grad():
        for op in test_sets:
            for data in op:
                images, labels_cpu = data

                images = images.to(device)
                labels = labels_cpu.to(device)

                outputs = net(images)
                
                predicted = sig(outputs)
                predicted = torch.round(predicted.data).cpu().numpy()

                outputs_test.append(predicted)
                labels_test.append(labels_cpu.numpy())

        outputs_test = np.concatenate(outputs_test).astype(np.int64)
        labels_test = np.concatenate(labels_test).astype(np.int64)

        test_f1 = f1_score(labels_test, outputs_test, average="macro")

        print("Test F1 %.3f" % (test_f1))
