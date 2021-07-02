from PIL import Image
import os
import numpy as np
import csv
import torch
from torch.utils import data
from torchvision import transforms


def prepare_dataset(opts):
    train_ops = [
        opts.data_folder + "1/",
        # opts.data_folder + "2/",
        # opts.data_folder + "3/"
    ]
    test_ops = [opts.data_folder + "4/"]

    train_set = load_ops(train_ops, opts)
    test_set = load_ops(test_ops, opts)

    return train_set, test_set


def load_ops(subfolders, opts):
    ops = []
    for path in subfolders:
        for ID in os.listdir(path):
            op_path = os.path.join(path, ID)
            if os.path.isdir(op_path):
                anno_file = opts.annotation_folder + "video" + ID + "-tool.txt"
                dataset = Cholec80Anticipation(
                    op_path, anno_file, opts.width, opts.height, opts.horizon
                )
                dataloader = data.DataLoader(
                    dataset, batch_size=opts.batch_size, shuffle=False, num_workers=2
                )
                ops.append((ID, dataloader))
    return ops


def generate_anticipation_gt_onetool(tool_code, horizon):
    anticipation = torch.zeros_like(tool_code).type(torch.FloatTensor)
    tool_count = horizon
    for i in torch.arange(len(tool_code) - 1, -1, -1):
        if tool_code[i]:
            tool_count = 0
        else:
            tool_count = min(1, tool_count + 1 / (60 * horizon))
        anticipation[i] = tool_count
    return anticipation


def generate_anticipation_gt(tools, horizon):
    return torch.stack(
        [generate_anticipation_gt_onetool(tool_code, horizon) for tool_code in tools]
    ).permute(1, 0)


class Cholec80Anticipation(data.Dataset):
    def __init__(self, image_path, annotation_path, width, height, horizon=5):
        self.image_path = image_path
        self.width = width
        self.height = height
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        with open(annotation_path, "r") as f:
            tool_presence = []
            reader = csv.reader(f, delimiter="\t")
            next(reader, None)
            for i, row in enumerate(reader):
                tool_presence.append([int(row[x]) for x in [2, 4, 5, 6, 7]])
            tool_presence = torch.LongTensor(tool_presence).permute(1, 0)

        self.target_reg = generate_anticipation_gt(tool_presence, horizon)
        self.target_cls = torch.where(
            (self.target_reg < 1) & (self.target_reg > 0),
            torch.Tensor([2]),
            self.target_reg,
        ).type(torch.long)

    def __getitem__(self, index):
        target_reg = self.target_reg[index]
        target_cls = self.target_cls[index]

        frame = self.image_path + "/%08d.png" % index
        im = Image.open(frame)
        w = im.width
        h = im.height
        height2 = int(self.width * (h / w))

        offset_y = (self.height - height2) // 2

        img_y = im.resize((self.width, height2))
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        img.paste(img_y, box=(0, offset_y))
        img = self.transform(img)

        return img, target_reg, target_cls

    def __len__(self):
        return len(self.target_reg)
