from __future__ import print_function
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch.utils.data as data
import csv
import scipy.ndimage.interpolation
import skimage.color
import skimage.transform
import torch


class InstrumentDataCholec80(data.Dataset):
    def __init__(self, image_path, width, height, transform=None, preload=True):
        self.transform = transform
        self.target = []
        self.preloaded = []
        self.target_ins=[]
        self.image_path = image_path
        self.width = width
        self.preload = preload
        self.height = height
        #print(image_path)
        label_path=r"/mnt/g27prist/TCO/TCO-Studenten/jyotirmaya/ins_ant-master/data/annotations/tool_annotations/video"
        for i in os.listdir(image_path):
            #print(i)
            self.path_image=image_path+ "/"+ str(i)
            #print(self.path_image)
            self.path_label=label_path+str(i)+"-tool.txt"

            self.x=len(np.loadtxt(self.path_label , delimiter="\t",skiprows=1)[:,1:])
            self.y=min([self.x,len(os.listdir(self.path_image))])
            self.target_ins.extend(np.genfromtxt(self.path_label , delimiter="\t",skip_header=1,skip_footer=1)[:,1:self.y])
            #print(len(np.genfromtxt(self.path_label , delimiter="\t",skip_header=1,skip_footer=1)[:,1:self.y]))
            print(np.array(self.target_ins).shape)

            #print(self.path_image)

            if self.preload:
                j=0
                for j in range((min(len(self.target_ins),len(os.listdir(self.path_image))))):
                    #print(len(self.target_ins))
                    #print(self.path_image + "/%08d.png" % (j))
                    frame = self.path_image + "/%08d.png" % (j)
                    #print(frame)
                    self.preloaded.append(self.load_image(frame))
                print(len(self.preloaded))

    def load_image(self, img_file):
        im = Image.open(img_file)
        w = im.width
        h = im.height
        height2 = int(self.width * (h / w))
        offset_y = (self.height - height2) // 2

        img_y = im.resize((self.width, height2))
        img = Image.new('RGB', (self.width, self.height), (0, 0, 0))
        img.paste(img_y, box=(0, offset_y))
        return img

    def __getitem__(self, index):
        #print(index)
        target_ins = self.target_ins[index]

        frame = self.path_image + "/%08d.png" % (index)

        if self.preload and self.preloaded[index] is not None:
            img = self.preloaded[index].copy()
        else:
            img = self.load_image(frame)
            if self.preload:
                self.preloaded[index] = img.copy()

        if self.transform is not None:
            img = self.transform(img)

        return img, target_ins

    def __len__(self):
        return len(self.target_ins)
