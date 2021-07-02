import os
import numpy as np
label_path = r"/local_home/bodenstse/cholec80_1fps/tool_annotations/"
image_path=r"/local_home/bodenstse/cholec80_1fps/frames/1/"
for i in os.listdir(image_path):
    label_path = label_path + "01" + "-tool.txt"
    print(label_path)
    target_ins = np.loadtxt(label_path, delimiter="\t",skiprows=1)[:, 1:]
    print(len(target_ins))
    break
