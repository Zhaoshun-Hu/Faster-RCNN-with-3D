from glob import glob
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

fileName = "data\\Topogram_L9\\"
mode = "test"
labelIdx=[1,2,3,4,5,6,7,8,9]

fileName1 = glob(fileName+mode+"\\*"+".npy")
fileName2 = glob(fileName+mode+"\\sagittal\\*"+".npy")
fileName3 = glob(fileName+mode+"\\transverse\\*"+".npy")


out = {}
for i in range(30):
    out[i] = []
for index in range(len(fileName1)):
    data = np.load(fileName1[index])
    data2 = np.load(fileName2[index])
    data3 = np.load(fileName3[index])

    for i in labelIdx:
        # if np.max(data[i]==0): continue
        mask = np.array(np.where(data[i]==1))
        y_min,x_min = np.min(mask,axis = 1)
        y_max,x_max = np.max(mask,axis = 1)
        dx = x_max-x_min
        out[int(i)].append([x_max-x_min,(y_max-y_min)/(x_max-x_min)])
    preIDX = 10
    for i in labelIdx:
        # if np.max(data[i]==0): continue
        mask = np.array(np.where(data2[i]==1))
        y_min,x_min = np.min(mask,axis = 1)
        y_max,x_max = np.max(mask,axis = 1)
        out[int(i+preIDX)].append([x_max-x_min,(y_max-y_min)/(x_max-x_min)])

    for i in labelIdx:
        # if np.max(data[i]==0): continue
        mask = np.array(np.where(data3[i]==1))
        y_min,x_min = np.min(mask,axis = 1)
        y_max,x_max = np.max(mask,axis = 1)
        out[int(i+preIDX*2)].append([x_max-x_min,(y_max-y_min)/(x_max-x_min)])
for i in range(30):
    out[i] = np.around(np.mean(out[i],axis=0),2)
    print(i,"\n",out[i])

# print(out)
