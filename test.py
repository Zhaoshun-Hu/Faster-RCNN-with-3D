import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from cv2 import FONT_HERSHEY_COMPLEX as font
from cv2 import putText as text
from cv2 import rectangle as dBbox
from skimage.color import gray2rgb, rgb2gray
from torchvision import transforms

from dataloader import TopoDataModule
from GUI import run as gui
# from model.faster_rcnn_module import FasterRCNN
from model_v1.faster_rcnn_module import FasterRCNN


def drawBboxSagittal(img_ct,l,x_min, y_min, x_max, y_max):
    if l==11:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(100,0,0),1)
        text(img_ct,'11',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(100,0,0),5)
    if l==12:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0, 100,0),1)
        text(img_ct,'12',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0, 100,0),5)
    if l==13:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,100),1)
        text(img_ct,'13',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,100),5)
    if l==14:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(55,0,0),1)
        text(img_ct,'14',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(55,0,0),5)
    if l==15:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,55,0),1)
        text(img_ct,'15',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,55,0),5)
    if l==16:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,55),1)
        text(img_ct,'16',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,55),5)
    if l==17:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),1)
        text(img_ct,'17',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(255,0,0),5)
    if l==18:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1)
        text(img_ct,'18',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,255,0),5)
    if l==19:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),1)
        text(img_ct,'19',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,255),5)
    return img_ct

def drawBboxCoronal(img_ct,l,x_min, y_min, x_max, y_max):
    if l==1:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(100,0,0),1)
        text(img_ct,'1',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(100,0,0),5)
    if l==2:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0, 100,0),1)
        text(img_ct,'2',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0, 100,0),5)
    if l==3:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,100),1)
        text(img_ct,'3',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,100),5)
    if l==4:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(55,0,0),1)
        text(img_ct,'4',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(55,0,0),5)
    if l==5:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,55,0),1)
        text(img_ct,'5',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,55,0),5)
    if l==6:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,55),1)
        text(img_ct,'6',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,55),5)
    if l==7:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),1)
        text(img_ct,'7',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(255,0,0),5)
    if l==8:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1)
        text(img_ct,'8',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,255,0),5)
    if l==9:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),1)
        text(img_ct,'9',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,255),5)
    return img_ct

def drawBboxTransvers(img_ct,l,x_min, y_min, x_max, y_max):
    if l==21:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(100,0,0),1)
        text(img_ct,'1',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(100,0,0),5)
    if l==22:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0, 100,0),1)
        text(img_ct,'2',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0, 100,0),5)
    if l==23:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,100),1)
        text(img_ct,'3',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,100),5)
    if l==24:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(55,0,0),1)
        text(img_ct,'4',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(55,0,0),5)
    if l==25:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,55,0),1)
        text(img_ct,'5',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,55,0),5)
    if l==26:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,55),1)
        text(img_ct,'6',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,55),5)
    if l==27:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),1)
        text(img_ct,'7',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(255,0,0),5)
    if l==28:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1)
        text(img_ct,'8',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,255,0),5)
    if l==29:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),1)
        text(img_ct,'9',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,255),5)
    return img_ct

def getIoU(size,bbox_pred,bbox_gt):
    def bboxInMask(bbox ,mask):
        x_min, y_min, x_max, y_max = bbox
        mask[int(y_min):int(y_max),int(x_min):int(x_max)] = 1
        return mask
    gt = np.zeros(size)
    pred = np.zeros(size)
    gt = bboxInMask(bbox_gt,gt)
    pred = bboxInMask(bbox_pred,pred)
    overlap = gt * pred
    allArea = gt + pred
    allArea[allArea>0] = 1
    return np.sum(overlap)/np.sum(allArea), np.sum(overlap*2)/(np.sum(gt)+np.sum(pred))

def getIoU_3D(size,bbox_pred,bbox_gt):
    def bboxInMask(bbox ,mask):
        x_min, y_min, z_min, x_max, y_max, z_max = bbox
        mask[int(z_min):int(z_max),int(y_min):int(y_max),int(x_min):int(x_max)] = 1
        return mask
    gt = np.zeros(size)
    pred = np.zeros(size)
    gt = bboxInMask(bbox_gt,gt)
    pred = bboxInMask(bbox_pred,pred)
    overlap = gt * pred
    allArea = gt + pred
    allArea[allArea>0] = 1
    return np.sum(overlap)/np.sum(allArea), np.sum(overlap*2)/(np.sum(gt)+np.sum(pred))

def showImg():
        fig = plt.figure()
        ax1 = fig.add_subplot(2,3,1)
        ax1.set_title('Coronal Pred')
        plt.imshow(img_pred)

        ax2 = fig.add_subplot(2,3,2)
        ax2.set_title('Coronal GT')
        plt.imshow(img_gt)

        ax3 = fig.add_subplot(2,3,3)
        ax3.set_title('Sagittal Pred')
        plt.imshow(img2_pred)

        ax4 = fig.add_subplot(2,3,4)
        ax4.set_title('Sagittal GT')
        plt.imshow(img2_gt)

        ax3 = fig.add_subplot(2,3,3)
        ax3.set_title('Transversal Pred')
        plt.imshow(img3_pred)

        ax4 = fig.add_subplot(2,3,4)
        ax4.set_title('Transversal GT')
        plt.imshow(img3_gt)

        plt.show()

def iouCorrection(data):#[0,1,2,3,4,5]
    def toNan(matrix): matrix[matrix == 0] = np.NaN; return matrix
    result = {}
    for i in range(1,10):
        score = []
        pd_bbox = np.zeros([3,6])
        gt_bbox = np.zeros([3,6])
        cor_l = i
        sag_l = i+10
        tan_l = i+20
        if cor_l in data.keys():
            score.append(data[cor_l][0])
            pd_bbox[0][[0,1,3,4]] = data[cor_l][-2]
            gt_bbox[0][[0,1,3,4]] = data[cor_l][-1]
        if sag_l in data.keys():
            score.append(data[sag_l][0])
            pd_bbox[1][[0,2,3,5]] = data[sag_l][-2]
            gt_bbox[1][[0,2,3,5]] = data[sag_l][-1]
        if tan_l in data.keys():
            score.append(data[tan_l][0])
            pd_bbox[2][[2,1,5,4]] = data[tan_l][-2]
            gt_bbox[2][[2,1,5,4]] = data[tan_l][-1]
        newScore = np.round(np.nanmean(toNan(np.array(score))),decimals=2)
        sz = img_gt.shape[0]
        iou, dLoss = getIoU_3D([sz,sz,sz],np.nanmean(pd_bbox,axis=0),np.nanmean(gt_bbox,axis=0))
        result[i] = [newScore,iou,dLoss,pd_bbox,gt_bbox]
    return result

# PATH = "logs\\Coronal only 2d-0.0001-500\\Topo-val_loss=0.61-epoch=231.ckpt" # 10 class 2d Cor only
# PATH = "logs\\Sagital only  2d-0.005-500\\Topo-val_loss=0.66-epoch=94.ckpt"
# PATH = "logs\\2d-0.001-500\\Topo-val_loss=0.55-epoch=349.ckpt"
# num_classes = 10

# PATH = "logs\\Cor&Sag 20cls-0.001-300\\Topo-val_loss=1.10-epoch=223.ckpt" # 20 class 3d
# num_classes = 20
# PATH = "logs\\20class-0.001-300\\Topo-val_loss=0.59-epoch=181.ckpt" # 20 class 3d
# PATH = "logs\\20class-0.0005-500\\Topo-val_loss=0.78-epoch=180.ckpt" # 20 class 3d
# num_classes = 20

# PATH = "logs\\C&S&T 30cls-0.001-300\\Topo-val_loss=1.20-epoch=275.ckpt" # 30 class
# num_classes = 30

PATH = "logs\\Feature+CLS 6label-0.0005-1001\\Topo-val_loss=1.40-epoch=989.ckpt" # 30 class
# PATH = "logs\\6label-0.0005-1000\\Topo-val_loss=1.75-epoch=633.ckpt" # 30 class
# PATH = "logs\\6label-0.0005-1000\\Topo-val_loss=1.59-epoch=134.ckpt" # 30 class
num_classes = 10

if __name__ == '__main__':
    # dataloader = TopoDataModule("data\\Topogram_L9\\",batch_size=1,mode="test",labelIdx=[1,2,3,4,5,6,7,8,9]) # _6L
    dataloader = TopoDataModule("data\\Topogram_L9\\",batch_size=1,mode="test_6L",labelIdx=[1,2,3,4,5,6,7,8,9]) # _6L
    dataloader.setup()
    fasterRCNN = FasterRCNN( learning_rate=7e-5,
                        num_classes=num_classes,
                        trainable_backbone_layers=5,
                        )
    # fasterRCNN = FasterRCNN_3D( learning_rate=7e-5,
    #                     num_classes=num_classes,
    #                     trainable_backbone_layers=5,
    #                     )
    model = fasterRCNN.load_from_checkpoint(PATH, learning_rate=7e-5,
                        num_classes=num_classes,
                        trainable_backbone_layers=5,strict=True)
    dataset = {}
    data_idx = 0
    for [img, img2, img3], label in dataloader.test_dataloader():

        labelList = model(img)
        if num_classes != 20:
            bbox = labelList[0]["boxes"].detach().numpy()
            labels = labelList[0]["labels"].detach().numpy()
            scores = labelList[0]["scores"].detach().numpy()
        else:
            # ! 20class
            labelList2 = model(img2)
            bbox= np.concatenate([labelList[0]["boxes"].detach().numpy(), labelList2[0]["boxes"].detach().numpy()])
            labels= np.concatenate([labelList[0]["labels"].detach().numpy(), labelList2[0]["labels"].detach().numpy()])
            scores= np.concatenate([labelList[0]["scores"].detach().numpy(), labelList2[0]["scores"].detach().numpy()])

        if "depths" in labelList[0].keys():
            depth = labelList[0]["depths"].detach().numpy()
            temp_bbox = []
            for i in range(len(bbox)):
                xmin,ymin,xmax,ymax = bbox[i]
                temp_bbox.append([xmin,ymin,(depth[i][0]+depth[i][1])/2,xmax,ymax,(depth[i][2]+depth[i][3])/2])
            bbox = temp_bbox
        img_rgb = np.transpose(img.numpy()[0], (1, 2, 0))
        img_ct = img_rgb/np.max(img_rgb)*255
        img_ct = np.ascontiguousarray(img_ct, dtype=np.uint8)

        img_rgb2 = np.transpose(img2.numpy()[0], (1, 2, 0))
        img_ct2 = img_rgb2/np.max(img_rgb2)*255
        img_ct2 = np.ascontiguousarray(img_ct2, dtype=np.uint8)

        img_rgb3 = np.transpose(img3.numpy()[0], (1, 2, 0))
        img_ct3 = img_rgb3/np.max(img_rgb3)*255
        img_ct3 = np.ascontiguousarray(img_ct3, dtype=np.uint8)

        img_pred = img_ct.copy()
        img_gt = img_ct.copy()
        img2_pred = img_ct2.copy()
        img2_gt = img_ct2.copy()
        img3_pred = img_ct3.copy()
        img3_gt = img_ct3.copy()

        label_gt = label["labels"].numpy()[0]
        bbox_gt = label["boxes"].numpy()[0]
        if len(bbox_gt[0])==4:
            for [x_min, y_min, x_max, y_max], l in zip(bbox_gt,label_gt):
                img_gt = drawBboxCoronal(img_gt, l, x_min, y_min, x_max, y_max)
                img2_gt = drawBboxSagittal(img2_gt,l,x_min, y_min, x_max, y_max)
                img3_gt = drawBboxTransvers(img3_gt,l,x_min, y_min, x_max, y_max)
        else:
            for [x_min, y_min, z_min, x_max, y_max, z_max], l in zip(bbox_gt,label_gt):
                img_gt = drawBboxCoronal(img_gt, l, x_min, y_min, x_max, y_max)
                img2_gt = drawBboxSagittal(img2_gt,l,y_min, z_min, y_max, z_max)
                img3_gt = drawBboxTransvers(img3_gt,l,x_min, z_min, x_max, z_max)
        res = {}
        if len(bbox[0])==4:
            for [x_min, y_min, x_max, y_max], l, score in zip(bbox,labels,scores):
                if score > 0.05:
                    # # ! temp for test
                    # l += 10
                    img_pred = drawBboxCoronal(img_pred, l, x_min, y_min, x_max, y_max)
                    img2_pred = drawBboxSagittal(img2_pred,l,x_min, y_min, x_max, y_max)
                    img3_pred = drawBboxTransvers(img3_pred,l,x_min, y_min, x_max, y_max)
                    if l<10:
                        iou,xLoss = getIoU(img_gt.shape,[x_min, y_min, x_max, y_max],bbox_gt[l-1])
                        if l in res:
                            if res[l][0]<score:
                                res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-1]]
                        else:
                            res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-1]]
                    elif 10<l<20:
                        iou,xLoss = getIoU(img_gt.shape,[x_min, y_min, x_max, y_max],bbox_gt[l-2])
                        if l in res:
                            if res[l][0]<score:
                                res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-2]]
                        else:
                            res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-2]]
                    elif 20<l<30:
                        iou,xLoss = getIoU(img_gt.shape,[x_min, y_min, x_max, y_max],bbox_gt[l-3])
                        if l in res:
                            if res[l][0]<score:
                                res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-3]]
                        else:
                            res[l] = [score,iou,xLoss,[x_min, y_min, x_max, y_max],bbox_gt[l-3]]
                    # print("Label: {}\n    Score: {:.2}\n    IoU:{:.2}\n    xxLoss: {:.2}\n".format(l,score,iou,xLoss))
        else:
            for [x_min, y_min, z_min, x_max, y_max, z_max], l, score in zip(bbox,labels,scores):
                if score > 0.15:
                    img_pred = drawBboxCoronal(img_pred, l, x_min, y_min, x_max, y_max)
                    img2_pred = drawBboxSagittal(img2_pred,l,y_min, z_min, y_max, z_max)
                    img3_pred = drawBboxTransvers(img3_pred,l,x_min, z_min, x_max, z_max)
                    if l<10:
                        iou,xLoss = getIoU_3D([512,512,512],[x_min, y_min, z_min, x_max, y_max, z_max],bbox_gt[l-1])
                        if l in res:
                            if res[l][0]<score:
                                res[l] = [score,iou,xLoss,[x_min, y_min, z_min, x_max, y_max, z_max],bbox_gt[l-1]]
                        else:
                            res[l] = [score,iou,xLoss,[x_min, y_min, z_min, x_max, y_max, z_max],bbox_gt[l-1]]
            # print(res)
        
        # dataset["testset_{}".format(data_idx)] = res
        if num_classes>10:
            dataset["testset_{}_fix".format(data_idx)] = iouCorrection(res)
        else:
            dataset["testset_{}".format(data_idx)] = res

        data_idx += 1

        # break
        # showImg()
        gui(img_ct,img_ct2,img_ct3,res)

    # from resultExport import export2xlsx
    # fname = PATH.split("\\")[-2]
    # export2xlsx(fileName=fname,dataset=dataset)