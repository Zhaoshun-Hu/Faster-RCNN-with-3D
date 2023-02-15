import copy
import tkinter as tk
import tkinter.messagebox

import numpy as np
import PIL.Image
import PIL.ImageTk
from cv2 import FONT_HERSHEY_COMPLEX as font
from cv2 import putText as text
from cv2 import rectangle as dBbox
from cv2 import resize


def drawBboxSagittal(img_ct,l,bbox):
    [x_min, y_min, x_max, y_max] = bbox
    if l==11:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(100,0,0),1)
        text(img_ct,'1',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(100,0,0),5)
    if l==12:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0, 100,0),1)
        text(img_ct,'2',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0, 100,0),5)
    if l==13:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,100),1)
        text(img_ct,'3',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,100),5)
    if l==14:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(55,0,0),1)
        text(img_ct,'4',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(55,0,0),5)
    if l==15:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,55,0),1)
        text(img_ct,'5',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,55,0),5)
    if l==16:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,55),1)
        text(img_ct,'6',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,55),5)
    if l==17:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),1)
        text(img_ct,'7',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(255,0,0),5)
    if l==18:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,255,0),1)
        text(img_ct,'8',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,255,0),5)
    if l==19:
        dBbox(img_ct,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),1)
        text(img_ct,'9',(int((x_min+x_max)/2),int((y_min+y_max)/2)),font,1,(0,0,255),5)
    return img_ct

def drawBboxCoronal(img_ct,l,bbox):
    [x_min, y_min, x_max, y_max] = bbox
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

def drawBboxTransvers(img_ct,l,bbox):
    [x_min, y_min, x_max, y_max] = bbox
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
class GUI():
    root = tk.Tk()
    strClass = tk.StringVar()
    corScore = tk.StringVar()
    corIoU = tk.StringVar()
    corLoss = tk.StringVar()
    sagScore = tk.StringVar()
    sagIoU = tk.StringVar()
    sagLoss = tk.StringVar()
    traScore = tk.StringVar()
    traIoU = tk.StringVar()
    traLoss = tk.StringVar()
    currentClass = 0
    def __init__(self,cor,sag,tra,res):
        self.cor = cor
        self.sag = sag
        self.tra = tra
        self.res = res
        # merge
        # self.mergeRes()
        for k in res.keys():
            if len(res[k][-2]) == 6:
                self.rebuildRes()
                break

        # init image
        initImg = np.random.random([400,400,3])*255
        self.corGT = self.cv2PIL(initImg)
        self.sagGT = self.cv2PIL(initImg)
        self.traGT = self.cv2PIL(initImg)
        self.corPred = self.cv2PIL(initImg)
        self.sagPred = self.cv2PIL(initImg)
        self.traPred = self.cv2PIL(initImg)
        # init param
        self.strClass.set(self.currentClass)
        self.corScore.set(None), self.corIoU.set(None), self.corLoss.set(None)
        self.sagScore.set(None), self.sagIoU.set(None), self.sagLoss.set(None)
        self.traScore.set(None), self.traIoU.set(None), self.traLoss.set(None)

        # 4 image + 2
        self.corGtLabel = tk.Label(self.root, image=self.corGT)
        self.corGtLabel.grid(row=1, column=2, padx=2, pady=2, columnspan=1, rowspan=5)
        self.corPredLabel = tk.Label(self.root, image=self.corPred)
        self.corPredLabel.grid(row=6, column=2, padx=2, pady=2, columnspan=1, rowspan=5)
        self.sagGtLabel = tk.Label(self.root, image=self.sagGT)
        self.sagGtLabel.grid(row=1, column=3, padx=2, pady=2, columnspan=1, rowspan=5)
        self.sagPredLabel = tk.Label(self.root, image=self.sagPred)
        self.sagPredLabel.grid(row=6, column=3, padx=2, pady=2, columnspan=1, rowspan=5)
        self.traGtLabel = tk.Label(self.root, image=self.traGT)
        self.traGtLabel.grid(row=1, column=4, padx=2, pady=2, columnspan=1, rowspan=5)
        self.traPredLabel = tk.Label(self.root, image=self.traPred)
        self.traPredLabel.grid(row=6, column=4, padx=2, pady=2, columnspan=1, rowspan=5)

        # class button
        # !place
        tk.Button(self.root, text='\n     RESET      \n', command=self.setClassTo0
            ).grid(row=1, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n 1. Right Kidney \n', command=self.setClassTo1
            ).grid(row=2, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n 2. Left Kidney  \n', command=self.setClassTo2
            ).grid(row=3, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n      3. Liver      \n', command=self.setClassTo3
            ).grid(row=4, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n     4. Spleen     \n', command=self.setClassTo4
            ).grid(row=5, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n  5. Right Lung  \n', command=self.setClassTo5
            ).grid(row=6, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n   6. Left Lung   \n', command=self.setClassTo6
            ).grid(row=7, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n    7. Pancreas   \n', command=self.setClassTo7
            ).grid(row=8, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n 8. Gallbladder \n', command=self.setClassTo8
            ).grid(row=9, column=1, padx=2, pady=2, columnspan=1, rowspan=1)
        tk.Button(self.root, text='\n     9. Aorta      \n', command=self.setClassTo9
            ).grid(row=10, column=1, padx=2, pady=2, columnspan=1, rowspan=1)

        # fest label
        # colNum
        colNum_Fest = 5
        tk.Label(self.root, text="Number:  ",width=10
            ).grid(row=1, column=colNum_Fest, padx=0, pady=0)
        tk.Label(self.root, text="Score:   ",width=10
            ).grid(row=3, column=colNum_Fest, padx=0, pady=0)
        tk.Label(self.root, text="IoU:     ",width=10
            ).grid(row=4, column=colNum_Fest, padx=0, pady=0)
        tk.Label(self.root, text="DiceLoss:",width=10
            ).grid(row=5, column=colNum_Fest, padx=0, pady=0)
        tk.Label(self.root, text="  Coronal  ",width=10
            ).grid(row=2, column=colNum_Fest+1, padx=0, pady=0)
        tk.Label(self.root, text="  Sagittal  ",width=10
            ).grid(row=2, column=colNum_Fest+2, padx=0, pady=0)
        tk.Label(self.root, text="Transversal",width=10
            ).grid(row=2, column=colNum_Fest+3, padx=0, pady=0)
        # free label
        # colNum
        colNum_Free = 6
        tk.Label(self.root ,textvariable=self.strClass, bg='green', width=5
            ).grid(row=1, column=colNum_Free, padx=0, pady=0)
        corScoreLabel = tk.Label(self.root ,textvariable=self.corScore, bg='yellow', width=5)
        corScoreLabel.grid(row=3, column=colNum_Free, padx=0, pady=0)
        corIoULabel = tk.Label(self.root ,textvariable=self.corIoU, bg='yellow', width=5)
        corIoULabel.grid(row=4, column=colNum_Free, padx=0, pady=0)
        corLossLabel = tk.Label(self.root ,textvariable=self.corLoss, bg='yellow', width=5)
        corLossLabel.grid(row=5, column=colNum_Free, padx=0, pady=0)
        sagScoreLabel = tk.Label(self.root ,textvariable=self.sagScore, bg='yellow', width=5)
        sagScoreLabel.grid(row=3, column=colNum_Free+1, padx=0, pady=0)
        sagIoULabel = tk.Label(self.root ,textvariable=self.sagIoU, bg='yellow', width=5)
        sagIoULabel.grid(row=4, column=colNum_Free+1, padx=0, pady=0)
        sagLossLabel = tk.Label(self.root ,textvariable=self.sagLoss, bg='yellow', width=5)
        sagLossLabel.grid(row=5, column=colNum_Free+1, padx=0, pady=0)
        traScoreLabel = tk.Label(self.root ,textvariable=self.traScore, bg='yellow', width=5)
        traScoreLabel.grid(row=3, column=colNum_Free+2, padx=0, pady=0)
        traIoULabel = tk.Label(self.root ,textvariable=self.traIoU, bg='yellow', width=5)
        traIoULabel.grid(row=4, column=colNum_Free+2, padx=0, pady=0)
        traLossLabel = tk.Label(self.root ,textvariable=self.traLoss, bg='yellow', width=5)
        traLossLabel.grid(row=5, column=colNum_Free+2, padx=0, pady=0)

        self.tableLabel =  [[corScoreLabel,self.corScore],
                            [corIoULabel,self.corIoU],
                            [corLossLabel,self.corLoss],
                            [sagScoreLabel,self.sagScore],
                            [sagIoULabel,self.sagIoU],
                            [sagLossLabel,self.sagLoss],
                            [traScoreLabel,self.traScore],
                            [traIoULabel,self.traIoU],
                            [traLossLabel,self.traLoss]]

        tk.Button(text = "\n  NEXT  \n", command = self.root.quit
            ).grid(row=1, column=8, padx=0, pady=0)

    def setClassTo1(self): self.currentClass = 1; self.strClass.set(self.currentClass), self.update()
    def setClassTo2(self): self.currentClass = 2; self.strClass.set(self.currentClass), self.update()
    def setClassTo3(self): self.currentClass = 3; self.strClass.set(self.currentClass), self.update()
    def setClassTo4(self): self.currentClass = 4; self.strClass.set(self.currentClass), self.update()
    def setClassTo5(self): self.currentClass = 5; self.strClass.set(self.currentClass), self.update()
    def setClassTo6(self): self.currentClass = 6; self.strClass.set(self.currentClass), self.update()
    def setClassTo7(self): self.currentClass = 7; self.strClass.set(self.currentClass), self.update()
    def setClassTo8(self): self.currentClass = 8; self.strClass.set(self.currentClass), self.update()
    def setClassTo9(self): self.currentClass = 9; self.strClass.set(self.currentClass), self.update()
    def setClassTo0(self): self.currentClass = 0; self.strClass.set(self.currentClass), self.update()

    def updateImg(self,img,label):
        img = resize(img, [400,400])
        pilImg = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.uint8(img)).convert('RGB'))
        label.configure(image=pilImg)
        label.image = pilImg

    def checkExist(self,res,label,a,b,c):
        if label in res.keys():
            return True
        else:
            a.set(None),b.set(None),c.set(None)
            types = "Coronal"
            l = label
            if 20>label>10:
                types = "Sagittal"
                l = label-10
            if label>20:
                types = "Transversal"
                l = label-20
            # tkinter.messagebox.showwarning('Misdetection in '+str(types),
            #                         str(types)+' class Number '+ str(l) + ' is misdetection !!!')
            return False

    def update(self):
        # copy image
        if self.currentClass == 0:
            self.corGt = copy.copy(self.cor)
            text(self.corGt,'Cor GT',(350,500),font,1,(0,255,255),3)
            self.sagGt = copy.copy(self.sag)
            text(self.sagGt,'Sag GT',(350,500),font,1,(0,255,255),3)
            self.traGt = copy.copy(self.tra)
            text(self.traGt,'Tra GT',(350,500),font,1,(0,255,255),3)
            self.corPred = copy.copy(self.cor)
            text(self.corPred,'Cor Pred',(350,500),font,1,(0,255,255),3)
            self.sagPred = copy.copy(self.sag)
            text(self.sagPred,'Sag Pred',(350,500),font,1,(0,255,255),3)
            self.traPred = copy.copy(self.tra)
            text(self.traPred,'Tra Pred',(350,500),font,1,(0,255,255),3)
        else:
            # Coronal
            if self.checkExist(self.res,self.currentClass,self.corScore,self.corIoU,self.corLoss):
                [score,iou,xLoss,bbox_Pred,bbox_GT] = self.res[self.currentClass]
                self.corScore.set(np.around(score, decimals=2))
                self.corIoU.set(np.around(iou, decimals=2))
                self.corLoss.set(np.around(xLoss, decimals=2))
                self.corGt = drawBboxCoronal(self.corGt,self.currentClass,bbox_GT)
                self.corPred = drawBboxCoronal(self.corPred,self.currentClass,bbox_Pred)
            # Sagittal
            if self.checkExist(self.res,self.currentClass+10,self.sagScore,self.sagIoU,self.sagLoss):
                [score,iou,xLoss,bbox_Pred,bbox_GT] = self.res[self.currentClass+10]
                self.sagScore.set(np.around(score, decimals=2))
                self.sagIoU.set(np.around(iou, decimals=2))
                self.sagLoss.set(np.around(xLoss, decimals=2))
                self.sagGt = drawBboxSagittal(self.sagGt,self.currentClass+10,bbox_GT)
                self.sagPred = drawBboxSagittal(self.sagPred,self.currentClass+10,bbox_Pred)
            # Transversal
            if self.checkExist(self.res,self.currentClass+20,self.traScore,self.traIoU,self.traLoss):
                [score,iou,xLoss,bbox_Pred,bbox_GT] = self.res[self.currentClass+20]
                self.traScore.set(np.around(score, decimals=2))
                self.traIoU.set(np.around(iou, decimals=2))
                self.traLoss.set(np.around(xLoss, decimals=2))
                self.traGt = drawBboxTransvers(self.traGt,self.currentClass+20,bbox_GT)
                self.traPred = drawBboxTransvers(self.traPred,self.currentClass+20,bbox_Pred)
        # update image
        self.updateImg(self.corGt,self.corGtLabel)
        self.updateImg(self.corPred,self.corPredLabel)
        self.updateImg(self.sagGt,self.sagGtLabel)
        self.updateImg(self.sagPred,self.sagPredLabel)
        self.updateImg(self.traGt,self.traGtLabel)
        self.updateImg(self.traPred,self.traPredLabel)

        self.colorfullResult()

    def colorfullResult(self):
        def _from_rgb(rgb):
            r, g, b = rgb
            return f'#{r:02x}{g:02x}{b:02x}'

        for label, var in self.tableLabel:
            if var.get() == "None": continue
            value = float(var.get())
            if value>0.5:
                i = int((value-0.5)/0.5*255)
                label.configure(bg=_from_rgb((255-i,255,0)))
            else:
                i = int(value/0.5*255)
                label.configure(bg=_from_rgb((255,i,0)))
        pass

    def cv2PIL(self, img):
        return PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(np.uint8(img)).convert('RGB'))

    def mergeRes(self):
        for i in range(9):
            i = i+1
            [_,_,_,bbox_cor,bbox_corGT] = self.res[i] # x y
            [_,_,_,bbox_sag,bbox_sagGT] = self.res[i+10] # y z
            [_,_,_,bbox_tra,bbox_traGT] = self.res[i+20] # x z
            # [278.  74. 474. 266.] [278. 133. 474. 418.] [133.  74. 418. 265.]
            # print(bbox_cor,bbox_sag,bbox_tra)
            cor0,cor1,cor2,cor3 = bbox_cor
            sag0,sag1,sag2,sag3 = bbox_sag
            tra0,tra1,tra2,tra3 = bbox_tra
            bbox_corNew = [(cor0+sag0)/2,(cor1+tra1)/2,(cor2+sag2)/2,(cor3+tra3)/2]
            bbox_sagNew = [(sag0+cor0)/2,(sag1+tra0)/2,(sag2+cor2)/2,(sag3+tra2)/2]
            bbox_traNew = [(tra0+sag1)/2,(tra1+cor1)/2,(tra2+sag3)/2,(tra3+cor3)/2]
            # putin IoU
            self.res[i][1:3] = getIoU([512,512],bbox_corNew,bbox_corGT)
            self.res[i+10][1:3] = getIoU([512,512],bbox_sagNew,bbox_sagGT)
            self.res[i+20][1:3] = getIoU([512,512],bbox_traNew,bbox_traGT)

            # putin IoU
            self.res[i][3] = bbox_corNew
            self.res[i+10][3] = bbox_sagNew
            self.res[i+20][3] = bbox_traNew

    def rebuildRes(self):
        def six2four(bbox_6):
            x_min, y_min, z_min, x_max, y_max, z_max = bbox_6
            return  [x_min, y_min, x_max, y_max,], \
                    [x_min, z_min, x_max, z_max], \
                    [z_min, y_min, z_max, y_max]
        for i in range(9):
            i = i+1
            if i not in self.res.keys(): continue
            [score,iou,xLoss,bbox,bbox_GT] = self.res[i] # x y
            xy, yz, xz = six2four(bbox)
            xy_gt, yz_gt, xz_gt = six2four(bbox_GT)
            self.res[i] = [score,iou,xLoss,xy,xy_gt]
            self.res[i+10] = [score,iou,xLoss,yz,yz_gt]
            self.res[i+20] = [score,iou,xLoss,xz,xz_gt]


def run(cor,sag,tra,res):
    gui = GUI(cor,sag,tra,res)
    gui.root.mainloop()
    # while True:
    #     gui.root.update()

# cor = np.random.random([512,512,3])*255
# sag = np.random.random([512,512,3])*255
# res = {}
# run(cor,sag,res)

