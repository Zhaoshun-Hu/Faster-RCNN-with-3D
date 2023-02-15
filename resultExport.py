import xlsxwriter as xls
import numpy as np

def resortData(data):
    def toNan(matrix): matrix[matrix == 0] = np.NaN; return matrix
    Score = np.zeros([3,9])
    Iou = np.zeros([3,9])
    Dice = np.zeros([3,9])
    bboxes = {}
    for i in range(1,10):
        c_label = i
        s_label = i + 10
        t_label = i + 20
        if c_label in data.keys():
            Score[0][i-1] = data[c_label][0]
            Iou[0][i-1]   = data[c_label][1]
            Dice[0][i-1]  = data[c_label][2]
        if s_label in data.keys():
            Score[1][i-1] = data[s_label][0]
            Iou[1][i-1]   = data[s_label][1]
            Dice[1][i-1]  = data[s_label][2]
        if t_label in data.keys():
            Score[2][i-1] = data[t_label][0]
            Iou[2][i-1]   = data[t_label][1]
            Dice[2][i-1]  = data[t_label][2]
    for k in data.keys():
        bboxes[k] = [data[k][-2],data[k][-1]]
    Score[Score == 0] = np.NaN
    return {"score":toNan(Score), "iou":toNan(Iou), "dice":toNan(Dice), "bboxes":bboxes}

class WorkSheet:
    def __init__(self, workbook, sheetName) -> None:
        self.ws = workbook.add_worksheet(sheetName)
        self.row = 0

    def addRow3(self, part1,part2,part3):
        c = 0
        for i in part1:
            self.ws.write(self.row,c,str(i))
            c += 1
        for i in part2:
            self.ws.write(self.row,c,str(i))
            c += 1
        for i in part3:
            self.ws.write(self.row,c,str(i))
            c += 1
        self.row+=1

    def addRow_empty(self): self.row += 1


def add_datasetSummary(dataset, summary, data_type):
    cache = []
    for k in dataset.keys():
        data = resortData(dataset[k])
        temp = np.round(np.nanmean(data[data_type],axis=0),decimals=3)
        cache.append(temp)
        summary.addRow3([k],temp,[np.round(np.nanmean(temp),decimals=3)])
    cls_mean = np.round(np.nanmean(np.array(cache),axis=0),decimals=3)
    summary.addRow3([""],cls_mean,["",np.round(np.nanmean(np.array(cache)),decimals=3)])
    summary.addRow_empty()

def export2xlsx(fileName, dataset):

    wb = xls.Workbook(fileName + '.xlsx')

    LabelName = ["1:Right Kidney","2:Left Kidney","3:Liver",
                "4:Spleen","5:Right Lung","6:Left Lung",
                "7:Pancreas","8:Gallbladder","9:Aorta"]

    summary = WorkSheet(wb, "Summary")

    summary.addRow3([""], LabelName, ["Average Score"])
    add_datasetSummary(dataset, summary, "score")
    summary.addRow3([""], LabelName, ["Average IoU"])
    add_datasetSummary(dataset, summary, "iou")
    summary.addRow3([""], LabelName, ["Average Dice Loss"])
    add_datasetSummary(dataset, summary, "dice")

    img_label = ["Score","IoU","Dice","Predict BBox","Ground Truth BBox"]
    for k in dataset.keys():
        temp_WS = WorkSheet(wb, k)
        temp_WS.addRow3(["Label"], img_label, [""])
        data = dataset[k]
        for i in range(30):
            if i in data.keys():
                score = np.round(data[i][0],decimals=3)
                iou = np.round(data[i][1],decimals=3)
                dice = np.round(data[i][2],decimals=3)
                pd = np.round(np.array(data[i][3]),decimals=3)
                gt = np.round(np.array(data[i][4]),decimals=3)
                temp_WS.addRow3([str(i)], [score, iou, dice, pd, gt],[""])
        temp_WS.addRow_empty()

    wb.close()