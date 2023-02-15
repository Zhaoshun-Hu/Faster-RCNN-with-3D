from glob import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import functional as F
from torch import nn

from dataloader import TopoDataModule
# from model.faster_rcnn_module import FasterRCNN
from model_v1.faster_rcnn_module import FasterRCNN
# from model3D.faster_rcnn_module import FasterRCNN as FasterRCNN_3D

if __name__ == '__main__':
    # ! set paramter
    lrs = [5e-4]
    epochs = [1000]
    num_classes = [10]
    modes = ["6label"]
    for lr, epoch, num_cls, mode in zip(lrs, epochs,num_classes,modes):
        fasterRCNN = FasterRCNN( learning_rate=lr,
                            num_classes=num_cls,
                            trainable_backbone_layers=5,
                            )

        dataloader = TopoDataModule("data\\Topogram_L9\\",batch_size=5,mode=mode,labelIdx=[1,2,3,4,5,6,7,8,9]) # 
        dataloader.setup()

        checkpoint_callback = ModelCheckpoint(  monitor="val_loss",
                                                dirpath="logs/{}-{}-{}".format(mode,lr,epoch),
                                                filename="Topo-{val_loss:.2f}-{epoch:02d}",
                                                save_top_k=5,
                                                mode="min",
                                                )
        trainer = pl.Trainer(max_epochs=epoch, gpus=1, callbacks=[checkpoint_callback])

        trainer.fit(fasterRCNN,dataloader)