import pytorch_lightning as pl
import torch
from torchvision.ops import box_iou

from model.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn

def _evaluate_iou(target, pred):
    if pred["boxes"].shape[0] == 0:
        # no box detected, 0 IOU
        return torch.tensor(0.0, device=pred["boxes"].device)
    return box_iou(target["boxes"], pred["boxes"]).diag().mean()

def dict2list(input):
    targets = []
    for i in range(len(input["labels"])):
        d = {}
        d["labels"] = input["labels"][i]
        d["boxes"] = input["boxes"][i]
        targets.append(d)
    return targets

class FasterRCNN(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0001,
        num_classes: int = 91,
        backbone = None,
        fpn: bool = True,
        trainable_backbone_layers: int = 3,
        # **kwargs: Any,
    ):
        super().__init__()

        self.learning_rate = learning_rate
        self.num_classes = num_classes
        self.backbone = backbone

        if backbone is None:
            self.model = fasterrcnn_resnet50_fpn(trainable_backbone_layers=trainable_backbone_layers, num_classes=self.num_classes)

            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)

    def forward(self, x):
        self.model.eval()
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # ！self add function！！！
        targets = dict2list(targets)

        targets = [{k: v for k, v in t.items()} for t in targets]
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        # print("\n", loss_dict)
        loss = sum(loss for loss in loss_dict.values())

        # * add log
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss, "log": loss_dict}

    def validation_step(self, batch, batch_idx):
        images, targets = batch

        # ！self add function！！！
        targets = dict2list(targets)

        # ! go training state
        self.switch_State(True)
        targets = [{k: v for k, v in t.items()} for t in targets]
        # fasterrcnn takes both images and targets for training, returns
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        # * add log
        self.log("val_loss", loss, on_step=True, on_epoch=True)

        # ! back val state
        self.switch_State(False)
        # fasterrcnn takes only images for eval() mode
        outs = self.model(images)
        iou = torch.stack([_evaluate_iou(t, o) for t, o in zip(targets, outs)]).mean()
        return {"val_iou": iou, "val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_iou = torch.stack([o["val_iou"] for o in outs]).mean()
        logs = {"val_iou": avg_iou}
        return {"avg_val_iou": avg_iou, "log": logs}

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=0.9,
            weight_decay=0.005,
        )

    def switch_State(self,state):
        self.model.training = state
        self.model.roi_heads.training = state
        self.model.rpn.training = state
