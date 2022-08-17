import torch.nn as nn
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
import cv2
import numpy as np
class Detector(nn.Module):
    def __init__(self):
        super(Detector, self).__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        num_classes = 2
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets):
        return self.model(images, targets)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))

    def test(self, name, scale_percent=60, show=False):
        model = self.model.eval()
        image = cv2.imread(name)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
    
        # resize image
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        img /= 255.0
        img = torchvision.transforms.ToTensor()(img)
        out = model([img])
        if show:
            for i, box in enumerate(out[0]['boxes'].detach().numpy()):
                if(out[0]['scores'][i].item()>0):
                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (220, 0, 0), 1)
            cv2.imshow('image', image)
            cv2.waitKey(0)
        return out[0]