from torchvision.models import vgg16
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import RoIPool
from config import cfg

resnet_vars = {
    'resnet18': [torchvision.models.resnet18, 256],
    'resnet34': [torchvision.models.resnet34, 256],
    'resnet50': [torchvision.models.resnet50, 1024],
    'resnet101': [torchvision.models.resnet101, 1024],
    'resnet152': [torchvision.models.resnet152, 1024],
}

class WSDDN(nn.Module):
    def __init__(self, num_classes, num_layers=None):
        super().__init__()
        self.n_classes = num_classes

        backbone = vgg16(pretrained=True)
        self.WSDDN_base = nn.Sequential(*list(backbone.features._modules.values())[:-1])
        self.proposal_features = nn.Sequential(*list(backbone.classifier._modules.values())[:-1])
        # Fix the layers before conv3 for VGG16:
        for layer in range(10):
            for p in self.WSDDN_base[layer].parameters(): p.requires_grad = False

        self.WSDDN_pool = RoIPool(cfg.GENERAL.POOLING_SIZE, 1.0/16.0)
        self.cls_stream = nn.Linear(4096, 20)
        self.det_stream = nn.Linear(4096, 20)


    def forward(self, im_data, im_info, boxes):
        batch_size = im_data.size(0)
        base_feature = self.WSDDN_base(im_data)
        rois = boxes.new_zeros((batch_size, boxes.shape[1], 5))
        for i in range(batch_size):
            rois[i, :, 0] = i
            rois[i, :, 1:5] = boxes[i, :, :4]

        pooled_feat = self.WSDDN_pool(base_feature, rois.view(-1, 5))
        pooled_feat = pooled_feat.view(pooled_feat.size(0), -1)
        proposal_features = self.proposal_features(pooled_feat)
        proposal_features = proposal_features.view(batch_size, cfg.TRAIN.NUM_PROPOSALS, -1)

        classification_scores = F.softmax(self.cls_stream(proposal_features), dim=2)
        detection_scores = F.softmax(self.det_stream(proposal_features), dim=1)
        combined_scores = classification_scores * detection_scores
        return combined_scores

    @staticmethod
    def calculate_loss(combined_scores, target):
        image_level_scores = torch.sum(combined_scores, dim=1).squeeze(1)
        image_level_scores = torch.clamp(image_level_scores, min=0.0, max=1.0)
        loss = F.binary_cross_entropy(image_level_scores, target, reduction="sum")
        return loss