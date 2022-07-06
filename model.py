import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchsummary import summary
from glob import glob
from tqdm.notebook import tqdm

model = models.resnet34(pretrained=True)
fc_in_features = model.fc.in_features

class FaceRecog(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(FaceRecog, self).__init__()

        # Pretrained resnet34
        self.resnet34 = models.resnet34(pretrained=True)
        for param in self.resnet34.parameters():
            param.requires_grad = False
        
        modified_fc = nn.Linear(in_features = fc_in_features, out_features=num_classes)
        self.resnet34.fc = modified_fc
        
    def forward(self, x):
        return self.resnet34(x)

    def summary(self, input_size):
        return summary(self, input_size)
