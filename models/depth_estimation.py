
import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights  # Updated import

class DepthEstimationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Modern weights handling
        weights = ResNet50_Weights.DEFAULT if config['model']['pretrained'] else None
        base_model = resnet50(weights=weights)
        
        self.encoder = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        )
        
        # Decoder remains the same
         # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
    
    def forward(self, x):
        features = self.encoder(x)
        return self.decoder(features)
    
    def scale_depth(self, depth):
        """Convert normalized output to metric depth"""
        min_depth = self.config['model']['min_depth']
        max_depth = self.config['model']['max_depth']
        return min_depth + (max_depth - min_depth) * depth