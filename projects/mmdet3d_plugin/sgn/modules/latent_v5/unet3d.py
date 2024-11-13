import torch
import torch.nn as nn

class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
                nn.LayerNorm(in_channels),
                nn.Linear(in_channels, out_channels),
            )
    
    def forward(self, x):
        return self.mlp(x)
    
class UNet3D(nn.Module):
    def __init__(self, in_channels, out_channels, init_features=32):
        super(UNet3D, self).__init__()
        features = init_features
        
         # Encoding Path
        self.encoder1 = ConvBlock3D(in_channels, features)  # [1, 20, 128, 128, 16] -> [1, 32, 128, 128, 16]
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)  # [1, 32, 128, 128, 16] -> [1, 32, 64, 64, 8]
        
        self.encoder2 = ConvBlock3D(features, features * 2)  # [1, 32, 64, 64, 8] -> [1, 64, 64, 64, 8]
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)  # [1, 64, 64, 64, 8] -> [1, 64, 32, 32, 4]
        
        self.encoder3 = ConvBlock3D(features * 2, features * 4)  # [1, 64, 32, 32, 4] -> [1, 128, 32, 32, 4]
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))  # [1, 128, 32, 32, 4] -> [1, 128, 16, 16, 4]
        
        # Bottleneck with 20 output channels
        self.bottleneck = MLP(features * 4, out_channels)  
        

    def forward_encoder(self, x):
        # Encoding Path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.pool3(enc3)  # [1, 128, 16, 16, 2]
        
        # Bottleneck
        # Latent feature extraction
        latent = enc4.view(enc4.size(0), enc4.size(1), -1)  # [1, 128, 16, 16, 4] -> [1, 128, 16 * 16 * 4]
        latent = latent.squeeze(0).permute(1, 0)  # [1, 128, 16 * 16 * 4] -> [16 * 16 * 4, 128]
        latent = self.bottleneck(latent)  # [16 * 16 * 4, 128] -> [16 * 16 * 4, 20]
        latent = latent.permute(1, 0)  # [16 * 16 * 4, 20] -> [20, 16 * 16 * 4]
        return latent

