import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalCrossAttention(nn.Module):
    def __init__(self, in_channels, feature_dim, height, width):
        super(TemporalCrossAttention, self).__init__()
        # Linear layers to project features to Q, K, V spaces
        self.query_linear = nn.Linear(in_channels, feature_dim)
        self.key_linear = nn.Linear(in_channels, feature_dim)
        self.value_linear = nn.Linear(in_channels, feature_dim)
        
        # Dimension scaling factor (sqrt of feature dimension)
        self.scale_factor = feature_dim ** 0.5
        
        # 3x3 convolution in the feed-forward module to capture positional information
        self.conv3x3 = nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1)
        
        # Feed-forward MLP with GELU activation and residual connection
        self.feed_forward = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Linear(feature_dim * 2, feature_dim)
        )
        
        self.height = height
        self.width = width

    def forward(self, features):
        """
        Apply temporal cross-attention to the input features.

        Args:
            features (torch.Tensor): The input tensor of shape [bs, frames, channels, height, width].

        Returns:
            torch.Tensor: Output tensor after applying temporal cross-attention.
        """
        bs, frames, channels, height, width = features.size()
        
        # Split current and previous frames
        current_frame = features[:, 0, :, :, :]  # Current frame [bs, channels, height, width]  torch.Size([1, 128, 24, 77])
        previous_frames = features[:, 1:, :, :, :]  # Previous frames [bs, frames-1, channels, height, width]  torch.Size([1, 4, 128, 24, 77])
        
        # Reshape frames for linear layers
        current_frame_flat = current_frame.view(bs, channels, -1).permute(0, 2, 1)  # [bs, height*width, channels]  torch.Size([1, 1848, 128])
        previous_frames_flat = previous_frames.view(bs, frames - 1, channels, -1).permute(0, 1, 3, 2)  # [bs, frames-1, height*width, channels]  torch.Size([1, 4, 1848, 128])
        
        # Compute Q, K, V
        Q = self.query_linear(current_frame_flat)  # [bs, height*width, feature_dim]
        K = self.key_linear(previous_frames_flat)  # [bs, frames-1, height*width, feature_dim]
        V = self.value_linear(previous_frames_flat)  # [bs, frames-1, height*width, feature_dim]
        
        # Attention mechanism: Q x K^T / sqrt(d)
        attention_scores = torch.einsum('bqd,bfkd->bfqk', Q, K) / self.scale_factor  # [bs, frames-1, height*width, height*width]
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Aggregate values from previous frames
        attended_values = torch.einsum('bfqk,bfkd->bqd', attention_weights, V)  # [bs, height*width, feature_dim]
        
        # Reshape to 2D feature map
        attended_values_2d = attended_values.permute(0, 2, 1).view(bs, -1, height, width)  # [bs, feature_dim, height, width]
        
        # Feed-forward module with 3x3 convolution and residual connection
        ff_out = self.conv3x3(attended_values_2d)
        ff_out = self.feed_forward(ff_out.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view(bs, -1, height, width)
        
        # Residual connection
        output = ff_out + attended_values_2d  # torch.Size([1, 128, 24, 77])
        
        return [output.unsqueeze(1)]