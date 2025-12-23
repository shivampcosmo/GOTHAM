import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat

class ChannelAttention3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        """
        Channel attention module for 3D data
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the MLP
        """
        super(ChannelAttention3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
        
        # Shared MLP for both pooled features
        self.mlp = nn.Sequential(
            nn.Conv3d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply average pooling and max pooling
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        # Combine the features and apply sigmoid activation
        out = self.sigmoid(avg_out + max_out)
        
        return out


class SpatialAttention3D(nn.Module):
    def __init__(self, kernel_size=5):
        """
        Spatial attention module for 3D data
        
        Args:
            kernel_size: Size of the convolutional kernel
        """
        super(SpatialAttention3D, self).__init__()
        
        assert kernel_size in (3, 5, 7), "Kernel size must be 3, 5, or 7"
        padding = kernel_size // 2
        
        self.conv = nn.Conv3d(2, 1, kernel_size=(kernel_size, kernel_size, kernel_size), 
                             padding=(padding, padding, padding), bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Apply average pooling and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate the features
        out = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid activation
        out = self.conv(out)
        out = self.sigmoid(out)
        
        return out


class CBAM3D(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4, patch_size=8):
        """
        Convolutional Block Attention Module (CBAM) for 3D data
        
        Args:
            in_channels: Number of input channels
            reduction_ratio: Reduction ratio for the channel attention MLP
            spatial_kernel_size: Kernel size for the spatial attention convolution
        """
        super(CBAM3D, self).__init__()
        
        self.channel_attention = ChannelAttention3D(in_channels, reduction_ratio)
        if patch_size == 4:
            spatial_kernel_size = 3
        elif patch_size == 6:
            spatial_kernel_size = 5
        elif patch_size == 8:
            spatial_kernel_size = 5
        else:
            raise ValueError("Unsupported patch_size. Supported values are 4, 6 and 8.")
        self.spatial_attention = SpatialAttention3D(spatial_kernel_size)
    
    def forward(self, x):
        # Store the input for the skip connection
        identity = x
        # print('in cbam: ',x.shape)
        # Apply channel attention
        x = x * self.channel_attention(x)
        # print('in cbam: ',x.shape)
        # Apply spatial attention
        x = x * self.spatial_attention(x)
        # print('in cbam: ',x.shape)
        # Add skip connection
        x = x + identity
        
        return x



class ResBlock3D_with_CBAM(nn.Module): 
    def __init__(self, in_channels, out_channels, patch_size): 
        super(ResBlock3D_with_CBAM, self).__init__()
        if patch_size == 4: 
            # Main path
            
            self.conv1 = nn.Conv3d(in_channels, out_channels//4, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn1 = nn.RMSNorm(out_channels//4) 
            self.relu1 = nn.LeakyReLU(inplace=True) 
            
            self.conv2 = nn.Conv3d(out_channels//4, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn2 = nn.RMSNorm(out_channels) 
            self.relu2 = nn.LeakyReLU(inplace=True)

            self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, dtype=torch.bfloat16), 
            # nn.BatchNorm3d(out_channels) 
            )
            '''
            self.conv1 = nn.Conv3d(in_channels, out_channels//8, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn1 = nn.RMSNorm(out_channels//8)
            self.relu1 = nn.LeakyReLU(inplace=True) 
            
            self.conv2 = nn.Conv3d(out_channels//8, out_channels//4, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn2 = nn.RMSNorm(out_channels//4) 
            self.relu2 = nn.LeakyReLU(inplace=True)

            self.conv3 = nn.Conv3d(out_channels//4, out_channels//2, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn3 = nn.RMSNorm(out_channels//2)
            self.relu3 = nn.LeakyReLU(inplace=True) 
            
            self.conv4 = nn.Conv3d(out_channels//2, out_channels, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn4 = nn.RMSNorm(out_channels) 
            self.relu4 = nn.LeakyReLU(inplace=True)
        
            self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, dtype=torch.bfloat16), 
            # nn.BatchNorm3d(out_channels) 
            )
            '''
        elif patch_size == 6:
            
            self.conv1 = nn.Conv3d(in_channels, out_channels//2, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn1 = nn.RMSNorm(out_channels//2)
            self.relu1 = nn.LeakyReLU(inplace=True) 
            
            self.conv2 = nn.Conv3d(out_channels//2, out_channels, kernel_size=2, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn2 = nn.RMSNorm(out_channels) 
            self.relu2 = nn.LeakyReLU(inplace=True)

            self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, bias=False, dtype=torch.bfloat16), 
            # nn.BatchNorm3d(out_channels) 
            )
        
        elif patch_size == 8: # 16 -> 8 (stride 2)
            self.conv1 = nn.Conv3d(in_channels, out_channels//8, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn1 = nn.RMSNorm(out_channels//8)
            self.relu1 = nn.LeakyReLU(inplace=True) 
            
            self.conv2 = nn.Conv3d(out_channels//8, out_channels//4, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn2 = nn.RMSNorm(out_channels//4) 
            self.relu2 = nn.LeakyReLU(inplace=True)

            self.conv3 = nn.Conv3d(out_channels//4, out_channels//2, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn3 = nn.RMSNorm(out_channels//2)
            self.relu3 = nn.LeakyReLU(inplace=True) 
            
            self.conv4 = nn.Conv3d(out_channels//2, out_channels, kernel_size=3, stride=1, padding=0, bias=False, dtype=torch.bfloat16)#, groups=in_channels) 
            self.bn4 = nn.RMSNorm(out_channels) 
            self.relu4 = nn.LeakyReLU(inplace=True)
            
            self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False, dtype=torch.bfloat16), 
            # nn.BatchNorm3d(out_channels) 
         )
        else: 
            raise ValueError("Unsupported patch_size. Supported values are 4, 6 and 8.") 

        # CBAM attention module 
        self.cbam = CBAM3D(out_channels, patch_size=patch_size) 
        # Skip connection 
        # self.skip = nn.Sequential() # if stride != 1 or in_channels != out_channels: # self.skip = nn.Sequential( # nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), # nn.BatchNorm3d(out_channels) # ) 
 
        
    def forward(self, x): 
        # Store input for skip connection 
        identity = x 
        out = self.conv1(x) 
        out = torch.moveaxis(self.bn1(torch.moveaxis(out, 1, -1)), -1, 1) 
        out = self.relu1(out) 
        
        out = self.conv2(out) 
        out = torch.moveaxis(self.bn2(torch.moveaxis(out, 1, -1)), -1, 1)
        '''
        out = self.relu2(out)

        out = self.conv3(out) 
        out = torch.moveaxis(self.bn3(torch.moveaxis(out, 1, -1)), -1, 1)
        out = self.relu3(out)

        out = self.conv4(out) 
        out = torch.moveaxis(self.bn4(torch.moveaxis(out, 1, -1)), -1, 1)
        '''
        # Apply CBAM 
        out = self.cbam(out) 
        # print(out.shape, self.skip(identity).shape) 
        # Add skip connection 
        out += self.skip(identity) 
        # Final activation 
        out = self.relu2(out) 
        return out
