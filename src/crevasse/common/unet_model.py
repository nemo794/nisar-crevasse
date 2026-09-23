"""
U-Net architecture for crevasse segmentation.

BIOMASS-specific divergence from the nisar-crevasse-unet original this file was copied
from: `UNet` gained an optional `aux_decoder` (a second, denoising-autoencoder decoder
sharing the same encoder) -- see the class docstring and
biomass-crevasse-unet/docs/PIPELINE.md for why. `aux_decoder=False` (the default)
reproduces the original architecture exactly, so this divergence does not affect any
checkpoint trained before it existed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two conv layers with BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling or transposed conv
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # After concatenation: in_channels (upsampled) + in_channels//2 (skip) = in_channels * 3//2
            # But we want to handle the concatenated channels properly
            # So conv takes in_channels as input (which is already the concatenated size)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Handle size mismatch due to padding
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net for semantic segmentation.

    Args:
        n_channels: Number of input channels (1 for single-band SAR)
        n_classes: Number of output classes (1 for binary crevasse segmentation)
        bilinear: Use bilinear upsampling (True) or transposed convolutions (False)
        base_channels: Number of channels in first layer (default 64)
        aux_decoder: BIOMASS-specific addition (not in the NISAR original this file
            was copied from -- see biomass-crevasse-unet/docs/PIPELINE.md for why).
            When True, adds a SECOND decoder sharing the same encoder and skip
            connections, trained as a plain denoising autoencoder (reconstructs the
            raw input tile) rather than to segment. The idea: a bottlenecked
            autoencoder's easiest way to minimize reconstruction error is to capture
            the DOMINANT, repeating structure in a tile -- here, the documented
            PSF/speckle confound -- since genuine crevasse structure is a comparatively
            sparse minority of the signal. `forward()` stays backward compatible: with
            `aux_decoder=False` (the default) it returns exactly what it always has (a
            single logits tensor), so every existing checkpoint, `bio_eval_shard.py`,
            and `bio_unet_predict.py` call keeps working unmodified. Only when
            `aux_decoder=True` does it return `(logits, noise_recon)`.
    """

    def __init__(self, n_channels=1, n_classes=1, bilinear=True, base_channels=64,
                 aux_decoder=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.aux_decoder = aux_decoder

        # Encoder
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)

        # Segmentation decoder - in_channels is the concatenated size (upsampled + skip)
        # For up1: bottleneck (512 with factor=2) + skip from down3 (512) = 1024
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        # For up2: up1 output (256 with factor=2) + skip from down2 (256) = 512
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        # For up3: up2 output (128) + skip from down1 (128) = 256
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        # For up4: up3 output (64) + skip from inc (64) = 128
        self.up4 = Up(base_channels * 2, base_channels, bilinear)

        # Output
        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)

        # Noise (reconstruction) decoder -- same shape as the segmentation decoder,
        # separate weights, same shared encoder features and skip connections. Only
        # built when requested so the non-aux model's parameter count/state_dict is
        # unchanged from before this option existed.
        if aux_decoder:
            self.up1_n = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
            self.up2_n = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
            self.up3_n = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
            self.up4_n = Up(base_channels * 2, base_channels, bilinear)
            self.outc_n = nn.Conv2d(base_channels, n_channels, kernel_size=1)

    def forward(self, x):
        # Encoder (shared)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Segmentation decoder
        s = self.up1(x5, x4)
        s = self.up2(s, x3)
        s = self.up3(s, x2)
        s = self.up4(s, x1)
        logits = self.outc(s)

        if not self.aux_decoder:
            return logits

        # Noise decoder -- independent weights, same encoder features/skips
        n = self.up1_n(x5, x4)
        n = self.up2_n(n, x3)
        n = self.up3_n(n, x2)
        n = self.up4_n(n, x1)
        noise_recon = self.outc_n(n)
        return logits, noise_recon

    def get_encoder_features(self, x):
        """Extract encoder features for pretraining."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # Bottleneck features
        return x5


class MAE_Encoder(nn.Module):
    """
    Masked Autoencoder Encoder for self-supervised pretraining.
    Uses same architecture as U-Net encoder.
    """

    def __init__(self, n_channels=1, base_channels=64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        self.down4 = Down(base_channels * 8, base_channels * 16)

        self.embedding_dim = base_channels * 16

    def forward(self, x, mask=None):
        """
        Args:
            x: Input image [B, C, H, W]
            mask: Optional binary mask [B, 1, H, W] (1 = visible, 0 = masked)

        Returns:
            Bottleneck features [B, embedding_dim, H/16, W/16]
        """
        if mask is not None:
            x = x * mask

        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        return x


class MAE_Decoder(nn.Module):
    """
    Masked Autoencoder Decoder for self-supervised pretraining.
    Reconstructs the original image from encoded features.
    Simple decoder without skip connections.
    """

    def __init__(self, embedding_dim, n_channels=1, base_channels=64):
        super().__init__()

        # Simple upsampling decoder (no skip connections for MAE)
        self.decoder = nn.Sequential(
            # [B, 1024, H/16, W/16] -> [B, 512, H/8, W/8]
            nn.ConvTranspose2d(base_channels * 16, base_channels * 8, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),

            # [B, 512, H/8, W/8] -> [B, 256, H/4, W/4]
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),

            # [B, 256, H/4, W/4] -> [B, 128, H/2, W/2]
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # [B, 128, H/2, W/2] -> [B, 64, H, W]
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # [B, 64, H, W] -> [B, 1, H, W]
            nn.Conv2d(base_channels, n_channels, kernel_size=1)
        )

    def forward(self, encoded):
        """
        Args:
            encoded: Bottleneck features [B, embedding_dim, H/16, W/16]

        Returns:
            Reconstructed image [B, C, H, W]
        """
        return self.decoder(encoded)


class MaskedAutoencoder(nn.Module):
    """
    Complete MAE model for self-supervised pretraining on SAR tiles.
    """

    def __init__(self, n_channels=1, base_channels=64, mask_ratio=0.75):
        super().__init__()
        self.encoder = MAE_Encoder(n_channels, base_channels)
        self.decoder = MAE_Decoder(base_channels * 16, n_channels, base_channels)
        self.mask_ratio = mask_ratio

    def random_masking(self, x, mask_ratio=None):
        """
        Random patch masking.

        Args:
            x: Input [B, C, H, W]
            mask_ratio: Fraction of patches to mask (default: self.mask_ratio)

        Returns:
            masked_x: Input with masked patches set to 0
            mask: Binary mask [B, 1, H, W]
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio

        B, C, H, W = x.shape
        patch_size = 16  # Patch size for masking

        # Create patch grid
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Random mask for patches
        mask = torch.rand(B, 1, num_patches_h, num_patches_w, device=x.device)
        mask = (mask > mask_ratio).float()

        # Upsample mask to image size
        mask = F.interpolate(mask, size=(H, W), mode='nearest')

        masked_x = x * mask
        return masked_x, mask

    def forward(self, x, mask=None):
        """
        Forward pass with optional masking.

        Args:
            x: Input image [B, C, H, W]
            mask: Optional mask (if None, creates random mask)

        Returns:
            reconstruction: Reconstructed image [B, C, H, W]
            mask: Binary mask used [B, 1, H, W]
        """
        if mask is None:
            masked_x, mask = self.random_masking(x)
        else:
            masked_x = x * mask

        # Encode
        encoded = self.encoder(masked_x, mask)

        # Decode
        reconstruction = self.decoder(encoded)

        return reconstruction, mask

    def get_encoder_for_unet(self):
        """
        Extract encoder weights to initialize U-Net.
        """
        return self.encoder


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test models
    print("Testing U-Net...")
    unet = UNet(n_channels=1, n_classes=1, base_channels=64)
    print(f"U-Net parameters: {count_parameters(unet):,}")

    x = torch.randn(2, 1, 512, 512)
    y = unet(x)
    print(f"Input: {x.shape} → Output: {y.shape}")

    print("\nTesting MAE...")
    mae = MaskedAutoencoder(n_channels=1, base_channels=64, mask_ratio=0.75)
    print(f"MAE parameters: {count_parameters(mae):,}")

    reconstruction, mask = mae(x)
    print(f"Input: {x.shape} → Reconstruction: {reconstruction.shape}")
    print(f"Mask ratio: {1 - mask.mean():.2%}")
