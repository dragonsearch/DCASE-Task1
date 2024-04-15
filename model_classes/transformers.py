import torchvision.transforms as transforms
import torch
from torch import nn
import random
from einops.layers.torch import Rearrange
from timm.models.vision_transformer import VisionTransformer

class BaselineViT(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.transformer = nn.Sequential(
            Rearrange('b c h w -> b (h w) c'),  # Reorganizar la entrada para que coincida con el formato de entrada del ViT
            nn.Linear(40 * 51, 512),  # Ajustar el tamaño de la entrada
            nn.LayerNorm(512),
        )
        self.vit = VisionTransformer(
            img_size=1,  # Ajustar el tamaño de la imagen
            patch_size=4,
            num_classes=10,  # Número de clases de salida
            embed_dim=512,  # Ajustar el tamaño de la salida del transformer
            depth=6,
            num_heads=8,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.1,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
        )

        self.dense_1 = nn.Linear(512, 100)  # Capa densa 1
        self.dropout_1 = nn.Dropout(0.5)
        self.dense_2 = nn.Linear(100, 10)   # Capa densa 2

    def forward(self, x):
        x = self.transformer(x)  # Pasar la entrada a través del transformer
        x = self.vit(x)  # Pasar la entrada a través del modelo ViT
        x = self.dense_1(x)  # Capa densa 1
        x = self.dropout_1(x)  # Dropout
        x = torch.relu(x)  # Función de activación ReLU
        x = self.dense_2(x)  # Capa densa 2
        return x