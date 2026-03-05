import torch
import torch.nn as nn

from backbone.mamba_vision import MambaVision
from neck.hybrid_encoder import HybridEncoder
from head.rtdetr_decoder import RTDETRTransformer

class MambaDETR(nn.Module):
    def __init__(self, num_classes=80, hidden_dim=256):
        super().__init__()
        
        # 1. Backbone
        # MambaVision 默认返回 C3, C4, C5 特征层
        # 默认 dim=128，各层维度依次为 128, 256, 512, 1024
        # 返回的后三层维度为 [256, 512, 1024]
        self.backbone = MambaVision(
            dim=128,
            depths=(3, 3, 10, 5),
            window_size=(8, 8, 14, 7),
            num_classes=0 # we only need features
        )
        
        backbone_out_channels = [256, 512, 1024]
        feat_strides = [8, 16, 32]

        # 2. Neck
        self.encoder = HybridEncoder(
            in_channels=backbone_out_channels,
            feat_strides=feat_strides,
            hidden_dim=hidden_dim,
            # 不再需要 AIFI 相关的参数，因为注意力机制已经由 MambaVision 承担
            use_encoder_idx=[], 
            num_encoder_layers=0,
        )
        
        # 3. Head
        # HybridEncoder 输出的多尺度特征通道数都等于 hidden_dim
        self.decoder = RTDETRTransformer(
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            feat_channels=[hidden_dim, hidden_dim, hidden_dim],
            feat_strides=feat_strides,
            num_levels=len(backbone_out_channels),
            num_queries=300,
            num_decoder_layers=6,
        )

    def forward(self, x, targets=None):
        # 提取骨干网络多尺度特征
        features = self.backbone(x)
        
        # 经过 Neck 进行特征融合
        encoded_features = self.encoder(features)
        
        # 经过 Head 进行解码预测
        outputs = self.decoder(encoded_features, targets)
        
        return outputs

if __name__ == "__main__":
    model = MambaDETR(num_classes=80)
    x = torch.randn(2, 3, 640, 640)
    outputs = model(x)
    print("Pred logits shape:", outputs["pred_logits"].shape)
    print("Pred boxes shape:", outputs["pred_boxes"].shape)
