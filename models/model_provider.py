import torch
from avalanche.models import PNN
from efficientnet_pytorch import EfficientNet
from torch import nn

from models.VGGSlim import VGGSlim, MultiHeadMLP
from models.multihead_efficentNet import MultiHeadEfficientnetNet, MultiHeadEfficientnetNetNotPretrained
from models.vit_models import ViT64SingleHead, ViT64MultiHead


def parse_model_name(args):
    if args.model_name == 'wide_VGG9':
        return VGGSlim(config='wide_VGG9',
                       num_classes=args.num_classes,
                       batch_norm=True,
                       img_input_channels=3,
                       classifier_inputdim=8192,
                       init_weights=False)
    elif args.model_name == 'EfficientNet':
        return EfficientNet.from_pretrained('efficientnet-b1', in_channels=3, num_classes=args.num_classes)
    elif args.model_name == 'EfficientNet_NotPretrained':
        return EfficientNet.from_name('efficientnet-b1', in_channels=3, num_classes=args.num_classes)
    elif args.model_name == 'EfficientNet_multihead':
        return MultiHeadEfficientnetNet(config='efficientnet-b1', num_classes=10, img_input_channels=3)
    elif args.model_name == 'EfficientNet_multihead_NotPretrained':
        return MultiHeadEfficientnetNetNotPretrained(config='efficientnet-b1', num_classes=10, img_input_channels=3)
    elif args.model_name == 'multi_head_mlp':
        return MultiHeadMLP(hidden_size=args.hs, hidden_layers=2)
    elif args.model_name == 'ResNet':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=False)
        model.fc = nn.Linear(512, args.num_classes)
        return model
    elif args.model_name == 'PNN':
        return PNN(num_layers=args.num_layer,
                   in_features=12288,
                   adapter="mlp", )
        
    # For class-incremental (no task info)
    elif args.model_name == 'ViT64':
        return ViT64SingleHead(
            num_classes=args.num_classes,  # 60 for full scenario
            patch_size=8,
            embed_dim=384,
            depth=12,
            num_heads=6
        )
    
    # For task-incremental (with task info)
    elif args.model_name == 'ViT64_multihead':
        return ViT64MultiHead(
            num_classes_per_task=10,  # Always 10 per task
            patch_size=8,
            embed_dim=384,
            depth=12,
            num_heads=6
        )
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", args.model_name)
