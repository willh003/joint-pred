import torch
from torchvision import transforms
from torch import nn
from torchvision import models
from dreamsim import dreamsim

def cosine_dist(emb_a, emb_b):
    return 1 - nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)

class HingeLoss(torch.nn.Module):
    def __init__(self, device, margin):
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x, y):
        y_rounded = torch.round(y) # Map [0, 1] -> {0, 1}
        y_transformed = -1 * (1 - 2 * y_rounded) # Map {0, 1} -> {-1, 1}
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()

class ProjectEmbedding(torch.nn.Module):
    def __init__(self, embed_module, embed_dim, output_dim):
        super(ProjectEmbedding, self).__init__()
        self.embed_module = embed_module
        self.proj = nn.Linear(embed_dim, output_dim) 

    def forward(self, x):
        emb = self.embed_module(x)
        pred = self.proj(emb)
        return pred

def load_dino_for_ft(device, output_dim, freeze_backbone=False):
    embed_module = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    DINO_EMB_DIM = 1024 # using dino l which has dim 1024
    model = ProjectEmbedding(embed_module, DINO_EMB_DIM, output_dim)
    model.to(device)

    if freeze_backbone:
        model.proj.train().requires_grad_()
        model.embed_module.requires_grad_(False)
    else:
        model.train().requires_grad_()

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return model, preprocess


def load_ds_for_ft(device, cache_dir):
    ds, _ = dreamsim(pretrained=True,
                                    device=device,
                                    cache_dir=cache_dir,
                                    normalize_embeds=True)
    
    ds.train().requires_grad_(True)

    transform = transforms.Compose([
        transforms.Resize((224, 224),
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    return ds, transform

def load_resnet50_for_ft(device, output_dim=256, pretrained=True, freeze_backbone=True, checkpoint=None):
    """
    Load a resnet50 pretrained model

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """
    # Initialize pre-trained model
    model = models.resnet50(pretrained)

    # Replace the class prediction head with an embedding prediction head
    model.fc = nn.Linear(model.fc.in_features, output_dim) 

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

    # Freeze the parameters of the pre-trained layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer for fine-tuning
        for param in model.fc.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True

    # from the resnet50 code, this is the image format expected
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    model = model.to(device) 

    return model, transform
