import torch
from torchvision import transforms
from torch import nn
from torch.distributions import Normal

from torchvision import models
from dreamsim import dreamsim

def cosine_dist(emb_a, emb_b):
    return 1 - nn.functional.cosine_similarity(emb_a, emb_b, dim=-1)

class Normal:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

class HingeLoss(torch.nn.Module):
    def __init__(self, device, margin):
        super(HingeLoss, self).__init__()
        self.device = device
        self.margin = margin

    def forward(self, x, y):
        y_rounded = torch.round(y) # Map [0, 1] -> {0, 1}
        y_transformed = -1 * (1 - 2 * y_rounded) # Map {0, 1} -> {-1, 1}
        return torch.max(torch.zeros(x.shape).to(self.device), self.margin + (-1 * (x * y_transformed))).sum()

def probabilistic_regression_loss(input: Normal, target: torch.Tensor):
    """
    https://arxiv.org/pdf/1612.01474
    
    input is a NormalPosterior containing params of a normal distribution representing the approximate posterior, with shape (B, N)
    target is a target vector of shape (B, N)

    TODO: I think this is slightly wrong, fix it
    """
    neg_log_likelihood = torch.log(torch.square(input.sigma)) / 2 + torch.square(target - input.mu) / (2 * torch.square(input.sigma))
    return torch.mean(neg_log_likelihood) # MLE: minimize the negative log likelihood

class Ensemble(nn.Module):
    def __init__(self, embed_module, heads, backbone_output_dict=None):
        """
        Initializes the ensemble of ProjectEmbeddingsPosterior applied to a shared backbone.

        Args:
            backbone (nn.Module): The shared feature extraction model (e.g., ResNet).
            heads (int): The classification heads.
            backbone_output_dict (dict): a dictionary mapping {output: torch.Tensor}, which stores the backbone output
                - If None, this will just use the forward pass
            project_embeddings_class (class): The class for the ProjectEmbeddings head.
            *project_args, **project_kwargs: Arguments to pass to ProjectEmbeddings class.
        """
        super(Ensemble, self).__init__()
        self.embed_module = embed_module
        self.num_heads = len(heads)
        # Create an ensemble of ProjectEmbeddings heads
        self.heads = nn.ModuleList(heads)

    def forward(self, x, head_idx=None):
        """
        Forward pass for the ensemble
        See https://arxiv.org/pdf/1612.01474

        Args:
            x (Tensor): Input batch of data.
            head_idx (int, optional): Index of the specific head to use. If None, outputs from all heads will be returned using vmap.
        
        Returns:
            Tensor: Output from the chosen head(s).
        """
        # Pass input through the shared backbone
        emb = self.embed_module(x)

        if head_idx is not None:
            # Pass through the selected head
            return self.heads[head_idx](emb)
        else:
            # TODO: use vmap to vectorize over all heads
            preds = torch.stack([head(emb) for head in self.heads])
            return preds.mean(dim=0), preds.var(dim=0)

class HookModule(nn.Module):
    def __init__(self, model, output_dict):
        super(HookModule, self).__init__()
        self.output_dict = output_dict
        self.model = model
    
    def forward(self, x):
        self.model(x)
        return self.output_dict["output"]

class NormalPosterior(torch.nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(NormalPosterior, self).__init__()
        self.proj = nn.Linear(embed_dim, output_dim * 2) 

        # Linear layers to predict the mean and log-variance of the Gaussian
        self.proj_mu = nn.Linear(embed_dim, output_dim)
        self.proj_log_var = nn.Linear(embed_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, emb) -> Normal:
        """
        Forward pass to project embeddings into a Gaussian distribution.

        Args:
            x (Tensor): Input data.
        
        Returns:
            dist.Normal: A Normal distribution representing the approximate posterior.
        """        
        mu = self.proj_mu(emb)
        log_var = self.proj_log_var(emb)
        var = torch.exp(log_var)
        sigma = torch.sqrt(var)

        return Normal(mu, sigma)

class BackboneHead(torch.nn.Module):
    def __init__(self, backbone, head):
        super(BackboneHead, self).__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        emb = self.backbone(x)
        pred = self.head(emb)
        return pred

class ProjectEmbedding(torch.nn.Module):
    def __init__(self, embed_module, embed_dim, output_dim):
        super(ProjectEmbedding, self).__init__()
        self.embed_module = embed_module
        self.proj = nn.Linear(embed_dim, output_dim) 

    def forward(self, x):
        emb = self.embed_module(x)
        pred = self.proj(emb)
        return pred

class MultipleOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()

class BackboneHeadReco(torch.nn.Module):
    def __init__(self, backbone, head, reconstruct):
        super(BackboneHeadReco, self).__init__()
        self.backbone = backbone
        self.head = head
        self.reconstruct = reconstruct
        
    def forward(self, x):
        emb = self.backbone(x)
        pred = self.head(emb)
        emb_reco = self.reconstruct(emb)
        return pred, emb, emb_reco

class Bottleneck(torch.nn.Module):
    def __init__(self, embed_dim, hidden_dim):
        super(Bottleneck, self).__init__()
        self.encode = nn.Linear(embed_dim, hidden_dim) 
        self.decode = nn.Linear(hidden_dim, embed_dim) 

    def forward(self, x):
        emb = self.encode(x)
        pred = self.decode(emb)
        return pred

def load_dino_for_ft(device, output_dim, freeze_backbone=False, checkpoint = None):
    embed_module = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
    DINO_EMB_DIM = 1024 # using dino l which has dim 1024
    model = ProjectEmbedding(embed_module, DINO_EMB_DIM, output_dim)
    model.to(device)

    if freeze_backbone:
        model.proj.train().requires_grad_()
        model.embed_module.requires_grad_(False)
    else:
        model.train().requires_grad_()

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)

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
    # model = models.resnet50(pretrained=pretrained)
    # model.fc = nn.Linear(model.fc.in_features, output_dim)
    # Initialize pre-trained model
    backbone_out = {}
    def hook(model, input, output):
        backbone_out["output"] = torch.flatten(output.detach(), 1)
    
    resnet = models.resnet50(pretrained=pretrained)
    resnet.avgpool.register_forward_hook(hook)
    backbone = HookModule(resnet, backbone_out)
    
    # Replace the class prediction head with a 3 layer MLP embedding prediction head
    intermediate_dim = backbone.model.fc.in_features // 2
    head = nn.Sequential(
        nn.Linear(backbone.model.fc.in_features, intermediate_dim),
        nn.Linear(intermediate_dim, intermediate_dim),
        nn.Linear(intermediate_dim, output_dim)
    )
    model = BackboneHead(backbone, head)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print("loaded state dict")
        
    # # Disable grad on the backbone if frozen    
    # for param in model.backbone.parameters():
    #     param.requires_grad = not freeze_backbone

    # # Unfreeze the last layer for fine-tuning
    # for param in model.head.parameters():
    #     param.requires_grad = True

    # from the resnet50 code, this is the image format expected
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    model = model.to(device) 

    return model, transform

def save_backbone_head(model, ckpt_path):
    checkpoint = {
        'head': model.head.state_dict()
    }
    # Save the checkpoint
    torch.save(checkpoint, ckpt_path)

def save_default(model, ckpt_path):
    """
    Save a model's state dict to a checkpoint
    """
    torch.save(model.state_dict(), ckpt_path) 

def load_resnet50_probabilistic(device, output_dim=256, pretrained=True, freeze_backbone=True, checkpoint=None):
    """
    Load a resnet50 pretrained model

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
        checkpoint: a checkpoint to a state dict saved with save_resnet50_probabilistic (must have "backbone_fc" and "head")
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """

    # Initialize pre-trained model
    backbone = models.resnet50(pretrained)

    embed_dim =  backbone.fc.in_features // 2
    # Replace the class prediction head with an embedding prediction head
    backbone.fc = nn.Linear(backbone.fc.in_features, embed_dim) 
    head = NormalPosterior(embed_dim=embed_dim, output_dim=output_dim)
    
    if checkpoint is not None:

        state_dict = torch.load(checkpoint)
        backbone.fc.load_state_dict(state_dict['backbone_fc'])
        head.load_state_dict(state_dict['head'])

        print("loaded state dict")
    
    model = BackboneHead(backbone, head)

    # Freeze the parameters of the pre-trained layers
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the last layer of resnet + projection layer for fine-tuning 
        for param in model.embed_module.fc.parameters():
            param.requires_grad = True
        for param in model.proj.parameters():
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

def load_resnet50_ensemble(device, output_dim=256, checkpoints=None):
    """
    Load a list of resnet50 pretrained models into an ensemble, using a forward hook to get backbone embeddings

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """

    backbone_out = {}
    def hook(model, input, output):
        backbone_out["output"] = torch.flatten(output.detach(), 1)
    
    resnet = models.resnet50(pretrained=True)
    resnet.avgpool.register_forward_hook(hook)
    backbone = HookModule(resnet, backbone_out)

    heads = []
    for i, ckpt_path in enumerate(checkpoints):
        head = nn.Linear(backbone.model.fc.in_features, output_dim)
        state_dict = torch.load(ckpt_path)
        linear_dict = {'weight': state_dict['fc.weight'], 'bias': state_dict['fc.bias']}
        head.load_state_dict(linear_dict)
        heads.append(head)
        print("loaded state dict")
    
    model = Ensemble(backbone, heads)

    # Ensemble is only used at inference time
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # from the resnet50 code, this is the image format expected
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    model = model.to(device) 

    return model, transform


def load_resnet50_reco(device, output_dim=256, pretrained=True, freeze_backbone=False, checkpoint=None):
    """
    Load a resnet50 pretrained model

    Inputs:
        pretrained: if the model should pretrained (on ImageNetV2) or not
    Outputs:
        torch.nn.Module: resnet50 torch model
        torch.transforms.Transform: transform to apply to PIL images before input
    """
    # model = models.resnet50(pretrained=pretrained)
    # model.fc = nn.Linear(model.fc.in_features, output_dim)
    # Initialize pre-trained model
    backbone_out = {}
    def hook(model, input, output):
        backbone_out["output"] = torch.flatten(output.detach(), 1)
    
    resnet = models.resnet50(pretrained=pretrained)
    resnet.avgpool.register_forward_hook(hook)
    backbone = HookModule(resnet, backbone_out)

    embed_dim = backbone.model.fc.in_features
    hidden_dim = embed_dim // 2
    reco_bottleneck_dim = 32
    
    head= nn.Sequential(
        nn.Linear(embed_dim, hidden_dim),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Linear(hidden_dim, output_dim)
    ) 

    reconstruct = nn.Sequential(
        nn.Linear(embed_dim, reco_bottleneck_dim),
        nn.Linear(reco_bottleneck_dim, embed_dim),
    ) 

    model = BackboneHeadReco(backbone, head, reconstruct)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict)
        print("loaded state dict")
        
    # # Disable grad on the backbone if frozen    
    for param in model.backbone.parameters():
        param.requires_grad = not freeze_backbone

    # Unfreeze the last layer for fine-tuning
    for param in model.head.parameters():
        param.requires_grad = True
    for param in model.reconstruct.parameters():
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