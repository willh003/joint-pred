import os
import sys

import torch
from torch import nn, optim

from torch.utils.data import DataLoader

from training.utils import get_output_folder_name, get_output_path, validate_and_preprocess_cfg, collate_fn_generator
from training.models import cosine_dist, HingeLoss, load_resnet50_for_ft, get_resnet50_transform, load_ds_for_ft 

from tqdm import tqdm
from loguru import logger
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset

def forward_similarity_model(model, distance_fn, criterion,
                             img_ref, img_0, img_1, target, 
                             device='cuda'):
    """
    Given a batch of triplets and target (denoting whether the target image is img_0 or 1)
    Compute the loss and accuracy over the batch
    """ 
    # number of samples in this batch
    n_samples_batch = target.shape[0]
    
    # Prepare data and send it to the proper device
    target = target.to(device)

    emb_ref = model(img_ref)
    emb_0 = model(img_0)
    emb_1 = model(img_1)

    d_0 = distance_fn(emb_ref, emb_0)
    d_1 = distance_fn(emb_ref, emb_1)

    # use dreamsim hinge loss formulation
    logit = d_0 - d_1        
    loss = criterion(logit.squeeze(), target)
    loss /= n_samples_batch # loss per sample, instead of per batch

    decisions = torch.lt(d_1, d_0)
    batch_acc = ((target >= 0.5) == decisions).sum() / n_samples_batch

    return loss, batch_acc

def forward_joint_similarity_model(model, distance_fn, criterion,
                        img_ref, img_0, img_1, target, 
                        joint_ref, joint_0, joint_1,
                        device='cuda', linear_model=nn.Linear(1, 1)):

    """
    Given a batch of triplets and target joint positions (denoting whether the target image is img_0 or 1)
    Compute the loss and accuracy over the batch for both joint and similarity loss
    """ 
    # number of samples in this batch
    n_samples_batch = target.shape[0]
    
    # Prepare data and send it to the proper device
    target = target.to(device)

    emb_ref = model(img_ref)
    emb_0 = model(img_0)
    emb_1 = model(img_1)

    d_img_0 = distance_fn(emb_ref, emb_0)
    d_img_1 = distance_fn(emb_ref, emb_1)
    
    d_joint_0 = torch.linalg.vector_norm(joint_ref - joint_0, dim=1)
    d_joint_1 = torch.linalg.vector_norm(joint_ref - joint_1, dim=1)

    # use dreamsim hinge loss formulation for triplet loss
    logit = d_img_0 - d_img_1        
    triplet_loss = criterion(logit.squeeze(), target)
    triplet_loss /= n_samples_batch # loss per sample, instead of per batch\

    preferences = torch.lt(d_img_1, d_img_0)
    batch_acc = ((target >= 0.5) == preferences).sum() / n_samples_batch

    # compute loss between scaled embedding-based distance and joint pos distance
    # scale because embedding distance will always be in [0,1] (cosine distance), but joint distance is unbounded
    d_img_0_scaled = linear_model(d_img_0)
    joint_pos_los_0 = nn.functional.mse_loss(d_joint_0, d_img_0_scaled)

    d_img_1_scaled = linear_model(d_img_1)
    joint_pos_los_1 = nn.functional.mse_loss(d_joint_1, d_img_1_scaled)

    loss = triplet_loss + joint_pos_los_0 + joint_pos_los_1

    return loss, batch_acc

def train_step(model, train_loader, optimizer, distance_fn, criterion, device='cuda'):
    # Enable batch norm, dropout
    model.train()
    training_loss = 0.0
    training_acc = 0

    # Training
    for i, (img_ref, img_0, img_1, target) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader), desc="Training"):
        
        optimizer.zero_grad()
        loss, batch_acc = forward_similarity_model(model, distance_fn, criterion,
                                                    img_ref, img_0, img_1,
                                                    target, device)
        loss.backward()
        optimizer.step()

        # Update loss and accuracy counts
        training_loss += loss.item()
        training_acc += batch_acc.item()

    # Calculate average training loss and acc
    epoch_train_loss = training_loss / len(train_loader)
    epoch_train_acc = training_acc / len(train_loader)

    return epoch_train_loss, epoch_train_acc

def val_step(model, val_loader, optimizer, distance_fn, criterion, device='cuda'):
    # Disable batch norm, dropout
    model.eval()
    val_loss = 0.0
    val_acc = 0

    # Validation
    with torch.no_grad():
        for i, (img_ref, img_0, img_1, target) in tqdm(enumerate(val_loader), leave=False, total=len(val_loader), desc="Validation"):
            breakpoint()
            loss, batch_acc = forward_similarity_model(model, distance_fn, criterion,
                                                    img_ref, img_0, img_1,
                                                    target, device)
            # Update loss and accuracy counts
            val_loss += loss.item()
            val_acc += batch_acc.item()

    # Calculate validation loss and acc
    epoch_val_loss = val_loss / len(val_loader)
    epoch_val_acc = val_acc / len(val_loader)
    return epoch_val_loss, epoch_val_acc


def train(device, run, cfg, model, distance_fn, train_loader, val_loader):
    # Define loss function 
    criterion = HingeLoss(margin=cfg.training.margin, device=device)
    optimizer = optim.SGD(model.parameters(), lr=cfg.training.lr, momentum=cfg.training.momentum)

    # Training loop
    num_epochs = cfg.training.epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []  # To store the losses for plotting
    best_val_loss = float('inf')  # Initialize with a very large value
    # Train the model
    for epoch in tqdm(range(num_epochs), total = num_epochs, desc="Epoch"):
        # Validate before starting training
        if epoch == 0: 
            initial_val_loss, initial_val_acc = val_step(model, val_loader, optimizer, distance_fn, criterion, device)
            run.log({"val_loss": initial_val_loss, "epoch": epoch}, commit=False)
            run.log({"val_acc": initial_val_acc, "epoch": epoch}, commit=True) # commit for this step
            logger.info(f"Epoch {epoch} initial (pre-ft) val_loss={initial_val_loss}, val_acc={initial_val_acc}")

        epoch_train_loss, epoch_train_acc = train_step(model, train_loader, optimizer, distance_fn, criterion, device)
        run.log({"train_loss": epoch_train_loss, "epoch": epoch}, commit=False)
        run.log({"train_acc": epoch_train_acc, "epoch": epoch}, commit=False)

        epoch_val_loss, epoch_val_acc = val_step(model, val_loader, optimizer, distance_fn, criterion, device)
        run.log({"val_loss": epoch_train_loss, "epoch": epoch}, commit=False)
        run.log({"val_acc": epoch_train_acc, "epoch": epoch}, commit=True) # commit for this step
        
        logger.info(f"Epoch {epoch} complete with train_loss={epoch_train_loss}, train_acc={epoch_train_acc}, val_loss={epoch_val_loss}, val_acc={epoch_val_acc}")

        # Save the model if it performs better on validation set
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(cfg.logging.run_path, "checkpoints", f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

    logger.info('Finished Training')

@hydra.main(config_name="train", config_path="./configs")
def main(cfg: DictConfig):
    # Process the config
    validate_and_preprocess_cfg(cfg)

    # only 1 gpu for now
    rank = 0
    device = f'cuda:{rank}'

    train_dataset = load_dataset(cfg.data.hf_path, split="train")
    val_dataset = load_dataset(cfg.data.hf_path, split="val")

    transform = get_resnet50_transform()
    collate_fn = collate_fn_generator(transform)

    model_name = cfg.training.tag
    if "resnet" in model_name:
        model = load_resnet50_for_ft()
    elif "dreamsim" in model_name or "ds" in model_name:
        model = load_ds_for_ft(device, cache_dir='./models')
        # modify forward pass of ds to only give embeddings (so it has same format as other embedding models)
        model.forward = model.embed

    with wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.run_name,
        tags=cfg.logging.wandb_tags,
        sync_tensorboard=True,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.logging.wandb_mode,
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    ) as wandb_run:



        train_loader = DataLoader(train_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)

        train(device, wandb_run, cfg, model, cosine_dist, train_loader, val_loader)

if __name__=="__main__":
    main()