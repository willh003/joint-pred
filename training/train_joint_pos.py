import os
import sys
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import numpy as np

from training.utils import get_output_folder_name, get_output_path, validate_and_preprocess_cfg
from training.models import cosine_dist, HingeLoss, load_resnet50_for_ft, load_ds_for_ft , load_dino_for_ft
from data.joint_dataset import load_joint_dataset, collate_fn_generator

from tqdm import tqdm
from loguru import logger
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset
from data.compute_rewards import compute_rewards


"""
Let S be the current joint state and T the target joint state
In this setting, we condition on T, to get P(R | T) ~ f(S, T)

P(S | I) ~ g(I)

I lies on the manifold corresponding to the space of observable mujoco images,
and P(I) is the probability of observing a certain state (which may depend on the policy)

We can parameterize g by our model (e.g., resnet)

Option 1: condition R on I and parameterize f on T, in which case we can learn a function h_T(I) corresponding to the target reward
- Offline reward learning
P(R | I, T) ~ f(g(I), T) = f_T(g(I)) = h_T(I)

Option 2: condition R on both I and T, and assign d to be a distance metric on the space shared by g(I) and T.
- We can train g(I) by using the same objective function as before, but replace f with d (which is non-parameteric)
- This will minimize the distance between g(I) and T
- In practice, we do not need to learn f, because we know the ground truth reward is defined by the same distance metric

P(R | I, T) ~ f(d(g(I), T))

"""

def forward_task_conditioned_model(model, img, gt_joint_pos, criterion, target_joint_pos, device='cuda'):
    """
    Compute the mse between rewards for the given image and target joint pos vs ground truth and target joint pos
    """

    img = img.to(device)
    gt_joint_pos = gt_joint_pos.to(device)
    target_joint_pos = target_joint_pos.to(device)
    
    pred_rewards = model(img)
    
    gt_rewards = compute_rewards(target_joint_pos[None], gt_joint_pos).unsqueeze(1)
    loss = criterion(pred_rewards, gt_rewards)
    return loss

def forward_joint_model(model, img, joint_pos, criterion, device='cuda'):
    """
    Inputs:
        model: torch model that outputs joint positions
        joint_pos: gt joint positions
        loss_fn: loss function that operates on output of model and joints
    Returns: joint prediction loss
    """ 
    # Prepare data and send it to the proper device
    img = img.to(device)
    joint_pos = joint_pos.to(device)
    b, n_joints, d_joint = joint_pos.shape
    
    joint_pos_flat = joint_pos.view(b, n_joints*d_joint)
    
    joint_pos_pred = model(img)

    loss = criterion(joint_pos_flat, joint_pos_pred)

    return loss

def train_step(model, train_loader, optimizer, loss_fn, device='cuda'):
    # Enable batch norm, dropout
    model.train()
    training_loss = 0.0

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_time = time.time()
    total_forward_pass_time = 0.0
    # Training
    for i, (img, xpos) in tqdm(enumerate(train_loader), leave=False, total=len(train_loader), desc="Training"):
        
        start_event.record()
        optimizer.zero_grad()
        #assert len(torch.nonzero(xpos[:, 1, :])) == 0, "Error: xpos was not normalized"
        loss = loss_fn(model, img, xpos, device)
        loss.backward()
        optimizer.step()
        end_event.record()
        torch.cuda.synchronize()
        
        # Update metric counts
        training_loss += loss.item()
        total_forward_pass_time += start_event.elapsed_time(end_event) / 1000 

    total_time = (time.time() - start_time) / 1000
    # Calculate average training loss
    epoch_train_loss = training_loss / len(train_loader)

    return epoch_train_loss

def val_step(model, val_loader, loss_fn, device='cuda'):
    # Disable batch norm, dropout
    model.eval()
    val_loss = 0.0

    # Validation
    with torch.no_grad():
        for i, (img, xpos) in tqdm(enumerate(val_loader), leave=False, total=len(val_loader), desc="Validation"):
            loss = loss_fn(model, img, xpos, device)

            # Update loss counts
            val_loss += loss.item()

    # Calculate validation loss
    epoch_val_loss = val_loss / len(val_loader)
    return epoch_val_loss

def train(device, run, cfg, model, loss_fn, train_loader, val_loader):
    # Define loss function 
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
            initial_val_loss = val_step(model, val_loader, loss_fn, device)
            run.log({"val_loss": initial_val_loss, "epoch": epoch})
            logger.info(f"Epoch {epoch} initial (pre-ft) val_loss={initial_val_loss}")

        epoch_train_loss = train_step(model, train_loader, optimizer, loss_fn, device)        
        epoch_val_loss = val_step(model, val_loader, loss_fn, device)

        run.log({"train_loss": epoch_train_loss, "val_loss": epoch_val_loss, "epoch": epoch})
        
        logger.info(f"Epoch {epoch} complete with train_loss={epoch_train_loss}, val_loss={epoch_val_loss}")

        # Save the model if it performs better on validation set
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            ckpt_path = os.path.join(cfg.logging.run_path, "checkpoints", f"best_model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), ckpt_path)

    logger.info('Finished Training')
    return model 

@hydra.main(config_name="train", config_path="../configs")
def main(cfg: DictConfig):
    # Process the config
    validate_and_preprocess_cfg(cfg)

    # only 1 gpu for now
    rank = 0
    device = f'cuda:{rank}'
    
    model_name = cfg.training.tag
    if "resnet" in model_name:
        # output dim should be the dim of the joints
        model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint)
    elif "dreamsim" in model_name or "ds" in model_name:
        model, transform = load_ds_for_ft(device, cache_dir='/share/portal/wph52/models/dm_local/models')
        # modify forward pass of ds to only give embeddings (so it has same format as other embedding models)
        model.forward = model.embed
    elif "dino" in model_name:
        model, transform = load_dino_for_ft(device, output_dim=cfg.training.output_dim)
    
    dataset = load_joint_dataset(splits=["train", "val", "mini"], 
                        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
                        use_hf=cfg.data.use_hf,
                        use_xpos=cfg.data.use_xpos)
    
    mini_sanity_check = False
    if mini_sanity_check:
        train_dataset = dataset["mini"]
    else:
        train_dataset = dataset["train"]
    val_dataset = dataset["val"]

    collate_fn = collate_fn_generator(transform)
    criterion = nn.functional.mse_loss

    if cfg.training.target_joint_pos is not None:
        target_joint_pos = torch.as_tensor(np.load(cfg.training.target_joint_pos)).float().to(device)
        target_joint_pos = target_joint_pos - target_joint_pos[:, 1, :].unsqueeze(1) # normalize targets

        def loss_fn(model, img, gt_joint_pos, device):
            return forward_task_conditioned_model(model, img, gt_joint_pos, criterion, target_joint_pos, device)  
    else:
        def loss_fn(model, img, gt_joint_pos, device):
            return forward_joint_model(model, img, gt_joint_pos, criterion, device)  

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

        model = train(device, wandb_run, cfg, model, loss_fn, train_loader, val_loader)
        return model, transform

if __name__=="__main__":
    main()