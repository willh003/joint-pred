from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import hydra
import torch
from training.models import load_resnet50_for_ft
from data.compute_rewards import compute_rewards
import matplotlib.pyplot as plt

def convert_gif_path_to_joints(path):
    return path.replace(".gif", "_geom_xpos_states.npy")

def load_gif_frames(path: str, output_type="torch"):
    """
    output_type is either "torch" or "pil"

    Load the gif at the path into a torch tensor with shape (frames, channel, height, width)
    """
    gif_obj = Image.open(path)
    frames = [gif_obj.seek(frame_index) or gif_obj.convert("RGB") for frame_index in range(gif_obj.n_frames)]
    if output_type == "pil":
        return frames

    frames_torch = torch.stack([torch.tensor(np.array(frame)).permute(2, 0, 1) for frame in frames])
    return frames_torch

def eval_sequences(model, transform, gif_paths, target_joints = None):
    model.requires_grad_(False)
    all_errors = []

    for gif_path in gif_paths:
        frames = load_gif_frames(gif_path, output_type='pil') # just load pil so we can use the transform on them
        gt_joints = torch.as_tensor(np.load(convert_gif_path_to_joints(gif_path)))
        gt_joints_flat = torch.flatten(gt_joints, start_dim=1)
        b, n_joints, d_joint = gt_joints.shape  # batch size, number of joints, and dimensionality of each joint spec (usually 3)

        errors = []
        gt_rewards = []
        pred_rewards = []
        for i, frame in enumerate(frames):
            frame_transformed = transform(frame)[None].cuda()
            joint_pos_pred = model(frame_transformed)[0].detach().cpu()
            
            mse = torch.nn.functional.mse_loss(gt_joints_flat[i], joint_pos_pred)
            errors.append(mse.item())      
            gt_reward = compute_rewards(target_joints[None], gt_joints[i][None])[0]
            pred_reward = compute_rewards(target_joints[None], joint_pos_pred.view(n_joints, d_joint)[None])[0]
            gt_rewards.append(gt_reward)
            pred_rewards.append(pred_reward)
        
        all_errors.append(errors)
        plt.plot(gt_rewards, label="gt")
        plt.plot(pred_rewards, label="pred")
        plt.title("Pred vs gt rewards")
        plt.legend()
        fp = gif_path.split('/')[-1]
        plt.savefig(f"{fp}_rewards.png")
        plt.clf()

    for i, errors in enumerate(all_errors):
        plt.plot(errors, label=gif_paths[i].split("/")[-1])
    plt.title("Mse of pred joints vs actual joints")
    plt.legend()
    plt.savefig("mse_pred_gt_joints.png")    


    


@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint)
    gif_paths = cfg.evaluation.gif_paths
    target_joints = torch.as_tensor(np.load(cfg.evaluation.target_joint_state))
    eval_sequences(model, transform, gif_paths, target_joints)

if __name__=="__main__":
    main()

