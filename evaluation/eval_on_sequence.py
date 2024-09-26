from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import hydra
import torch
from training.models import load_resnet50_for_ft, load_dino_for_ft, load_resnet50_ensemble, load_resnet50_reco
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

def eval_sequences_qpos(model, transform, gif_paths, target_joints):
    model.requires_grad_(False)
    model.eval()
    all_errors = []
    
    for gif_path in gif_paths:
        gif_path = gif_path.replace(".gif", "_states.npy")
        frames = load_gif_frames(gif_path, output_type='pil') # just load pil so we can use the transform on them
        gt_joints = torch.as_tensor(np.load(gif_path))[:22]
        b, n_joints, d_joint =  gt_joints.shape
        gt_joints_flat = gt_joints.view(b, n_joints * d_joint)

def eval_sequences_xpos(model, transform, gif_paths, target_joints, target_conditioned=False, estimate_uncertainty=False):
    model.requires_grad_(False)
    all_errors = []
    all_uncertainties = []

    for gif_path in gif_paths:
        frames = load_gif_frames(gif_path, output_type='pil') # just load pil so we can use the transform on them
        gt_joints = torch.as_tensor(np.load(convert_gif_path_to_joints(gif_path)))
        gt_joints = gt_joints - gt_joints[:, 1, :].unsqueeze(1)
        b, n_joints, d_joint =  gt_joints.shape
        gt_joints_flat = gt_joints.view(b, n_joints * d_joint)

        # assert len(torch.nonzero(target_joints[1])) == 0, "Error: target joints were not centered on torso"
        # assert len(torch.nonzero(gt_joints[:, 1, :])) == 0, "Error: gt joints were not centered on torso"

        errors = []
        gt_rewards = []
        pred_rewards = []
        uncertainties = []
        for i, frame in enumerate(frames):
            frame_transformed = transform(frame)[None].cuda()
            gt_reward = compute_rewards(target_joints[None], gt_joints[i][None])
            
            if target_conditioned:
                pred_reward = model(frame_transformed).detach().cpu()[0]
            else:
                if estimate_uncertainty:
                    joint_pos_pred, emb, emb_reco = model(frame_transformed)
                    joint_pos_pred = joint_pos_pred.detach().cpu()[0]
                    uncertainty = torch.nn.functional.mse_loss(emb, emb_reco)
                    uncertainties.append(uncertainty.item()) # average uncertainty over joints
                else:
                    joint_pos_pred = model(frame_transformed).detach().cpu()[0]
                mse = torch.nn.functional.mse_loss(gt_joints_flat[i], joint_pos_pred)
                errors.append(mse.item())      
                pred_reward = compute_rewards(target_joints[None], joint_pos_pred.view(n_joints, d_joint)[None])[0]
            
            gt_rewards.append(gt_reward)
            pred_rewards.append(pred_reward)
            
        all_errors.append(errors)
        all_uncertainties.append(uncertainties)

        plt.plot(gt_rewards, label="gt")
        plt.plot(pred_rewards, label="pred")
        plt.title("Pred vs gt rewards")
        plt.legend()
        fp = gif_path.split('/')[-1]
        plt.savefig(f"{fp}_rewards.png")
        plt.clf()

    if estimate_uncertainty:
        for i, uncertainty in enumerate(all_uncertainties):
            plt.plot(uncertainty, label=gif_paths[i].split("/")[-1])
        plt.title("Average uncertainty over the joints")
        plt.legend()
        plt.savefig(f"uncertainty.png")
        plt.clf()

    if not target_conditioned:
        for i, errors in enumerate(all_errors):
            plt.plot(errors, label=gif_paths[i].split("/")[-1])
        plt.title("Mse of pred joints vs actual joints")
        plt.legend()
        plt.savefig("mse_pred_gt_joints.png")    

@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if cfg.training.tag == "resnet_linear":
        model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint)
    elif cfg.training.tag == "dino_linear":
        model, transform = load_dino_for_ft(device, output_dim=cfg.training.output_dim, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint)
    elif cfg.training.tag == "resnet_ensemble":
        model,transform = load_resnet50_ensemble(device, cfg.training.output_dim, cfg.training.checkpoint)
    elif cfg.training.tag == "resnet_reco":
        model,transform = load_resnet50_reco(device, output_dim=cfg.training.output_dim, checkpoint=cfg.training.checkpoint)

    gif_paths = cfg.evaluation.gif_paths
    estimate_uncertainty = "ensemble" in cfg.training.tag or "reco" in cfg.training.tag

    if cfg.evaluation.target_joint_state.endswith(".png") or cfg.evaluation.target_joint_state.endswith(".jpg"):
        image = Image.open(cfg.evaluation.target_joint_state)
        target_joints = model(transform(image)[None].to(device))[0].view(18,3).detach().cpu()
        target_conditioned = cfg.training.target_joint_pos is not None
    else:
        target_joints = torch.as_tensor(np.load(cfg.evaluation.target_joint_state))
        target_joints = target_joints - target_joints[1]
        target_conditioned = cfg.training.target_joint_pos is not None

    eval_sequences_xpos(model, transform, gif_paths, target_joints, target_conditioned, estimate_uncertainty=estimate_uncertainty)

if __name__=="__main__":
    main()

