from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import hydra
import torch
from torch.utils.data import DataLoader

from training.models import load_resnet50_for_ft
from training.train_joint_pos import val_step
from data.joint_dataset import load_joint_dataset, collate_fn_generator

from data.compute_rewards import compute_rewards
import matplotlib.pyplot as plt

@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint)
    
    dataset = load_joint_dataset(splits=["manual_test", "test"], 
                        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
                        use_hf=cfg.data.use_hf,
                        use_xpos=cfg.data.use_xpos)

    manual_test_dataset = dataset["manual_test"]
    test_dataset = dataset["test"]
    collate_fn = collate_fn_generator(transform)

    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)
    manual_test_loader = DataLoader(manual_test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)

    criterion = torch.nn.functional.mse_loss

    if cfg.training.target_joint_pos is not None:
        target_joint_pos = torch.as_tensor(np.load(cfg.training.target_joint_pos)).float().to(device)

        def loss_fn(model, img, gt_joint_pos, device):
            return forward_task_conditioned_model(model, img, gt_joint_pos, criterion, target_joint_pos, device)  
    else:
        def loss_fn(model, img, gt_joint_pos, device):
            return forward_task_conditioned_model(model, img, gt_joint_pos, criterion, device)  


    test_loss = val_step(model, test_loader, loss_fn, device)
    manual_test_loss = val_step(model, manual_test_loader, loss_fn, device)

if __name__=="__main__":
    main()

