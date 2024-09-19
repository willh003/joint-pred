import gymnasium
from loguru import logger
import os
import imageio
import numpy as np
import copy
from PIL import Image
import argparse
import hydra
import torch
from torchvision.utils import save_image
from training.models import load_resnet50_for_ft
from training.train_joint_pos import main as train_main
from training.train_joint_pos import forward_joint_model
from data.joint_dataset import load_joint_dataset, collate_fn_generator
from torch.utils.data import DataLoader

# Set up
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# Get egl (mujoco) rendering to work on cluster
os.environ["NVIDIA_VISIBLE_DEVICES"] = "all"
os.environ["NVIDIA_DRIVER_CAPABILITIES"] = "compute,graphics,utility,video"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "device"

def convert_gif_path_to_state(path):
    return path.replace(".gif", "_states.npy")

def render_env_from_obs(env, obs):
    env.reset()
    init_qpos = env.unwrapped.init_qpos
    new_qpos = copy.deepcopy(init_qpos)
    new_qpos[2:24] = copy.deepcopy(obs[0:22])
    env.unwrapped.set_state(qpos=new_qpos, qvel=np.zeros((23,)))
    frame = env.render()
    image = Image.fromarray(frame)
    return image

def render_model_pred(env, model, transform, gt_obs):
    # Reset the environment
    loss=0
    for i, _ in enumerate(gt_obs):
        gt_image = render_env_from_obs(env, gt_obs[i])
        
        gt_image.save(f"gt_{i}.png")
        pred_obs = model(gt_image[None].cuda()).detach().cpu()

        pred_image = render_env_from_obs(env, pred_obs[0])
        loss += torch.nn.functional.mse_loss(gt_obs, pred_obs)
        
        pred_image.save(f"pred_{i}.png") 
    return loss

def render_model_preds_from_dataloader(env, model, transform, dataloader):
    loss = 0
    for i, (image, obs) in enumerate(dataloader):
        pred = model(image.cuda()).detach().cpu()

        b, n_joints, d_joint = obs.shape
        
        obs_flat = obs.view(b, n_joints*d_joint)
        
        loss += torch.nn.functional.mse_loss(obs_flat, pred)

        pred_image = render_env_from_obs(env, pred[0])
        save_image(image, f"gt_{i}.png")
        pred_image.save(f"pred_{i}.png") 
    logger.info(f"mse: {loss / len(dataloader)}")


def render_model_preds_from_gif(env, model, transform, path):
    
    gt_states = torch.as_tensor(np.load(convert_gif_path_to_state(path)))[:22]
    for state in gt_states: 
        render_model_pred(env, model, transform, state[None])

@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):
    
    gymnasium.register(
        "HumanoidSpawnedUpCustom",
        "vis.gym_env:HumanoidEnvCustom",
    )

    # Load the humanoid environment
    make_env_kwargs = dict(
        episode_length = 120,
        reward_type = "original"
    )
    env = gymnasium.make(
                'HumanoidSpawnedUpCustom',
                render_mode="rgb_array",
                **make_env_kwargs,
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone, checkpoint=cfg.training.checkpoint) 
    
    dataset = load_joint_dataset(splits=["mini"], 
                        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
                        use_hf=cfg.data.use_hf,
                        use_xpos=cfg.data.use_xpos)

    test_dataset = dataset["mini"]
    collate_fn = collate_fn_generator(transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)
    
    render_model_preds_from_dataloader(env, model, transform, test_loader)
    #render_model_preds_from_gif(env, model, transform, "/share/portal/hw575/CrossQ/train_logs/2024-09-10-113620_sb3_sac_envr=left_arm_out_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=debug-sdtw/eval/250000_rollouts.gif")
    #render_model_preds(env, model, transform, test_loader)

if __name__=="__main__":
    main()