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
from training.models import load_resnet50_for_ft
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
    
    gt_image = render_env_from_obs(env, gt_obs)
    
    gt_image.save(f"eval/gt.png")

    pred_obs = model(transform(gt_image))

    pred_image = render_env_from_obs(env, pred_obs)
    pred_image.save(f"eval/pred.png")    

def render_model_preds(env, model, transform, dataloader):
    for _, obs in dataloader:
        render_model_pred(env, model, transform, obs)

@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):
    
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
    
    model, transform = load_resnet50_for_ft(device, output_dim=cfg.training.output_dim, pretrained=cfg.training.start_from_pretrained, freeze_backbone=cfg.training.freeze_backbone)
    
    dataset = load_joint_dataset(splits=["manual_test"], 
                        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
                        use_hf=cfg.data.use_hf,
                        use_xpos=cfg.data.use_xpos)

    test_dataset = dataset["manual_test"]
    collate_fn = collate_fn_generator(transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)
    
    render_model_preds(env, model, transform, test_loader)

if __name__=="__main__":
    main()