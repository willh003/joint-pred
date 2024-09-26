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
import torch.nn.functional as F
import torchvision
from torchvision.utils import save_image
from training.models import load_resnet50_for_ft
from training.train_joint_pos import main as train_main
from training.train_joint_pos import forward_joint_model
from data.joint_dataset import load_joint_dataset, collate_fn_generator
from torch.utils.data import DataLoader
from evaluation.eval_on_sequence import load_gif_frames

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

def render_model_preds_from_dataloader(env, model, transform, dataloader):
    loss = 0
    for i, (image, obs) in enumerate(dataloader):
        pred = model(image.cuda()).detach().cpu()

        b, n_joints, d_joint = obs.shape
        
        obs_flat = obs.view(b, n_joints*d_joint)
        
        loss += torch.nn.functional.mse_loss(obs_flat, pred)

        pred_image = render_env_from_obs(env, pred[0])
        pred_image_torch = transform(pred_image)[None]
        
        save_image(torch.cat((image, pred_image_torch)), f"output_left_gt_{i}.png")
    logger.info(f"mse: {loss / len(dataloader)}")

def save_images_side_by_side(img1: Image, img2: Image):
    """
    Save two PIL images next to each other
    """
    # Resize the second image to match the first image's height
    img2 = img2.resize((img2.width * img1.height // img2.height, img1.height))
    
    # Concatenate the images horizontally
    total_width = img1.width + img2.width
    result = Image.new('RGB', (total_width, img1.height))
    result.paste(img1, (0, 0))
    result.paste(img2, (img1.width, 0))
    return result

def render_model_preds_from_gif(env, model, transform, path):
    
    gt_states = torch.as_tensor(np.load(convert_gif_path_to_state(path)))[:, :22]
    gif_frames = load_gif_frames(path, output_type="pil")
    display_images = []

    loss = 0
    for image, obs in zip(gif_frames, gt_states):
        image_transformed = transform(image)[None]
        pred = model(image_transformed.cuda()).detach().cpu()
        pred_image = render_env_from_obs(env, pred[0])
        loss += torch.nn.functional.mse_loss(obs[None], pred)
        
        side_by_side = save_images_side_by_side(image, pred_image)
        display_images.append(side_by_side)

    gif_name = os.path.basename(path).split('.')[0]
    display_images[0].save(
        f"{gif_name}_gt_vs_pred.gif", 
        save_all=True, 
        append_images=display_images[1:])
        
    logger.info(f"mse: {loss / len(gif_frames)}")

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
    #model, transform = train_main(cfg)

    dataset = load_joint_dataset(splits=["manual_test"], 
                        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
                        use_hf=cfg.data.use_hf,
                        use_xpos=cfg.data.use_xpos)

    test_dataset = dataset["manual_test"]
    collate_fn = collate_fn_generator(transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.training.batch_size, num_workers=cfg.training.num_cpu_workers, shuffle=False, collate_fn=collate_fn)
    
    #render_model_preds_from_dataloader(env, model, transform, test_loader)
    render_model_preds_from_gif(env, model, transform, "/share/portal/hw575/CrossQ/train_logs/2024-09-10-113620_sb3_sac_envr=left_arm_out_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=debug-sdtw/eval/10000_rollouts.gif")
    render_model_preds_from_gif(env, model, transform, "/share/portal/hw575/CrossQ/train_logs/2024-09-10-113620_sb3_sac_envr=left_arm_out_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=debug-sdtw/eval/100000_rollouts.gif")
    render_model_preds_from_gif(env, model, transform, "/share/portal/hw575/CrossQ/train_logs/2024-09-10-113620_sb3_sac_envr=left_arm_out_goal_only_euclidean_geom_xpos_rm=hand_engineered_s=9_nt=debug-sdtw/eval/250000_rollouts.gif")

    #render_model_preds(env, model, transform, test_loader)

if __name__=="__main__":
    main()