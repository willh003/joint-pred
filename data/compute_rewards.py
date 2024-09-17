import torch

def compute_rewards(target_joint_pos, joint_pos):
    # Only the arms are relevant
    joint_pos = joint_pos - joint_pos[:, 1, :].unsqueeze(1)
    joint_pos_relevant = joint_pos[:, 12:, :].flatten(start_dim=1)
    target_joint_pos_relevant = target_joint_pos[:, 12:, :].flatten(start_dim=1)

    pose_matching_reward = torch.exp(-torch.linalg.vector_norm(target_joint_pos_relevant - joint_pos_relevant, dim=1))
    return pose_matching_reward