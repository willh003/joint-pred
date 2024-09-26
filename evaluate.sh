# Visualize qpos
python -m evaluation.visualize_preds "data=ft_joints_qpos_seq" "training.output_dim=22" "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-20-102822_model\=resnet_linear_xpos\=False_datajoints_xpos_large_startpt\=True_freezebackbone\=False_lr\=0.008nt\=None/checkpoints/best_model_epoch_47.pth" "run_notes=big_model"

# Joint pos prediction

# python -m evaluation.eval_on_sequence "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-20-103012_model\=resnet_linear_xpos\=True_datajoints_xpos_large_startpt\=True_freezebackbone\=False_lr\=0.008nt\=None/checkpoints/best_model_epoch_48.pth" "evaluation.target_joint_state=/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out.png" "run_notes=big_model_pred_joints"

# python -m evaluation.eval_on_sequence "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-19-155952_model\=resnet_linear_startpt\=True_freezebackbone\=False_xpos\=True_lr\=0.008nt\=None/checkpoints/best_model_epoch_98.pth" "evaluation.target_joint_state=/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out.png" "run_notes=small_model_pred_joints"

# Direct reward prediction/share/portal/wph52/models/joint_pred/train_logs/default/
# python -m evaluation.eval_on_sequence "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-17-080137_model\=resnet_linear_startpt\=True_freezebackbone\=False_xpos\=True_lr\=0.0003nt\=None/checkpoints/best_model_epoch_14.pth" "training.target_joint_pos=/share/portal/hw575/CrossQ/create_demo/demos/lft-arm-out_geom-xpos.npy" "training.output_dim=1" "run_notes=rew"

