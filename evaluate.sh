# Visualize qpos
python -m evaluation.visualize_preds "data=ft_joints_qpos" "training.output_dim=22" "training.batch_size=1" "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-19-091655_model\=resnet_linear_startpt\=True_freezebackbone\=False_xpos\=False_lr\=0.005nt\=None/checkpoints/best_model_epoch_36.pth" "training.lr=.005" "training.epochs=100" "logging.wandb_mode=disabled"

# Joint pos prediction
# python -m evaluation.eval_on_sequence "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-15-090415_model\=resnet_linear_startpt\=True_freezebackbone\=False_lr\=0.016nt\=None/checkpoints/best_model_epoch_38.pth" "run_notes=joint"

# Direct reward prediction
# python -m evaluation.eval_on_sequence "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-17-080137_model\=resnet_linear_startpt\=True_freezebackbone\=False_xpos\=True_lr\=0.0003nt\=None/checkpoints/best_model_epoch_14.pth" "training.target_joint_pos=/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_geom-xpos.npy" "training.output_dim=1" "run_notes=rew"

