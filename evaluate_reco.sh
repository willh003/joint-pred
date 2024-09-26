python -m evaluation.eval_on_sequence "data=ft_joints_xpos_seq" "training.tag=resnet_reco" "training.output_dim=54" "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-19-155952_model\=resnet_linear_startpt\=True_freezebackbone\=False_xpos\=True_lr\=0.008nt\=None/checkpoints/best_model_epoch_93.pth" 



# python -m evaluation.visualize_preds  "training.checkpoint=/share/portal/wph52/models/joint_pred/train_logs/default/2024-09-25-175335_model\=resnet_reco_xpos\=False_data\=joints_qpos_seq_startpt\=True_freezebackbone\=True_lr\=0.0016nt\=None/checkpoints/best_model_epoch_96.pth" "data=ft_joints_qpos_seq" "training.output_dim=22"