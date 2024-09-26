python -m training.train_joint_pos "training=resnet_reco" "training.output_dim=54" "training.batch_size=8" "training.epochs=200" "training.freeze_backbone=False" "data=ft_joints_xpos_seq"

# python -m training.train_joint_pos "training.tag=resnet_reco" "training.output_dim=22" "training.batch_size=8" "training.lr=.0016" "training.epochs=100" "training.freeze_backbone=true" "data=ft_joints_qpos_seq"


