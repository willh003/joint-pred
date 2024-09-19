#python -m training.train_joint_pos


python -m training.train_joint_pos "data=ft_joints_qpos_seq" "training.output_dim=22" "training.batch_size=16" "training.lr=.1" "training.epochs=100" 


#python -m training.train_joint_pos "data=ft_joints_qpos" "training.output_dim=22" "training.lr=.005" "training.epochs=100" "run_notes=mini_dataset"

# python -m training.train_joint_pos "training.target_joint_pos=/share/portal/hw575/CrossQ/create_demo/demos/left-arm-out_geom-xpos.npy" "training.output_dim=1" "training.lr=.0003"