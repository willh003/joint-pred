import random
import torch
import gymnasium
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from data.joint_dataset import load_joint_dataset, collate_fn_generator
from torchvision import transforms

def visualize_random_samples(dataset, transform, num_samples=8):
    """Function to visualize randomly drawn images from the dataset."""
    # Randomly sample a few images from the dataset
    random_indices = random.sample(range(len(dataset)), num_samples)
    images = []
    
    for idx in random_indices:
        img, _ = dataset[idx]  # Assuming the dataset returns (image, label)
        img = transform(img)   # Apply the transformation (optional)
        images.append(img)
    
    # Create a grid of images and show them
    grid_img = make_grid(images, nrow=4)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid_img.permute(1, 2, 0))  # Rearrange axes for matplotlib
    plt.axis("off")
    plt.savefig("samples.png")

@hydra.main(config_name="eval", config_path="../configs")
def main(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_joint_dataset(
        splits=["train"], 
        dataset_path=cfg.data.hf_path if cfg.data.use_hf else cfg.data.root, 
        use_hf=cfg.data.use_hf,
        use_xpos=cfg.data.use_xpos
    )

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]) 

    split = dataset["train"]

    # Visualize random images from the dataset
    visualize_random_samples(split, transform, num_samples=8)

if __name__=="__main__":
    main()