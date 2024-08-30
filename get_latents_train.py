import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image
from accelerate import PartialState
from diffusers import AutoencoderKL
from tqdm import tqdm

distributed_state = PartialState()
device = distributed_state.device

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema", torch_dtype=torch.float16)
vae.to(device)
vae.eval()

output_dir = 'imagenet_latents_256_64k_64m/train'
os.makedirs(output_dir, exist_ok=True)

from PIL import Image

from PIL import Image

def center_crop_and_resize(img):
    original_width, original_height = img.size

    if max(original_width, original_height) > 256:
        scale_factor = 256 / max(original_width, original_height)
    elif min(original_width, original_height) < 64:
        scale_factor = 64 / min(original_width, original_height)
    else:
        scale_factor = 1  

    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    new_width = max(new_width, 64)
    new_height = max(new_height, 64)

    img = img.resize((new_width, new_height), Image.LANCZOS)
    original_width, original_height = img.size

    new_width = (original_width // 64) * 64
    new_height = (original_height // 64) * 64

    left = max((original_width - new_width) // 2, 0)
    top = max((original_height - new_height) // 2, 0)
    right = left + new_width
    bottom = top + new_height

    cropped_img = img.crop((left, top, right, bottom))

    return cropped_img



# Define the transform to include center_crop_and_resize
transform = transforms.Compose([
    transforms.Lambda(center_crop_and_resize),  # Apply the custom center crop and resize
    # transforms.Lambda(lambda img: img if max(img.size) <= 256 else img.resize(
    #     (256, int(img.size[1] * 256 / img.size[0])) if img.size[0] > img.size[1] else (int(img.size[0] * 256 / img.size[1]), 256))),
    transforms.ToTensor(),
    transforms.Lambda(lambda img: img.half()),  
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

dataset = ImageFolder(root='imagenet_folder/train', transform=transform)

def image_to_latent(image):
    image = image.unsqueeze(0).to(device).half() 
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    return latent.cpu().float()  

def save_latent(latent, label, image_path):
    class_dir = os.path.join(output_dir, str(label))
    os.makedirs(class_dir, exist_ok=True)
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(class_dir, f"{image_id}.pt")
    if not os.path.exists(save_path):  
        torch.save(latent, save_path)

data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

with distributed_state.split_between_processes(data_loader) as split_loader:
    for index, (image, label) in tqdm(enumerate(split_loader), total=len(dataset), desc="Processing Images"):
        image_path, _ = dataset.samples[index]
        latent = image_to_latent(image[0])
        save_latent(latent, label.item(), image_path)

print("Latent creation completed.")
