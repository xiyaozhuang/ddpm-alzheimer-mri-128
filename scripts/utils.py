import glob

import matplotlib.pyplot as plt
import torch
from config import TrainingConfig
from diffusers import DDPMScheduler
from PIL import Image
from torchvision import transforms


def transform(examples):
    preprocess = transforms.Compose(
        [
            transforms.Resize((TrainingConfig.image_size, TrainingConfig.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    images = [preprocess(image.convert("RGB")) for image in examples["image"]]

    return {"images": images}


def plot_input_sample(dataset):
    fig, axs = plt.subplots(1, 4)

    for i, image in enumerate(dataset[:4]["image"]):
        axs[i].imshow(image)
        axs[i].set_axis_off()

    plt.savefig("images/input_sample.png")


def plot_noisy_sample(dataset):
    dataset.set_transform(transform)
    sample_image = dataset[0]["images"].unsqueeze(0)

    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    noise = torch.randn(sample_image.shape)
    timesteps = torch.LongTensor([50])
    noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

    Image.fromarray(
        ((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0]
    ).save("images/noisy_sample.png")


def plot_output_sample():
    sample_images = sorted(glob.glob(f"{TrainingConfig.output_dir}/samples/*.png"))
    Image.open(sample_images[-1]).save("images/output_sample.png")
