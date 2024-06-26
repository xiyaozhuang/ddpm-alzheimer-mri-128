# ddpm-alzheimer-mri-128

This repository contains the model training algorithm for the [xiyaozhuang/ddpm-alzheimer-mri-128](https://huggingface.co/xiyaozhuang/ddpm-alzheimer-mri-128/tree/main) [UNet2DModel](https://huggingface.co/docs/diffusers/v0.28.2/en/api/models/unet2d#diffusers.UNet2DModel) on a subset of the [Falah/Alzheimer_MRI](https://huggingface.co/datasets/Falah/Alzheimer_MRI) dataset.

The code is based off the [Hugging Face training with diffusers](https://huggingface.co/docs/diffusers/en/tutorials/basic_training) tutorial.

For sample images see the `images` directory!

![output_sample](images/output_sample.png)

## Installation

To install the dependencies for this project:

1. Clone this repository `git clone https://github.com/xiyaozhuang/ddpm-alzheimer-mri-128.git`
1. Change the working directory `cd ddpm-alzheimer-mri-128`
1. Create a virtual environment `python3 -m venv <my_venv>`
1. Activate the virtual environment `<my_venv>/Scripts/activate` for Windows or `source <my_venv>/bin/activate` for UNIX based systems.
1. Install dependencies with `pip install -r requirements.txt`

Note that the requirements file specifies an index url to make use of GPUs with PyTorch. To install the CPU version, upgrade [PyTorch manually](https://pytorch.org/get-started/locally/) or remove the index url in the requirements file. This is not recommended as training is very computationally intensive.

## Usage

### Training

To run the training algorithm run the `scripts/main.py` file.

### Plotting

To create sample plots run the `scripts/plot.py` file after training.

Note that the input samples and noisy sample images can be plotted prior to training by removing `plot_output_sample()` from `scripts/plot.py` and should not be very computationally expensive.

### Inference

To generate MRI scans from the pretrained model see the `inference.ipynb` notebook for an example!
