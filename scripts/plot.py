from config import TrainingConfig
from datasets import load_dataset
from utils import plot_input_sample, plot_noisy_sample, plot_output_sample

dataset = load_dataset(TrainingConfig.dataset_name, split="train")

plot_input_sample(dataset)
plot_noisy_sample(dataset)
plot_output_sample()
