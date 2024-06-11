from dataclasses import dataclass


@dataclass
class TrainingConfig:
    dataset_name = "Falah/Alzheimer_MRI"
    image_size = 128

    num_epochs = 50
    train_batch_size = 16
    eval_batch_size = 16
    gradient_accumulation_steps = 1
    learning_rate = 1e-4
    lr_warmup_steps = 500
    mixed_precision = "fp16"

    save_image_epochs = 10
    save_model_epochs = 30
    output_dir = "ddpm-alzheimer-mri-128"

    hub_model_id = "xiyaozhuang/ddpm-alzheimer-mri-128"
    push_to_hub = False
    hub_private_repo = False
    overwrite_output_dir = True
    seed = 0
