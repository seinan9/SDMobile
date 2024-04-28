import os
import torch
from PIL import Image


def convert_to_model_name(model_path):
    return model_path.split(os.path.sep)[-1].split(".")[0]


def convert_to_model_path(home, model_name):
    return join_paths(home, "models", f"{model_name}.safetensors")


def join_paths(*paths: str) -> str:
    return os.path.join(*paths)


def list_dir(directory):
    return os.listdir(directory)


def load_image(file):
    return Image.open(file)


def upscale_image(image, width, height):
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)


def upscale_image_by(image, factor):
    return image.resize((factor * image.size[0], factor * image.size[1]), resample=Image.Resampling.LANCZOS)


def get_random_int(low, high):
    return torch.randint(low, high, (1,))[0].item()
