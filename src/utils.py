from PIL import Image

def load_image(file):
    return Image.open(file)

def upscale_image(image, width, height):
    return image.resize((width, height), resample=Image.Resampling.LANCZOS)

def upscale_image_by(image, factor):
    return image.resize((factor * image.size[0], factor * image.size[1]), resample=Image.Resampling.LANCZOS)
