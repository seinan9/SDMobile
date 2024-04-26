from hidiffusion import apply_hidiffusion, remove_hidiffusion
import torch
from compel import Compel
from diffusers import (StableDiffusionImg2ImgPipeline,
                       EulerDiscreteScheduler,
                       EulerAncestralDiscreteScheduler,
                       DPMSolverSDEScheduler,
                       DPMSolverMultistepScheduler)
from PIL import Image

from utils import upscale_image

class StableDiffusion():

    def __init__(self, model_path):
        self.pipeline = StableDiffusionImg2ImgPipeline.from_single_file(model_path,torch_dtype=torch.float16,use_safetensors=True)
        self.compel = Compel(tokenizer=self.pipeline.tokenizer,
                             text_encoder=self.pipeline.text_encoder)
        self.pipeline.to("cuda")

    def generate(self, positive, negative, image = None, denoise = 1, seed=-1, steps=25, cfg=7, width=512, height=512):
        if seed == -1:
            seed = torch.randint(0, 9999999, (1,))[0].item()
        generator = torch.Generator("cpu").manual_seed(seed)

        pos_emb = self.compel(positive)
        neg_emb = self.compel(negative)

        if image == None:
            image = torch.randn(4, height // 8, width // 8, generator=generator)
        else:
            image = upscale_image(width, height)

        return self.pipeline(prompt_embeds=pos_emb,
                             negative_prompt_embeds=neg_emb,
                             image=image,
                             strength=denoise,
                             num_inference_steps=steps,
                             guidance_scale=cfg,
                             generator=generator,
                             width=width,
                             height=height).images[0], seed

    def set_scheduler(self, name):
        if name == "euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        elif name == "euler_a":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        elif name == "dpm_sde":
            self.pipeline.scheduler = DPMSolverSDEScheduler.from_config(
                self.pipeline.scheduler.config)
        elif name == "dpm_m":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config)

    def apply_hidi(self):
        apply_hidiffusion(self.pipeline)

    def remove_hidi(self):
        remove_hidiffusion(self.pipeline)
