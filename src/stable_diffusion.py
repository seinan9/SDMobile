import hidiffusion
import torch
from compel import Compel
from diffusers import (StableDiffusionPipeline,
                       EulerDiscreteScheduler,
                       EulerAncestralDiscreteScheduler,
                       DPMSolverSDEScheduler,
                       DPMSolverMultistepScheduler)


class StableDiffusion():

    def __init__(self, model_path, scheduler_name="euler"):
        self.pipeline = StableDiffusionPipeline.from_pretrained(model_path,
                                                                torch_dtype=torch.float16,
                                                                use_safetensors=True)
        self.set_scheduler(scheduler_name)
        self.compel = Compel(
            tokenizer=self.pipeline.tokenizer,
            text_encoder=self.pipeline.text_encoder)
        self.pipeline.to("cuda")

    def set_scheduler(self, scheduler_name):
        if scheduler_name == "euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        elif scheduler_name == "euler_a":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        elif scheduler_name == "dpm_sde":
            self.pipeline.scheduler = DPMSolverSDEScheduler.from_config(
                self.pipeline.scheduler.config)
        elif scheduler_name == "dpm_m":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config)

    def generate(self, positive, negative, seed=-1, steps=25, cfg=7, width=512, height=512):
        if seed == -1:
            seed = torch.randint(0, 9999999, (1,))[0].item()
        generator = torch.Generator("cuda").manual_seed(seed)

        pos_emb = self.compel(positive)
        neg_emb = self.compel(negative)

        return self.pipeline(prompt_embeds=pos_emb,
                             negative_prompt_embeds=neg_emb,
                             num_inference_steps=steps,
                             guidance_scale=cfg,
                             generator=generator,
                             width=width,
                             height=height).images[0], seed

    def apply_hidiffusion(self):
        hidiffusion.apply_hidiffusion(self.pipeline)

    def remove_hidiffusion(self):
        hidiffusion.remove_hidiffusion(self.pipeline)
