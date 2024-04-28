import torch
from compel import Compel
from diffusers import StableDiffusionPipeline
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler)
from hidiffusion import apply_hidiffusion, remove_hidiffusion


SCHEDULERS = [
    "DPM++ SDE"
    "DPM++ SDE Karras"
    "DPM++ 2M",
    "DPM++ 2M Karras",
    "Euler",
    "Euler Karras"
    "Euler a",
    "Euler a Karras"
]


class PipelineManager():

    pipeline: StableDiffusionPipeline
    scheduler: str
    device: str
    model_name: str
    compel: Compel
    hidiffusion_applied: bool

    def load_pipeline(self, model_path, scheduler, device):
        self.pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=model_path,
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.set_scheduler(scheduler)
        self.move_pipeline(device)
        self.model_name = model_path.split("\\")[-1].split(".")[0]
        self.compel = Compel(
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer)
        self.hidiffusion_applied = False

    def move_pipeline(self, device):
        self.pipeline.to(device)
        self.device = device

    def remove_pipeline(self):
        self.pipeline = None
        self.model_name = None
        torch.cuda.empty_cache()

    def apply_hidiffusion(self):
        apply_hidiffusion(self.pipeline)
        self.hidiffusion_applied = True

    def remove_hidiffusion(self):
        remove_hidiffusion(self.pipeline)
        self.hidiffusion_applied = False

    def update_state(self, model_path, apply_hidiffusion, scheduler):
        model_name = model_path.split("\\")[-1].split(".")[0]
        if self.model_name != None:
            if self.model_name != model_name:
                self.remove_pipeline()
                self.load_pipeline(model_path, self.device)

        if scheduler != self.scheduler:
            self.set_scheduler(scheduler)

        if apply_hidiffusion:
            if not self.hidiffusion_applied:
                self.apply_hidiffusion()
        else:
            if self.hidiffusion_applied:
                self.remove_hidiffusion()

    def set_scheduler(self, scheduler):
        if scheduler == "DPM++ SDE":
            self.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                self.pipeline.scheduler.config)
        if scheduler == "DPM++ SDE Karras":
            self.pipeline.scheduler = DPMSolverSinglestepScheduler.from_config(
                self.pipeline.scheduler.config, use_karras_sigmas=True)
        if scheduler == "DPM++ 2M":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config)
        if scheduler == "DPM++ 2M Karras":
            self.pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipeline.scheduler.config, use_karras_sigmas=True)
        if scheduler == "Euler":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        if scheduler == "Euler Karras":
            self.pipeline.scheduler = EulerDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, use_karras_sigmas=True)
        if scheduler == "Euler a":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config)
        if scheduler == "Euler a Karras":
            self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self.pipeline.scheduler.config, use_karras_sigmas=True)

        self.scheduler = scheduler

    def queue_pipeline(self, seed, width, height, steps, guidance_scale, positive_prompt, negative_prompt):
        positive_embeds = self.compel(positive_prompt)
        negative_embeds = self.compel(negative_prompt)

        return self.pipeline(
            generator=torch.Generator("cpu").manual_seed(seed),
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            positive_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds
        ).images[0]
