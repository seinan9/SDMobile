import torch
from compel import Compel
from diffusers import StableDiffusionPipeline
from diffusers import (
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler)
from hidiffusion import apply_hidiffusion, remove_hidiffusion

from backend.config import Config
from backend.utils import (
    convert_to_model_name,
    convert_to_model_path,
    join_paths,
    list_dir)


class PipelineManager():

    SCHEDULERS = [
        "DPM++ SDE",
        "DPM++ SDE Karras",
        "DPM++ 2M",
        "DPM++ 2M Karras",
        "Euler",
        "Euler Karras",
        "Euler a",
        "Euler a Karras"
    ]

    available_models: list
    model_name: str
    pipeline: StableDiffusionPipeline
    scheduler: str
    compel: Compel
    hidiffusion_applied: bool

    def __init__(self):
        self.fetch_available_models()
        self.model_name = None
        self.pipeline = None
        self.scheduler = None
        self.compel = None
        self.hidiffusion_applied = False

    def load_pipeline(self, model_name):
        self.model_name = model_name
        self.pipeline = StableDiffusionPipeline.from_single_file(
            pretrained_model_link_or_path=convert_to_model_path(Config.HOME.value, model_name),
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.compel = Compel(
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer)
        self.hidiffusion_applied = False
        self.move_pipeline(Config.DEVICE.value)

    def move_pipeline(self, device):
        self.pipeline.to(device)

    def remove_pipeline(self):
        self.pipeline = None
        self.model_name = None
        torch.cuda.empty_cache()

    def update_pipeline(self, model_name, scheduler, use_hidiffusion):
        if self.model_name != model_name:
            if self.model_name != None:
                self.remove_pipeline()
            self.load_pipeline(model_name)

        if self.scheduler != scheduler:
            self.set_scheduler(scheduler)

        if use_hidiffusion:
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

    def fetch_available_models(self):
        model_dir = join_paths(Config.HOME.value, "models")
        self.available_models = [convert_to_model_name(
            model_path) for model_path in list_dir(model_dir)]

    def apply_hidiffusion(self):
        apply_hidiffusion(self.pipeline)
        self.hidiffusion_applied = True

    def remove_hidiffusion(self):
        remove_hidiffusion(self.pipeline)
        self.hidiffusion_applied = False

    def queue_pipeline(self, seed, width, height, steps, guidance_scale, positive_prompt, negative_prompt):
        positive_embeds = self.compel(positive_prompt)
        negative_embeds = self.compel(negative_prompt)

        return self.pipeline(
            generator=torch.Generator("cpu").manual_seed(seed),
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            prompt_embeds=positive_embeds,
            negative_prompt_embeds=negative_embeds
        ).images[0]
