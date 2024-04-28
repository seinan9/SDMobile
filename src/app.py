import gradio as gr

from backend.pipeline_manager import PipelineManager
from frontend.seafoam import Seafoam

js_func = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'light') {
        url.searchParams.set('__theme', 'light');
        window.location.href = url.href;
    }
}
"""

seafoam = Seafoam()
pipeline_manager = PipelineManager()
available_models = pipeline_manager.available_models
available_schedulers = pipeline_manager.SCHEDULERS


def start_pipeline(model_path, scheduler, use_hidiffusion, seed, width, height, steps, guidance_scale, positive_prompt, negative_prompt):
    pipeline_manager.update_pipeline(model_path, scheduler, use_hidiffusion)
    image = pipeline_manager.queue_pipeline(
        seed, width, height, steps, guidance_scale, positive_prompt, negative_prompt)
    return image


image = gr.Image()

with gr.Blocks(js=js_func, theme=seafoam) as demo:
    with gr.Tab("Txt2Img"):
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    positive_prompt = gr.Text(label="Positive Prompt", lines=3)
                    negative_prompt = gr.Text(label="Negative Prompt", lines=3)
                with gr.Column():
                    model = gr.Dropdown(label="Model", choices=available_models, value="dreamshaper")
                    use_hidiffusion = gr.Checkbox(label="Hidiffusion")
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            seed = gr.Number(label="Seed")
                            with gr.Row():
                                width = gr.Slider(
                                    label="Width", value=512, minimum=0, maximum=2048, step=64)
                                height = gr.Slider(
                                    label="Height", value=512, minimum=0, maximum=2048, step=64)
                with gr.Column():
                    with gr.Group():
                        with gr.Row():
                            scheduler = gr.Dropdown(
                                label="Scheduler", choices=available_schedulers, value="Euler")
                            with gr.Row():
                                steps = gr.Slider(
                                    label="Steps", value=25, minimum=0, maximum=100, step=1)
                                guidance_scale = gr.Slider(
                                    label="Classifier-free Guidance", value=7, minimum=1, maximum=20, step=0.5)

                queue = gr.Button("Queue")
                queue.click(start_pipeline, inputs=[
                            model, scheduler, use_hidiffusion, seed, width, height, steps, guidance_scale, positive_prompt, negative_prompt], outputs=image)
            image.render()

def launch():
    demo.launch()

demo.launch()