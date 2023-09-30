import numpy as np
import gradio as gr
import cv2
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
from mmengine.runner import load_checkpoint
import torch

register_all_modules()

#cfg = Config.fromfile('configs/controlnet/controlnet-1xb1-fill50k_lora_inference.py')
checkpoint_file = 'karameru_10000_revised.pth'
cfg = Config.fromfile('configs/dreambooth/dreambooth-lora-inference.py')
dreambooth_model = MODELS.build(cfg.model)
ckpt = load_checkpoint(dreambooth_model, checkpoint_file, map_location='cpu')

if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")

dreambooth_model.to(device)
dreambooth_model.eval()

def predict(prompt):
    #prompt = 'A karameru cat doing weight lifting in a gym'
    seed = 20
    samples = []
    for seed in range(12, 13):
        output_dict = dreambooth_model.infer(prompt, num_inference_steps=100, negative_prompt='colorful', seed=seed, num_images_per_prompt=2)
        samples.extend(output_dict['samples'])
    return samples

def show():
    imgs = ['IMG_5371.jpg', 'IMG_5381.jpg', 'IMG_5388.jpg', 'IMG_5408.jpg']
    return imgs


with gr.Blocks(title='Doodle Artist') as demo:
    gr.Markdown(
    """
    <h1 style="text-align: center;">Doodle Artist Helper</h1>
    """)
    with gr.Column(variant="panel"):
        with gr.Row():
            t_btn = gr.Button("Show Samples of Training Images", scale=0)
            t_gallery = gr.Gallery(
                label="Samples of Training Images", show_label=False, elem_id="gallery"
                , columns=[4], rows=[1], object_fit="contain", height="auto")
        with gr.Row():
            text = gr.Textbox(
                label="Enter your prompt",
                max_lines=1,
                value="A karameru cat dancing on the floor",
                container=False,
            )
            btn = gr.Button("Generate image", scale=0)

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        , columns=[2], rows=[1], object_fit="contain", height="auto")

    btn.click(predict, text, gallery)

    t_btn.click(show, None, t_gallery)

if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7860)
