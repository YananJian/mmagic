import numpy as np
import gradio as gr
import cv2
import mmcv
from mmengine import Config
from PIL import Image

from mmagic.registry import MODELS
from mmagic.utils import register_all_modules
from mmengine.runner import load_checkpoint

register_all_modules()

#cfg = Config.fromfile('configs/controlnet/controlnet-1xb1-fill50k_lora_inference.py')
checkpoint_file = '/data/dreambooth-lora/iter_10000_revised.pth'
cfg = Config.fromfile('configs/dreambooth/dreambooth-lora-inference.py')
dreambooth_model = MODELS.build(cfg.model)
ckpt = load_checkpoint(dreambooth_model, checkpoint_file, map_location='cpu')

dreambooth_model.to('cuda:0')
dreambooth_model.eval()

def predict(prompt):
    #prompt = 'A karameru cat doing weight lifting in a gym'
    seed = 20
    samples = []
    for seed in range(10,11):
        output_dict = dreambooth_model.infer(prompt, num_inference_steps=80, negative_prompt='colorful', seed=seed, num_images_per_prompt=2)
        samples.extend(output_dict['samples'])
    return samples


with gr.Blocks() as demo:
    with gr.Column(variant="panel"):
        with gr.Row():
            text = gr.Textbox(
                label="Enter your prompt",
                max_lines=1,
                placeholder="A karameru cat ...",
                container=False,
            )
            btn = gr.Button("Generate image", scale=0)

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        , columns=[2], rows=[1], object_fit="contain", height="auto")

    btn.click(predict, text, gallery)



#demo = gr.Interface(sepia, gr.Image(shape=(200, 200)), "image")
'''
demo = gr.Interface(fn=predict,
             inputs=gr.Textbox(lines=2, placeholder="A karameru cat ..."),
             outputs=["image", "image"],
             examples=["A karameru cat doing weight lifting in a gym"])
gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        , columns=[2], rows=[2], object_fit="contain", height="auto")

'''
if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0', server_port=7006)
