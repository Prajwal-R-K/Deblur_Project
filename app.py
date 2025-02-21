import gradio as gr
import os
import subprocess
import torch
from PIL import Image
import numpy as np

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Fix CUDA Out of Memory issue by enabling memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def gradio_interface(image):
    input_path = os.path.join(UPLOAD_FOLDER, "input.png")
    output_path = os.path.join(OUTPUT_FOLDER, "output.png")

    image.save(input_path)

    try:
        # Ensure CUDA memory is freed before running inference
        torch.cuda.empty_cache()

        # Run the NAFNet model with controlled memory usage
        command = [
            "python", "NAFNet/demo.py",
            "-opt", "NAFNet/options/test/REDS/NAFNet-width64.yml",
            "--input_path", input_path,
            "--output_path", output_path
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return f"Error: {result.stderr}"

        return Image.open(output_path)

    except Exception as e:
        return f"Exception: {str(e)}"

# Launch Gradio with public link
iface = gr.Interface(
    fn=gradio_interface,
    inputs=gr.Image(type="pil"),
    outputs=gr.Image(type="pil"),
    title="Image Restoration with NAFNet"
)

iface.launch(share=True)
