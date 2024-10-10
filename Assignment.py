import streamlit as st
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import numpy as np
import cv2
from PIL import Image

# Load the control net and stable diffusion model
def load_models():
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

# Load the image and apply edge detection
def load_image_and_apply_edge_detection(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

# Generate an image using the Stable Diffusion model with a ControlNet
def generate_image(pipe, prompt, canny_image):
    generator = torch.manual_seed(0)
    image = pipe(
        prompt, num_inference_steps=20, generator=generator, image=canny_image
    ).images[0]
    return image

# Create the Streamlit app
def main():
    st.title("Stable Diffusion Image Generator")
    st.write("Enter a text prompt and upload an image to generate a new image.")

    # Load the models
    if "pipe" not in st.session_state:
        st.session_state["pipe"] = load_models()

    # Get the user input
    prompt = st.text_input("Prompt")
    uploaded_image = st.file_uploader("Upload an image")

    # Generate the image
    if prompt and uploaded_image:
        with st.spinner("Generating..."):
            image = Image.open(uploaded_image)
            canny_image = load_image_and_apply_edge_detection(image)
            image = generate_image(st.session_state["pipe"], prompt, canny_image)
            st.image(image, caption="Generated Image", use_column_width=True)

if __name__ == "__main__":
    main()