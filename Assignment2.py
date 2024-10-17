import streamlit as st
from diffusers import AutoPipelineForText2Image, LCMScheduler
import torch
from PIL import Image

# Initialize the pipeline
model = 'lykon/dreamshaper-8-lcm'
pipe = AutoPipelineForText2Image.from_pretrained(model, torch_dtype=torch.float16)
pipe.to("cuda")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

# Create the Streamlit app
def main():
    st.title("Dreamshaper Image Generator")
    st.write("Enter a text prompt and/or upload an image to generate a new image.")

    # Get the user input
    prompt = st.text_input("Prompt")
    uploaded_image = st.file_uploader("Upload an image", accept_multiple_files=False)

    # Generate the image
    if prompt:
        with st.spinner("Generating..."):
            # Check if an image was uploaded
            if uploaded_image:
                # Process the uploaded image
                image = Image.open(uploaded_image)
                canny_image = convert_to_canny(image)
                # Generate the image using the uploaded image and the prompt
                image = pipe(prompt, image=canny_image, num_inference_steps=8, guidance_scale=1.5).images[0]
            else:
                # Generate the image using only the prompt
                image = pipe(prompt, num_inference_steps=8, guidance_scale=1.5).images[0]
            # Display the generated image
            st.image(image, caption="Generated Image", use_column_width=True)

# Function to convert an image to Canny format (if needed)
def convert_to_canny(image):
    # Convert the image to grayscale
    gray_image = image.convert("L")
    # Apply the Canny edge detection
    canny_image = np.array(gray_image)
    canny_image = cv2.Canny(canny_image, 100, 200)
    canny_image = canny_image[:, :, None]
    canny_image = np.concatenate([canny_image, canny_image, canny_image], axis=2)
    canny_image = Image.fromarray(canny_image)
    return canny_image

if __name__ == "__main__":
    main()