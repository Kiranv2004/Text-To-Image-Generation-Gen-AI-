from diffusers import StableDiffusionPipeline
import torch

# Check if GPU is available and set the device accordingly
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the pre-trained Stable Diffusion pipeline
if device == "cuda":
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16  # Use mixed precision for GPU
    )
else:
    pipeline = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")

pipeline = pipeline.to(device)

# Enable optimizations
if device == "cuda":
    pipeline.enable_xformers_memory_efficient_attention()  # Faster attention computation

# Enable attention slicing for reduced memory usage
pipeline.enable_attention_slicing()

# Function to generate an image from text
def generate_image(prompt, steps=25, width=512, height=512):
    """
    Generate an image based on the provided text prompt.

    Args:
        prompt (str): The text prompt for image generation.
        steps (int): Number of denoising steps (fewer steps are faster).
        width (int): Width of the output image.
        height (int): Height of the output image.

    Returns:
        None
    """
    print(f"Generating image for prompt: '{prompt}'")
    print(f"Settings - Steps: {steps}, Width: {width}, Height: {height}")

    # Extract the first word as the image name
    image_name = prompt.split()[0].lower() + ".png"

    # Generate the image
    if device == "cuda":
        with torch.autocast(device):  # Use mixed precision for GPU
            image = pipeline(prompt, num_inference_steps=steps, width=width, height=height).images[0]
    else:
        image = pipeline(prompt, num_inference_steps=steps, width=width, height=height).images[0]  # Standard precision for CPU

    # Save the generated image
    image.save(image_name)
    print(f"Image saved at: {image_name}")

# Example usage
if __name__ == "__main__":
    # Take user input for the text prompt
    text_prompt = input("Enter a text prompt for image generation: ")

    # Call the function with reduced steps for faster generation
    generate_image(text_prompt, steps=20, width=512, height=512)
