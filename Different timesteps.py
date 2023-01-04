import torch
import time
from PIL import Image
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

# Define scheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)


# pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)


# Function to concatenate images
def get_concat_h(im1, im2):
    dst = Image.new('RGB', (im1.width + im2.width, im1.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (im1.width, 0))
    return dst


images = []

# Try different timesteps
for i in range(200, 1001, 200):
    # Set timesteps
    pipe.scheduler.set_timesteps(num_inference_steps=i)

    # Use GPU
    pipe = pipe.to("cuda")

    # Generate image
    # prompt = ["a photograph of an astronaut riding a horse"]
    # prompt = ["a photo of medieval knight crying in the rain"]
    prompt = ["Disneyland painted by Van Gogh"]

    start = time.process_time()
    start_2 = time.time()
    image = pipe(prompt).images[0]
    end = time.process_time()
    end_2 = time.time()

    # Print time
    print("Timesteps: " + str(i))
    print("Process time: " + str(end - start) + "s")
    print("Wall-clock time: " + str(end_2 - start_2) + "s")

    images.append(image)

image = get_concat_h(images[0], images[1])
image = get_concat_h(image, images[2])
image = get_concat_h(image, images[3])
image = get_concat_h(image, images[4])
image.show()
