import torch
import time
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, DDPMScheduler, DDIMScheduler, PNDMScheduler

pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)

# Define scheduler
pipe.scheduler = LMSDiscreteScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
# pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)

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
print("Process time: " + str(end - start) + "s")
print("Wall-clock time: " + str(end_2 - start_2) + "s")

image.show()
image.save(r'Different schedulers.png')
