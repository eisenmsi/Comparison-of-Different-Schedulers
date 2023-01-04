import torch
import time

torch_device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

# 1. Load the autoencoder model which will be used to decode the latents into image space.
vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

# 2. Load the tokenizer and text encoder to tokenize and encode the text.
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# 3. The UNet model for generating the latents.
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

# Now instead of loading the pre-defined scheduler, we load a K-LMS scheduler instead.

from diffusers import LMSDiscreteScheduler

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear",
                                 num_train_timesteps=1000)

# Next we move the models to the GPU.
vae = vae.to(torch_device)
text_encoder = text_encoder.to(torch_device)
unet = unet.to(torch_device)

# We now define the parameters we'll use to generate images.
# prompt = ["a photograph of an astronaut riding a horse"]
# prompt = ["a photo of medieval knight crying in the rain"]
prompt = ["Disneyland painted by Van Gogh"]

height = 512  # default height of Stable Diffusion
width = 512  # default width of Stable Diffusion

num_inference_steps = 100  # Number of denoising steps

guidance_scale = 7.5  # Scale for classifier-free guidance

generator = torch.manual_seed(32)  # Seed generator to create the inital latent noise

batch_size = 1

# First, we get the text_embeddings for the prompt. These embeddings will be used to condition the UNet model.

start_all = time.process_time()
start_2_all = time.time()

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                       return_tensors="pt")

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Generate the intial random noise.

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)

scheduler.set_timesteps(num_inference_steps)

# The K-LMS scheduler needs to multiply the latents by its sigma values. Let's do this here
latents = latents * scheduler.init_noise_sigma

# We are ready to write the denoising loop.
from tqdm.auto import tqdm
from torch import autocast

start_denoising = time.process_time()
start_2_denoising = time.time()

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

end_denoising = time.process_time()
end_2_denoising = time.time()

# We now use the vae to decode the generated latents back into the image.
# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents

with torch.no_grad():
    image = vae.decode(latents).sample

end_all = time.process_time()
end_2_all = time.time()

# Print time
print("Process time denoising: " + str(end_denoising - start_denoising) + "s")
print("Wall-clock time denoising: " + str(end_2_denoising - start_2_denoising) + "s")
print("Process time all: " + str(end_all - start_all) + "s")
print("Wall-clock time all: " + str(end_2_all - start_2_all) + "s")

# And finally, let's convert the image to PIL so we can display or save it.
from PIL import Image

image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].show()
