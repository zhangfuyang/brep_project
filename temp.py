import torch
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float32)

prompt = "a photograph of an astronaut riding a horse"

image = pipe(prompt).images[0]
