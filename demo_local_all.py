
import torch
print(torch.__version__)
import os
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, DDIMScheduler, AutoencoderKL
from PIL import Image
import datetime

#from ip_adapter.ip_adapter_resample_input import IPAdapterPlusDual
from ip_adapter import MultiPromptAdapter
#from diffusers import DiffusionPipeline
from diffusers import StableDiffusion3Pipeline

base_model_path = "runwayml/stable-diffusion-v1-5"
#vae_model_path = "stabilityai/sd-vae-ft-mse"


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
#vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)

# load SD pipeline
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
#    scheduler=noise_scheduler,
#    vae=vae,
 #   torch_dtype=torch.float16,
    feature_extractor=None,
    safety_checker=None
)
pipe.enable_model_cpu_offload()

device = "cuda" if torch.cuda.is_available() else "cpu"
ip_ckpt= "models/MultiPrompt_15k.bin"
ip_model = MultiPromptAdapter(pipe, ip_ckpt, device,num_tokens=32 )

images = ip_model.generate(prompt="river",prompt_1="cat",prompt_2="a banana", num_samples=1, num_inference_steps=30, seed=45)

#images = pipe(
#    "A cat holding a sign that says hello world",
#    negative_prompt="",
#    num_inference_steps=28,
#    guidance_scale=7.0,
#).images

grid = image_grid(images, 1, 1)
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_folder = 'grid_test'
save_path = f"{save_folder}/grid_{timestamp}.png"

# Ensure the folder exists
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# Save the grid
print("Saving images grid...")
grid.save(save_path)
print(f"Images grid saved to {save_path}")