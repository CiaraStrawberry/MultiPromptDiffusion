import torch
import os
from diffusers import StableDiffusionPipeline, DDIMScheduler
from PIL import Image, ImageDraw, ImageFont
import datetime

from ip_adapter import MultiPromptAdapter  # Make sure this is your correct import

###############################################################################
# 1. Define helper: image_grid()
###############################################################################
def image_grid(imgs, rows, cols):
    """
    Create a single grid image from a list of PIL images.

    :param imgs: list of PIL images
    :param rows: number of rows in the grid
    :param cols: number of columns in the grid
    :return: a PIL image arranged in a grid
    """
    assert len(imgs) == rows*cols, (
        f"Number of images ({len(imgs)}) does not match rows*cols ({rows*cols})"
    )

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, img in enumerate(imgs):
        grid.paste(img, box=((i % cols) * w, (i // cols) * h))
    return grid

###############################################################################
# 2. (Optional) helper to add row/column labels to a grid
###############################################################################
def add_labels_to_grid(
    grid_image, 
    row_labels, 
    col_labels, 
    cell_width, 
    cell_height,
    left_margin=220,     # space for row labels
    top_margin=60,       # space for column labels
    font_size=18,
):
    """
    Creates a new image with `grid_image` in the bottom-right area,
    and writes row_labels on the left side, and col_labels at the top.
    
    :param grid_image: the already-created grid (PIL Image)
    :param row_labels: list of row label strings (e.g. checkpoint names)
    :param col_labels: list of column label strings (e.g. short prompt text)
    :param cell_width: width of each cell in the grid
    :param cell_height: height of each cell in the grid
    :param left_margin: extra width (in pixels) on the left for row labels
    :param top_margin: extra height (in pixels) on top for column labels
    :param font_size: font size for the labels
    :return: new PIL image with text labels + the pasted grid
    """

    rows = len(row_labels)
    cols = len(col_labels)

    # Calculate new canvas size
    new_width = grid_image.width + left_margin
    new_height = grid_image.height + top_margin

    # Create a white background
    labeled_img = Image.new('RGB', (new_width, new_height), color='white')
    draw = ImageDraw.Draw(labeled_img)

    # Paste the original grid into the new labeled image
    labeled_img.paste(grid_image, (left_margin, top_margin))

    # Optionally load a font (or let PIL use default)
    # If you have a TTF font file, you can do:
    # font = ImageFont.truetype("arial.ttf", font_size)
    # For no custom font, just use ImageFont.load_default():
    font = ImageFont.load_default()

    # Draw column labels (top)
    for col_index, col_text in enumerate(col_labels):
        # Approximate X-center of each column
        x_pos = left_margin + col_index * cell_width + cell_width // 2
        y_pos = top_margin // 2  # halfway in the top margin
        draw.text((x_pos, y_pos), col_text, fill="black", anchor="mm", font=font)

    # Draw row labels (left side)
    for row_index, row_text in enumerate(row_labels):
        # Approximate Y-center of each row
        x_pos = left_margin // 2
        y_pos = top_margin + row_index * cell_height + cell_height // 2
        draw.text((x_pos, y_pos), row_text, fill="black", anchor="mm", font=font)

    return labeled_img

###############################################################################
# 3. Prepare the base SD pipeline
###############################################################################
#base_model_path = "runwayml/stable-diffusion-v1-5"
base_model_path = "stablediffusionapi/realistic-vision-v51"

# Example custom scheduler (optional)
noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

pipe = StableDiffusionPipeline.from_pretrained(
    base_model_path,
    scheduler=noise_scheduler,
    feature_extractor=None,
    safety_checker=None,
)

# Enable CPU offload or GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.enable_model_cpu_offload()

###############################################################################
# 4. Collect checkpoints and define prompt sets
###############################################################################
ckpt_folder = "models"
# Gather every .bin file in the folder
all_ckpts = [
    f for f in os.listdir(ckpt_folder) 
    if f.endswith(".bin")
]
#all_ckpts[0] = ""

# Define each column of the grid as a *set* of prompts.
prompt_sets = [
    {
        "prompt":   "a cow in a garden",
        "prompt_1": "the garden has roses",
        "prompt_2": "there's a white gazebo",
        "prompt_3": "the butterfly is blue",
        "prompt_4": "the butterfly is near some flowers",
        
    },
    {
        "prompt":   "a picture of a boat on the sea",
        "prompt_1": "the boat is a sailboat",
        "prompt_2": "cloudy",
    },
    {
        "prompt":   "a car driving on the road",
        "prompt_1": "colorful stage lights",
        "prompt_2": "enthusiastic audience",
    },
    {
        "prompt":   "chef in restaurant",
        "prompt_1": "cooking in professional kitchen",
        "prompt_2": "wearing chef's uniform",
        "prompt_3": "using professional equipment",
        "prompt_4": "decorating a birthday cake",
    },
]
###############################################################################
# 5. Generate images for each (ckpt, prompt_set) pair
###############################################################################
all_images = []
num_inference_steps = 30
scale = 1.0
seed = 991423
# We will track which checkpoint row we are in, so
# we can label them properly later.
row_labels = []
col_labels = []

# Create a short label for each column, e.g., from "prompt"
# or some short substring if you prefer
for pset in prompt_sets:
    prompts_for_label = []
    for k in sorted(pset.keys()):  # sort so prompt < prompt_1 < prompt_2 ...
        if k.startswith("prompt"):
            prompts_for_label.append(pset[k])
    # Join them with newlines
    col_labels.append("\n".join(prompts_for_label))

first = True
for ckpt in all_ckpts:
    if first:
        used_scale = 0
        first = False
    else:
        used_scale = scale
    # Load the current checkpoint into the MultiPromptAdapter
    ip_ckpt_path = os.path.join(ckpt_folder, ckpt)
    print(f"\nLoading checkpoint: {ip_ckpt_path}")
    row_labels.append(ckpt)  # We'll just use the filename as the row label

    ip_model = MultiPromptAdapter(
        pipe,
        ip_ckpt_path,
        device=device,
    )

    # Generate one image per prompt set for this checkpoint
    for pset in prompt_sets:
        images = ip_model.generate(
            prompt=pset.get("prompt", None),
            prompt_1=pset.get("prompt_1", None),
            prompt_2=pset.get("prompt_2", None),
            prompt_3=pset.get("prompt_3", None),
            num_samples=1,
            num_inference_steps=num_inference_steps,
            seed=seed,
            scale=scale,
        )
        all_images.append(images[0])

###############################################################################
# 6. Create the final (unlabeled) grid
###############################################################################
rows = len(all_ckpts)  # one row per checkpoint
cols = len(prompt_sets)  # one column per prompt set
grid = image_grid(all_images, rows, cols)

# Dimensions of a single cell (use the size of the first image)
cell_width, cell_height = all_images[0].size

###############################################################################
# 7. Add text labels for each row/column
###############################################################################
labeled_grid = add_labels_to_grid(
    grid,
    row_labels=row_labels,
    col_labels=col_labels,
    cell_width=cell_width,
    cell_height=cell_height,
    left_margin=220,  # Adjust if you need more/less space for row labels
    top_margin=60,    # Adjust if you need more/less space for column labels
    font_size=18,
)

###############################################################################
# 8. Save the labeled grid
###############################################################################
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
save_folder = "grid_test"
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

save_path = f"{save_folder}/grid_{timestamp}.png"
labeled_grid.save(save_path)
print(f"\nSaved labeled grid to: {save_path}")
