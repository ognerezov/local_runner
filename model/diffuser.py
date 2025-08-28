import os
os.environ["HF_HOME"] = "/Volumes/LaCie/hf"
from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5")

prompt = "Chocolate labrador dog playing football, cold color palette, muted colors, detailed, 8k"
images = pipe(prompt).images

# -------------------------------------------------
# 4️⃣ Save the image(s) to disk
# -------------------------------------------------
output_dir = "/Volumes/LaCie/output/img/1"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(images):
    filename = os.path.join(output_dir, f"image_{i}.png")
    img.save(filename)          # or .save(filename, "JPEG") if you prefer
    print(f"Saved {filename}")