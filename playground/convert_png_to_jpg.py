from PIL import Image
import os

folder_with_png_images = (
    "/workspace/home/pipeline_outputs/ISIC2019/random_inpaintings_lama/inpainting"
)

# get all the png images in the folder
png_images = sorted(os.listdir(folder_with_png_images))

for i, png in enumerate(png_images):
    if png.endswith(".png"):
        # open the image
        img = Image.open(os.path.join(folder_with_png_images, png))
        # convert the image to jpg
        img = img.convert("RGB")
        # save the image as jpg
        img.save(
            os.path.join(folder_with_png_images, png.replace(".png", ".jpg")), "JPEG"
        )
        # remove the png image
        os.remove(os.path.join(folder_with_png_images, png))

    if i % 100 == 0:
        print(f"Converted {i} images")

print("All PNG images converted to JPG")
