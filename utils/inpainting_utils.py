from PIL import Image
from utils import dilate_mask


def inpainting_preprocessing(
    image_path, mask_path, dilation_size=None, kernel_size=5, iterations=4
):
    img = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").point(lambda x: 255 if x > 128 else 0)
    if dilation_size is not None:
        mask = mask.resize(dilation_size)
    dilated_mask = dilate_mask(mask, kernel_size=kernel_size, iterations=iterations)
    dilated_mask = dilated_mask.resize(img.size)
    return img, dilated_mask
