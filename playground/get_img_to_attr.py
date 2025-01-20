import os
import lpips
import torch
from PIL import Image, ImageFilter
from torchvision.transforms.functional import pil_to_tensor


ids = [33, 41, 59, 229]
path = "/workspace/home/A-Typology-for-Exploring-the-Mitigation-of-Shortcut-Behavior/output_images/raw/ISIC19-expl/Vanilla_ig"
img_base = "ISIC19-Vanilla-test-np-ig-"

loss_fn_alex = lpips.LPIPS(net="alex")

# get raw images with patch
patch_imgs_path = "/workspace/home/pipeline_outputs/ISIC2019/raw/patch"
patch_imgs = sorted(os.listdir(patch_imgs_path))

for id_tag in ids:
    print(f"Processing {id_tag}...")
    img_path = f"{path}/{img_base}{id_tag}.png"
    img = Image.open(img_path).convert("L").filter(ImageFilter.FIND_EDGES)
    img_tensor = pil_to_tensor(img).unsqueeze(0).float()

    lpips_scores = []

    for patch_img in patch_imgs:
        patch_img_path = f"{patch_imgs_path}/{patch_img}"
        patch_img = (
            Image.open(patch_img_path).convert("L").filter(ImageFilter.FIND_EDGES)
        )
        patch_img_tensor = pil_to_tensor(patch_img).unsqueeze(0).float()

        lpips_score = loss_fn_alex(img_tensor, patch_img_tensor)
        lpips_scores.append(lpips_score.item())

    # get img name of best 10 scores
    best_scores = sorted(lpips_scores)[:10]
    best_img_names = [patch_imgs[lpips_scores.index(score)] for score in best_scores]

    print(f"Best 10 scores for {id_tag}:\n{best_img_names}")


# Best 10 scores for 33:
