import lpips
import torch
import h5py
import csv
import yaml
import numpy as np
import subprocess

loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
loss_fn_vgg = lpips.LPIPS(
    net="vgg"
)  # closer to "traditional" perceptual loss, when used for optimization


def generate_data_dir_lpips(id_tag):
    datapath = (
        "/workspace/home/data/ISIC2019/h5_files/"  # "/workspace/datasets/ISIC19/"
    )
    inpainted_datapath = f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/h5_files/"
    random_inpainted_datapath = (
        "/workspace/home/pipeline_outputs/ISIC2019/random_inpaintings/h5_files/"
    )

    print("Loading data...")
    with h5py.File(datapath + "not_cancer_imgs.h5", "r") as hf:
        not_cancer_imgs = hf["not_cancer_imgs"][:]
    with h5py.File(datapath + "cancer_imgs.h5", "r") as hf:
        cancer_imgs = hf["cancer_imgs"][:]
    with h5py.File(inpainted_datapath + "not_cancer_imgs.h5", "r") as hf:
        inpainted_not_cancer_imgs = hf["not_cancer_imgs"][:]
    with h5py.File(inpainted_datapath + "cancer_imgs.h5", "r") as hf:
        inpainted_cancer_imgs = hf["cancer_imgs"][:]
    with h5py.File(random_inpainted_datapath + "not_cancer_imgs.h5", "r") as hf:
        random_not_cancer_imgs = hf["not_cancer_imgs"][:]
    with h5py.File(random_inpainted_datapath + "cancer_imgs.h5", "r") as hf:
        random_cancer_imgs = hf["cancer_imgs"][:]
    with h5py.File(datapath + "not_cancer_flags.h5", "r") as hf:
        not_cancer_flags = hf["not_cancer_flags"][:]

    # sort images by not_cancer_flags, for cancer images the flag is 0 (0: no patch, 1: patch)
    patch_imgs = not_cancer_imgs[not_cancer_flags == 1]
    no_patch_imgs = np.concatenate(
        (not_cancer_imgs[not_cancer_flags == 0], cancer_imgs)
    )
    patch_inpainted_imgs = inpainted_not_cancer_imgs[not_cancer_flags == 1]
    no_patch_inpainted_imgs = np.concatenate(
        (inpainted_not_cancer_imgs[not_cancer_flags == 0], inpainted_cancer_imgs)
    )
    patch_random_inpainted_imgs = random_not_cancer_imgs[not_cancer_flags == 1]
    no_patch_random_inpainted_imgs = np.concatenate(
        (random_not_cancer_imgs[not_cancer_flags == 0], random_cancer_imgs)
    )

    # transfor to torch
    patch_imgs = torch.tensor(patch_imgs).float()
    no_patch_imgs = torch.tensor(no_patch_imgs).float()
    patch_inpainted_imgs = torch.tensor(patch_inpainted_imgs).float()
    no_patch_inpainted_imgs = torch.tensor(no_patch_inpainted_imgs).float()
    patch_random_inpainted_imgs = torch.tensor(patch_random_inpainted_imgs).float()
    no_patch_random_inpainted_imgs = torch.tensor(
        no_patch_random_inpainted_imgs
    ).float()

    data_dir = {
        "patch_imgs": patch_imgs,
        "no_patch_imgs": no_patch_imgs,
        "patch_inpainted_imgs": patch_inpainted_imgs,
        "no_patch_inpainted_imgs": no_patch_inpainted_imgs,
    }
    return data_dir


def generate_data_dir_fid(id_tag):
    data_dir = {
        "patch_imgs": "/workspace/home/pipeline_outputs/ISIC2019/raw/patch",
        "no_patch_imgs": "/workspace/home/pipeline_outputs/ISIC2019/raw/no_patch",
        "patch_inpainted_imgs": f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/patch",
        "no_patch_inpainted_imgs": f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}/no_patch",
    }
    return data_dir


def calculate_lpips(imgs1, imgs2, loss_fn):
    # img0 = torch.zeros(1, 3, 64, 64)  # image should be RGB, IMPORTANT: normalized to [-1,1]
    # img1 = torch.zeros(1, 3, 64, 64)
    # d = loss_fn_alex(img0, img1)
    lpips_scores = []
    for img1, img2 in zip(imgs1, imgs2):
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        lpips_score = loss_fn(img1, img2)
        lpips_scores.append(lpips_score.item())
    return lpips_scores


def run_lpips(csv_path, data_dir, random_inpainted=False):
    loss_fn_alex = lpips.LPIPS(net="alex")  # best forward scores
    loss_fn_vgg = lpips.LPIPS(
        net="vgg"
    )  # closer to "traditional" perceptual loss, when used for optimization

    patch_imgs = data_dir["patch_imgs"]
    no_patch_imgs = data_dir["no_patch_imgs"]
    patch_inpainted_imgs = data_dir["patch_inpainted_imgs"]
    no_patch_inpainted_imgs = data_dir["no_patch_inpainted_imgs"]

    # Calculate LPIPS scores
    if not random_inpainted:
        print("Calculating LPIPS scores for inpainted images (Alex)...")
        lpips_scores_orig_inpainted_alex = calculate_lpips(
            patch_imgs, patch_inpainted_imgs, loss_fn_alex
        )

        print("Calculating LPIPS scores for inpainted images (VGG)...")
        lpips_scores_orig_inpainted_vgg = calculate_lpips(
            patch_imgs, patch_inpainted_imgs, loss_fn_vgg
        )

    else:
        print("Calculating LPIPS scores for random inpainted images (Alex)...")
        lpips_scores_orig_inpainted_alex = calculate_lpips(
            no_patch_imgs, no_patch_inpainted_imgs, loss_fn_alex
        )

        print("Calculating LPIPS scores for random inpainted images (VGG)...")
        lpips_scores_orig_inpainted_vgg = calculate_lpips(
            no_patch_imgs, no_patch_inpainted_imgs, loss_fn_vgg
        )

    # Save LPIPS scores to csv file
    print("Saving LPIPS scores to csv file...")
    with open(csv_path, "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "lpips_scores_orig_inpainted_alex",
                "lpips_scores_orig_inpainted_vgg",
            ]
        )
        for scores in zip(
            lpips_scores_orig_inpainted_alex,
            lpips_scores_orig_inpainted_vgg,
        ):
            writer.writerow(scores)


def save_lpips(csv_path, yaml_path):
    # read csv path
    with open(csv_path, mode="r") as csv_file:
        reader = csv.reader(csv_file)
        next(reader)  # skip header

        lpips_scores_orig_inpainted_alex = []
        lpips_scores_orig_inpainted_vgg = []

        for row in reader:
            lpips_scores_orig_inpainted_alex.append(float(row[0]))
            lpips_scores_orig_inpainted_vgg.append(float(row[1]))

    # Save average LPIPS scores to yaml file
    print("Saving average LPIPS scores to yaml file...")

    random_tag = "random" if "random" in csv_path else "patch"
    avg_lpips_scores = {
        f"lpips_scores_orig_inpainted_{random_tag}_alex": float(
            np.mean(lpips_scores_orig_inpainted_alex)
        ),
        f"lpips_scores_orig_inpainted_{random_tag}_vgg": float(
            np.mean(lpips_scores_orig_inpainted_vgg)
        ),
    }

    std_lpips_scores = {
        f"lpips_scores_orig_inpainted_{random_tag}_alex": float(
            np.std(lpips_scores_orig_inpainted_alex)
        ),
        f"lpips_scores_orig_inpainted_{random_tag}_vgg": float(
            np.std(lpips_scores_orig_inpainted_vgg)
        ),
    }

    median_lpips_scores = {
        f"lpips_scores_orig_inpainted_{random_tag}_alex": float(
            np.median(lpips_scores_orig_inpainted_alex)
        ),
        f"lpips_scores_orig_inpainted_{random_tag}_vgg": float(
            np.median(lpips_scores_orig_inpainted_vgg)
        ),
    }

    with open(yaml_path, "w") as file:
        yaml.dump(
            {
                "average": avg_lpips_scores,
                "std_dev": std_lpips_scores,
                "median": median_lpips_scores,
            },
            file,
        )


def calculate_fid(img_dir1, img_dir2):
    result = subprocess.run(
        ["python", "-m", "pytorch_fid", img_dir1, img_dir2],
        capture_output=True,
        text=True,
    )
    output = result.stdout
    print(output)
    print(result.stderr)
    fid_value = None
    for line in output.split("\n"):
        if "FID:" in line:
            fid_value = float(line.split()[-1])
            break
    return fid_value


def run_and_save_fid(yaml_path_fid, data_dir, random_inpainted=False):
    patch_imgs = data_dir["patch_imgs"]
    no_patch_imgs = data_dir["no_patch_imgs"]
    patch_inpainted_imgs = data_dir["patch_inpainted_imgs"]
    no_patch_inpainted_imgs = data_dir["no_patch_inpainted_imgs"]

    if not random_inpainted:
        print("Calculating FID scores for inpainted images...")
        fid_score_orig_inpainted_patch = calculate_fid(patch_imgs, patch_inpainted_imgs)

        fid_score_orig_np_inpainted_patch = calculate_fid(
            no_patch_imgs, patch_inpainted_imgs
        )

    else:
        print("Calculating FID scores for random inpainted images...")
        fid_score_orig_random_inpainted = calculate_fid(
            no_patch_imgs, no_patch_inpainted_imgs
        )

    # Save FID scores to yaml file
    print("Saving FID scores to yaml file...")
    with open(yaml_path_fid, "w") as file:
        if not random_inpainted:
            yaml.dump(
                {
                    "fid_score_orig_inpainted_patch": fid_score_orig_inpainted_patch,
                    "fid_score_orig_np_inpainted_patch": fid_score_orig_np_inpainted_patch,
                },
                file,
            )
        else:
            yaml.dump(
                {
                    "fid_score_orig_random_inpainted": fid_score_orig_random_inpainted,
                },
                file,
            )


if __name__ == "__main__":
    id_tags = ["random_inpaintings_sdxl"]

    for id_tag in id_tags:
        output_dir = f"/workspace/home/pipeline_outputs/ISIC2019/{id_tag}"

        csv_path_lpips = f"{output_dir}/lpips_scores.csv"
        yaml_path_lpips = f"{output_dir}/lpips_scores_avg.yaml"

        if "random" in id_tag:
            random_inpainted = True
        else:
            random_inpainted = False

        data_dir_lpips = generate_data_dir_lpips(id_tag)
        run_lpips(csv_path_lpips, data_dir_lpips, random_inpainted)
        save_lpips(csv_path_lpips, yaml_path_lpips)

        yaml_path_fid = f"{output_dir}/fid_scores.yaml"

        data_dir_fid = generate_data_dir_fid(id_tag)
        run_and_save_fid(yaml_path_fid, data_dir_fid, random_inpainted)
