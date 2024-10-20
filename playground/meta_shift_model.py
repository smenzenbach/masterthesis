from cProfile import label
import os
import sys
from PIL import Image
import torchvision.models as models
from torchvision.models.vgg import VGG16_Weights
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from torch import nn
import torch
import numpy as np


# Add the project root directory to sys.path
project_root = os.path.abspath("..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import get_images_from_directory


def VGG16_pretrained(freeze_features=False):
    """
    Loads the VGG16 model pretrained on ImageNet.
    Replaces the last layer of the classifier to output 2 classes instead of 1000.

    From pytorch:
        Expect input images normalized in the same way, i.e. mini-batches of 3-channel RGB images
        of shape (3 x H x W), where H and W are expected to be at least 224. The images have
        to be loaded in to a range of [0,1] and then normalized using
        mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].

    Args:
        freeze_features: if True then freezes all feature layers. Used to train model only
            on classifier i.e. transfer learning.
    """
    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False
    model.classifier[-1] = nn.Linear(4096, 2)
    return model


def load_images_from_dict(image_dicts):
    images = []
    for image_dict in image_dicts:
        img = Image.open(image_dict["file_path"]).convert("RGB")
        images.append(img)
    return images


def images_to_tensors(images):
    tensors = []
    # vgg16 preprpcessing
    transform = VGG16_Weights.IMAGENET1K_V1.transforms()
    for img in images:
        tensor = transform(img)
        tensors.append(tensor)
    return torch.stack(tensors)


def create_tensor_dataset(images_path, label):
    image_dicts = get_images_from_directory(images_path)  # Load the image dicts
    images = load_images_from_dict(image_dicts)  # Load the images
    image_tensors = images_to_tensors(images)  # Convert images to tensors
    labels = torch.full((len(image_tensors),), label)  # Create labels
    return TensorDataset(image_tensors, labels)  # Create a TensorDataset


def create_train_test_dataset(dataset, train_size):
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(0),
    )
    return train_dataset, test_dataset


def get_meta_shift_dataset(data_path, class1, class2, word1, word2, batch_size=64):
    c1w1 = f"{data_path}{class1}({word1})"
    c2w1 = f"{data_path}{class2}({word1})"
    c1w2 = f"{data_path}{class1}({word2})"
    c2w2 = f"{data_path}{class2}({word2})"

    label_c2n_dict = {class1: 0, class2: 1}
    label_n2c_dict = {0: class1, 1: class2}

    c1w1_dataset = create_tensor_dataset(c1w1, label=label_c2n_dict[class1])
    c2w1_dataset = create_tensor_dataset(c2w1, label=label_c2n_dict[class2])
    c1w2_dataset = create_tensor_dataset(c1w2, label=label_c2n_dict[class1])
    c2w2_dataset = create_tensor_dataset(c2w2, label=label_c2n_dict[class2])

    # compute train ratios
    train_ratio = 0.8
    w_ratio = 0.9
    nw_ratio = 1 - w_ratio
    c1w1_len = len(c1w1_dataset)
    c1w2_len = len(c1w2_dataset)
    c2w1_len = len(c2w1_dataset)
    c2w2_len = len(c2w2_dataset)
    c1w1_train_size = np.floor(c1w1_len * train_ratio * w_ratio)
    c1w2_train_size = np.floor(c1w2_len * train_ratio * nw_ratio)
    c2w1_train_size = np.floor(c2w1_len * train_ratio * nw_ratio)
    c2w2_train_size = np.floor(c2w2_len * train_ratio * w_ratio)

    c1w1_train, c1w1_test = create_train_test_dataset(
        c1w1_dataset, train_size=c1w1_train_size
    )
    c2w1_train, c2w1_test = create_train_test_dataset(
        c2w1_dataset, train_size=c2w1_train_size
    )
    c1w2_train, c1w2_test = create_train_test_dataset(
        c1w2_dataset, train_size=c1w2_train_size
    )
    c2w2_train, c2w2_test = create_train_test_dataset(
        c2w2_dataset, train_size=c2w2_train_size
    )

    train_dataset = ConcatDataset([c1w1_train, c2w1_train, c1w2_train, c2w2_train])
    test_dataset = ConcatDataset([c1w1_test, c2w1_test, c1w2_test, c2w2_test])

    # weighted loss
    c1_ratio = (len(c1w1_train) + len(c1w2_train)) / len(train_dataset)
    c2_ratio = 1 - c1_ratio
    c1_weight = 1 / c1_ratio
    c2_weight = 1 / c2_ratio
    weights = torch.tensor([c1_weight, c2_weight])
    weights /= weights.sum()

    dataloaders = {}
    dataloaders["train"] = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    dataloaders["test"] = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return dataloaders, weights, label_n2c_dict


def train(model, loss_fn, optimizer, dataloaders, device, num_epochs):
    best_model = model
    best_acc = 0.0
    for epoch in range(num_epochs):
        print_string = f"Epoch: {epoch+1}"
        for phase in ["train", "test"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = loss_fn(outputs, labels)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item()  # * inputs.size(0)
                running_corrects += torch.sum(
                    torch.argmax(outputs, 1) == labels.data
                ).item()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            if phase == "test" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = model

            print_string += f" | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}"

        print(print_string)
    return best_model


if __name__ == "__main__":
    data_path = "/workspace/home/data/MetaShift/raw/"
    class1 = "man"
    class2 = "woman"
    word1 = "bottle"
    word2 = "umbrella"

    print("Loading data")
    dataloaders, weights, label_dict = get_meta_shift_dataset(
        data_path, class1, class2, word1, word2, batch_size=64
    )

    print("Label", label_dict)
    print("Weights", weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VGG16_pretrained(freeze_features=True).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weights).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training model")
    model = train(model, loss_fn, optimizer, dataloaders, device, num_epochs=20)
    # torch.save(model.state_dict(), "model.pth")
    # print("Model saved to model.pth")
