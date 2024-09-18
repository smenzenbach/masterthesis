from email.mime import image
import os
import sys
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
import base64
import requests

# Add the project root directory to sys.path
project_root = os.path.abspath("../..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import sort_csv_by_column, create_label_from_output


# OpenAI API Key
api_key = "sk-proj-OGJKcZVx1iC24oK0dphFT3BlbkFJMCLgutIeFQMWQRMUZOXM"


# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def _gpt_request(image_path, prompt):

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }

    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    )
    return response


def run_instance_gpt4(img_dict, prompt, output_for_true):
    """
    Run a single instance of the GPT-4 model on an image.

    Parameters
    ----------
    img_dict : dict
        A dictionary containing the image metadata. It should have keys "file_path" and "file_name".
    prompt : str
        The prompt to send to the GPT-4 model.
    output_for_true : list
        A list of outputs that correspond to a label of 1 (confounder is present).


    Returns
    -------
    output : str
        The output from the GPT-4 model.
    label : int
        The predicted label (1 for confounder and 0 for no confounder).
    """
    image_path = img_dict["file_path"]
    image_file_name = img_dict["file_name"]

    response = _gpt_request(image_path, prompt)

    try:
        response_json = response.json()
        output = response_json["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error in response for image {image_file_name}.")
        # output =  error message
        output = f"error {e}"

    # Find predicted label
    label = create_label_from_output(output, output_for_true)

    return output, label


def build_pred_csv_gpt4(
    csv_path,
    images,
    prompt,
    output_for_true,
    save_interval=100,
):
    """
    Generate a csv file with the image file names and their corresponding labels (1 for confounder and 0 for no confounder).


    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    images: list of dict
        A list of dictionaries containing the image objects and their metadata.
    user_input_list : list
        A list of user input questions given to the sgl function.
    output_for_true : list
        A list of outputs that correspond to a label of 1 (confounder is present).
    save_interval : int, optional
        Save interval of images for intermediate results, by default 100.


    Returns
    -------
    None
    """

    data = []
    num_errors = 0

    for i, img_dict in enumerate(tqdm(images)):
        image_file_name = img_dict["file_name"]

        output, label = run_instance_gpt4(img_dict, prompt, output_for_true)

        # see if output is an error
        if "error" in output.lower():
            num_errors += 1
            print(f"Already encountered {num_errors} errors.")

        data.append([image_file_name, label, output])

        # Save intermediate results every `save_interval` images
        if (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(
                data, columns=["image_file_name", "label", "outputs"]
            )
            temp_df.to_csv(csv_path, index=False)
            print(f"Saved {i + 1} images to {csv_path}")

    # Final save
    df = pd.DataFrame(data, columns=["image_file_name", "label", "outputs"])
    df.to_csv(csv_path, index=False)
    # Sort the CSV file by the first column (image_file_name)
    sort_csv_by_column(csv_path, column_index=0, header=True)
    print(f"Final data saved to {csv_path}")
