from json import JSONDecodeError
import os
import sys
import pandas as pd
from tqdm import tqdm
import base64
import requests
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Add the project root directory to sys.path
project_root = os.path.abspath("../..")  # Adjust this path if needed
sys.path.append(project_root)

from utils import sort_csv_by_column, create_label_from_output, merge_csv_files


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

    num_tries = 3
    for t in range(num_tries):
        try:
            response = _gpt_request(image_path, prompt)
            response_json = response.json()
            output = response_json["choices"][0]["message"]["content"]
            break
        except KeyError as e:
            print(f"Error {e} in response for image {image_file_name}.")
            error = response_json["error"]["message"]
            print(f"Error message: {error}")
            if "Invalid image." in error:       # maybe due to prompt length
                print(image_path)
                output = f"error {error}"
                break
            if t < num_tries - 1:
                # get time to sleep: Rate limit reached for gpt-4o-mini in organization org-Bq1wEbsBxgeiOXBsN1zx2xEf on requests per day (RPD): Limit 10000, Used 10000, Requested 1. Please try again in 8.64s
                time_to_sleep = error.split("Please try again in ")[1].split("s")[0]
                # check if last char is "m" (if ms)
                if time_to_sleep[-1] == "m":
                    time_to_sleep = float(time_to_sleep[:-1]) / 1000
                else:
                    time_to_sleep = float(time_to_sleep)
                print(f"Sleeping for {time_to_sleep} seconds")
                time.sleep(time_to_sleep)
            print(f"Retrying image {image_file_name}...")
        except JSONDecodeError as e:
            print(
                f"Error {e} in response for image {image_file_name}. Try number {t + 1} of {num_tries}"
            )
            # output =  error message
            output = f"error {e}"

    # Find predicted label
    label = create_label_from_output(output, output_for_true)

    return output, label


def _process_images(
    images_chunk, prompt, output_for_true, temp_csv_path, save_interval
):
    """
    Process a chunk of images and save the result to a temporary CSV file.
    """
    data = []
    for i, img_dict in enumerate(tqdm(images_chunk)):
        image_file_name = img_dict["file_name"]

        output, label = run_instance_gpt4(img_dict, prompt, output_for_true)

        data.append([image_file_name, label, output])

        if (i + 1) % save_interval == 0:
            temp_df = pd.DataFrame(
                data, columns=["image_file_name", "label", "outputs"]
            )
            temp_df.to_csv(temp_csv_path, index=False)
            print(f"Saved {i + 1} images to {temp_csv_path}")

    # Final save for this chunk
    df = pd.DataFrame(data, columns=["image_file_name", "label", "outputs"])
    df.to_csv(temp_csv_path, index=False)
    return temp_csv_path


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

    _process_images(images, prompt, output_for_true, csv_path, save_interval)
    # Sort the CSV file by the first column (image_file_name)
    sort_csv_by_column(csv_path, column_index=0, header=True)
    print(f"Final data saved to {csv_path}")


def build_pred_csv_gpt4_parallel(
    csv_path, images, prompt, output_for_true, save_interval=100, num_workers=4
):
    """
    Parallelized version of build_pred_csv_gpt4 to process images in chunks and save results.
    """

    # Split the images into chunks for parallel processing
    chunk_size = len(images) // num_workers
    image_chunks = [
        images[i : i + chunk_size] for i in range(0, len(images), chunk_size)
    ]

    # Ensure the temp directory exists
    temp_dir = "temp_csvs"
    os.makedirs(temp_dir, exist_ok=True)

    temp_csv_paths = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for idx, chunk in enumerate(image_chunks):
            temp_csv_path = os.path.join(temp_dir, f"temp_{idx}.csv")
            temp_csv_paths.append(temp_csv_path)
            futures.append(
                executor.submit(
                    _process_images,
                    chunk,
                    prompt,
                    output_for_true,
                    temp_csv_path,
                    save_interval,
                )
            )

        # Collect results
        for future in as_completed(futures):
            future.result()

    # Merge all the temporary CSV files into the final one
    merge_csv_files(csv_path, temp_csv_paths, sort=True, delete_merged=True)
    print(f"Final data saved to {csv_path}")


def fix_errors_in_csv(csv_path, images, prompt, output_for_true):

    # Load the CSV file
    df = pd.read_csv(csv_path)

    # Find the rows with errors
    error_rows = df[df["outputs"].str.contains("error", case=False)]

    # Process the images with errors
    for i, row in tqdm(error_rows.iterrows()):
        image_file_name = row["image_file_name"]
        img_dict = next(
            (img for img in images if img["file_name"] == image_file_name), None
        )

        if img_dict is not None:
            output, label = run_instance_gpt4(img_dict, prompt, output_for_true)
            df.loc[i, "label"] = label
            df.loc[i, "outputs"] = output

    # Save the fixed CSV file
    df.to_csv(csv_path, index=False)
