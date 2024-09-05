import sglang as sgl
from sglang.lang.chat_template import get_chat_template
import csv
import multiprocessing as mp
import pandas as pd
from tqdm import tqdm
import numpy as np


@sgl.function
def _image_multi_turn(s, image_path, user_input_list):
    """

    Returns the generated answers to the user input questions.
    The user input questions are stored in a list. The number of items in the list determine the number QA turns.

    Parameters
    ----------
    image_path : str
        The path to the image.
    user_input_list : list
        A list of user input questions.

    Returns
    -------
    list
        A list of generated answers.
    """
    s += sgl.user(sgl.image(image_path))

    for i, user_input in enumerate(user_input_list):
        s += sgl.user(user_input)
        s += sgl.assistant(sgl.gen(f"output_{i+1}", max_tokens=256))


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for split in np.array_split(lst, n):
        yield split


def build_pred_csv_sgl(
    csv_path,
    images,
    user_input_list,
    output_for_true,
    num_batches=1,
    save_interval=1,
):
    """
    Generate a csv file with the image file names and their corresponding labels (1 for confounder and 0 for no confounder).

    The user input questions are stored in a list. The number of items in the list determine the number QA turns.

    ###Using sgl:
    Ensure that the sgl server is running before calling this function. The sgl server can be started by running the following command in the terminal:
    ```
    python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000
    ```
    The model-path, tokenizer-path, chat-template, and port can be changed to the desired values.
    Also set the runtime endpoint to the sgl server by adding the following line of code before calling this function:
    ```
    sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
    ```
    The port number should match the port number used to start the sgl server.


    Parameters
    ----------
    csv_path : str
        The path to save the csv file.
    images: output of load_images_from_directory
        A list of dictionaries containing the image objects and their metadata.
    user_input_list : list
        A list of user input questions given to the sgl function.
    output_for_true : list
        A list of outputs that correspond to a label of 1: confounder is present.

    Returns
    -------
    None
    """

    def create_label_from_output(output, output_for_true):
        output = output.lower().replace(" ", "")
        if output in output_for_true:
            return 1
        else:
            return 0

    data = []
    for batch_index, img_dicts in enumerate(
        tqdm(chunks(images, num_batches), total=num_batches)
    ):
        inp = [
            {"image_path": ip, "user_input_list": user_input_list}
            for ip in [img_dict["file_path"] for img_dict in img_dicts]
        ]
        # Generate answers for the user input questions
        states = _image_multi_turn.run_batch(inp, max_new_tokens=128, temperature=0)
        for idx, img_dict in enumerate(img_dicts):
            image_file_name = img_dict["file_name"]
            outputs = [
                states[idx][f"output_{i+1}"] for i in range(len(user_input_list))
            ]

            # Find predicted label
            label = create_label_from_output(outputs[-1], output_for_true)

            data.append([image_file_name, label, outputs])
            # Save intermediate results every `save_interval` batches

        if (batch_index + 1) % save_interval == 0 or batch_index == num_batches - 1:
            temp_df = pd.DataFrame(
                data, columns=["image_file_name", "label", "outputs"]
            )
            temp_df.to_csv(csv_path, index=False)
            print(f"Saved batch {batch_index + 1}/{num_batches} to {csv_path}")

    # Final save
    df = pd.DataFrame(data, columns=["image_file_name", "label", "outputs"])
    df.to_csv(csv_path, index=False)
    print(f"Final data saved to {csv_path}")


def build_pred_csv_sgl_with_backend(
    csv_path,
    images,
    user_input_list,
    output_for_true,
    num_batches=1,
    save_interval=1,
    model_path="liuhaotian/llava-v1.6-vicuna-7b",
    tokenizer_path="llava-hf/llava-1.5-7b-hf",
    chat_template="vicuna_v1.1",
):
    mp.set_start_method("spawn", force=True)
    runtime = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
    )
    runtime.endpoint.chat_template = get_chat_template(chat_template)
    sgl.set_default_backend(runtime)

    build_pred_csv_sgl(
        csv_path, images, user_input_list, output_for_true, num_batches, save_interval
    )

    # shutdown sgl server
    runtime.shutdown()


def merge_csv_files(output_filename, input_filenames):
    with open(output_filename, "w", newline="") as out_file:
        writer = csv.writer(out_file)
        header_written = False
        for filename in input_filenames:
            with open(filename, "r") as in_file:
                reader = csv.reader(in_file)
                header = next(reader)
                if not header_written:
                    writer.writerow(header)
                    header_written = True
                for row in reader:
                    writer.writerow(row)


# def build_pred_csv_sgl(
#     csv_path,
#     images,
#     user_input_list,
#     output_for_true,
#     start_img_idx=0,
# ):
#     """
#     Generate a csv file with the image file names and their corresponding labels (1 for confounder and 0 for no confounder).

#     The user input questions are stored in a list. The number of items in the list determine the number QA turns.

#     ###Using sgl:
#     Ensure that the sgl server is running before calling this function. The sgl server can be started by running the following command in the terminal:
#     ```
#     python3 -m sglang.launch_server --model-path liuhaotian/llava-v1.6-vicuna-7b --tokenizer-path llava-hf/llava-1.5-7b-hf --chat-template vicuna_v1.1 --port 30000
#     ```
#     The model-path, tokenizer-path, chat-template, and port can be changed to the desired values.
#     Also set the runtime endpoint to the sgl server by adding the following line of code before calling this function:
#     ```
#     sgl.set_default_backend(sgl.RuntimeEndpoint("http://localhost:30000"))
#     ```
#     The port number should match the port number used to start the sgl server.


#     Parameters
#     ----------
#     csv_path : str
#         The path to save the csv file.
#     images: output of load_images_from_directory
#         A list of dictionaries containing the image objects and their metadata.
#     user_input_list : list
#         A list of user input questions given to the sgl function.
#     output_for_true : list
#         A list of outputs that correspond to a label of 1: confounder is present.

#     Returns
#     -------
#     None
#     """

#     def create_label_from_output(output, output_for_true):
#         output = output.lower().replace(" ", "")
#         if output in output_for_true:
#             return 1
#         else:
#             return 0

#     with open(csv_path, mode="w") as csv_file:
#         writer = csv.writer(csv_file)

#         # Write header
#         writer.writerow(["image_file_name", "label", "outputs"])

#         for img_dict in images:
#             image_file_path = img_dict["file_path"]
#             image_file_name = img_dict["file_name"]

#             index = image_file_name.split("_")[1]
#             # remove the file extension
#             index = index.split(".")[0]
#             # remove the leading zeros
#             index = index.lstrip("0")
#             index = int(index)

#             # skip images before the start index
#             if index < start_img_idx:
#                 continue

#             # Generate answers for the user input questions
#             state = _image_multi_turn.run(
#                 image_file_path, user_input_list, max_new_tokens=128, temperature=0
#             )
#             outputs = [state[f"output_{i+1}"] for i in range(len(user_input_list))]

#             # Find predicted label
#             label = create_label_from_output(outputs[-1], output_for_true)

#             # Write to csv
#             writer.writerow([image_file_name, label, outputs])
