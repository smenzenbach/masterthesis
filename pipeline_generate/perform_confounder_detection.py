import sglang as sgl
import csv


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


def build_pred_csv_sgl(
    csv_path,
    images,
    user_input_list,
    output_for_true,
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
    images: output of load_images_and_combine
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

    with open(csv_path, mode="w") as csv_file:
        writer = csv.writer(csv_file)

        # Write header
        writer.writerow(["image_file_name", "label", "outputs"])

        for img_dict in images:
            image_file_path = img_dict["meta"]["file_path"]
            image_file_name = img_dict["meta"]["file_name"]

            # Generate answers for the user input questions
            state = _image_multi_turn.run(
                image_file_path, user_input_list, max_new_tokens=128, temperature=0
            )
            outputs = [state[f"output_{i+1}"] for i in range(len(user_input_list))]

            # Find predicted label
            label = create_label_from_output(outputs[-1], output_for_true)

            # Write to csv
            writer.writerow([image_file_name, label, outputs])
