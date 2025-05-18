import json
import os
import textwrap
import random


def split_dataset(input_file, train_ratio=0.8, test_ratio=0.1, val_ratio=0.1):
    """
    Splits the dataset into training, test and validation sets.
    """
    with open(input_file, 'r') as f:
        data = json.load(f)
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    test_size = int(total_size * test_ratio)
    val_size = int(total_size * val_ratio)
    # Ensure the sizes add up to total_size
    if train_size + test_size + val_size != total_size:
        raise ValueError(
            "The sum of train, test and validation sizes does not equal the total size of the dataset.")
    # Shuffle the data
    data = data[:train_size + test_size + val_size]
    random.shuffle(data)
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]

    return train_data, test_data, val_data


def format_data_gemma(jsonElement, prompt):
    """
    Formats the data for the Gemma model.
    """
    user_prompt = prompt + jsonElement["english_translation"]
    user_prompt = textwrap.dedent(user_prompt)

    messages = [
        {
            "role": "user",
            "content": user_prompt
        },
        {
            "role": "assistant",
            "content": jsonElement["gorgias_code"]
        }
    ]
    return {"messages": messages}


def save_split_data(train_data, test_data, val_data, output_dir):
    """
    Saves the split data into separate JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, 'train.json'), 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(os.path.join(output_dir, 'test.json'), 'w') as f:
        json.dump(test_data, f, indent=4)
    with open(os.path.join(output_dir, 'val.json'), 'w') as f:
        json.dump(val_data, f, indent=4)


if __name__ == "__main__":
    # set seed
    random.seed(42)

    input_file = "data/dataset_10k.json"
    output_dir = "data/split_formatted_10k"
    train_data, test_data, val_data = split_dataset(input_file)
    with open("data/prompt_gemma_template.txt", "r") as f:
        prompt = f.read()
        for i in range(len(train_data)):
            train_data[i] = format_data_gemma(train_data[i], prompt)
        for i in range(len(test_data)):
            test_data[i] = format_data_gemma(test_data[i], prompt)
        for i in range(len(val_data)):
            val_data[i] = format_data_gemma(val_data[i], prompt)
    save_split_data(train_data, test_data, val_data, output_dir)
    print(f"Data split and saved to {output_dir}")
