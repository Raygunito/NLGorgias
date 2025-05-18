import os
import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_data(train_path, valid_path, test_path):
    """
    Load the dataset from the specified paths.
    """
    data = load_dataset("json", data_files={
        "train": train_path,
        "validation": valid_path,
        "test": test_path
    })
    return data


def tokenize_fnc(example, max_length=1024):
    """
    Tokenize the input text using the provided tokenizer.
    """
    chat_text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False
    )

    tokenized = tokenizer(
        chat_text,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return {
        "input_ids": tokenized["input_ids"][0],
        "attention_mask": tokenized["attention_mask"][0]
    }


def prepare_model(model_id):
    """
    Prepare the model for training.
    """
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation='eager'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation='eager',
            torch_dtype=torch.bfloat16,
        )

    return model


def prepare_sft_config():
    """
    Prepare the SFT configuration.
    """
    sft_config = SFTConfig(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        max_steps=args.train_steps,
        warmup_steps=100,
        gradient_accumulation_steps=2,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        logging_steps=10,
        learning_rate=1e-4,
        optim="paged_adamw_8bit",
        bf16=True,
        dataset_text_field="input_ids",  # Optional if tokenizer uses standard field
    )
    return sft_config


def prepare_lora_config():
    """
    Prepare the LoRA configuration.
    """
    lora_config = LoraConfig(
        r=32,  # the rank (power of 2)
        lora_alpha=32,  # Scaling factor: balances adapter output vs. original model output
        target_modules=["q_proj", "v_proj"],
        bias="all",
        task_type=TaskType.CAUSAL_LM
    )
    return lora_config


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune a language model for GORGIAS <> English task.")
    parser.add_argument("output_dir", type=str,
                        help="Directory to save the fine-tuned model weights.")
    parser.add_argument("-tr_p", "--train_path", type=str,
                        default=None, help="Path to the training data.")
    parser.add_argument("-v_p", "--valid_path", type=str,
                        default=None, help="Path to the validation data.")
    parser.add_argument("-te_p", "--test_path", type=str,
                        default=None, help="Path to the test data.")
    parser.add_argument("--max_seq_length", type=int,
                        default=1024, help="Maximum sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int,
                        default=2, help="Batch size for training.")
    parser.add_argument('--train_steps', type=int,
                        default=800, help="Train steps for the model")
    args = parser.parse_args()

    model_id = "google/gemma-2-2b-it"

    # load the dataset
    data = load_data(args.train_path, args.valid_path, args.test_path)

    # tokenize the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_data = data.map(lambda x: tokenize_fnc(
        x, max_length=args.max_seq_length), remove_columns=["messages"])

    # load the model and configure it for lora
    model = prepare_model(model_id)
    model = prepare_model_for_kbit_training(model)
    lora_config = prepare_lora_config()
    model = get_peft_model(model, lora_config)

    sft_config = prepare_sft_config()
    trainer = SFTTrainer(
        model=model,
        train_dataset=tokenized_data["train"],
        eval_dataset=tokenized_data["validation"],
        args=sft_config,
        peft_config=lora_config
    )

    trainer.train()
    # Save if needed
    trainer.model.save_pretrained(args.output_dir + "_peft_weight")
    base_model = AutoModelForCausalLM.from_pretrained(model_id)
    merged_model = PeftModel.from_pretrained(model, model)
    merged_model = merged_model.merge_and_unload()

    # Save the final merged merged_model and tokenizer
    merged_model.save_pretrained(args.output_dir + "_merged", safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir + "_merged")
