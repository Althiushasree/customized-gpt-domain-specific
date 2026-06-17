"""
train.py — Fine-tune GPT-2 on a custom domain dataset using Hugging Face Transformers.

Usage:
    python train.py
    python train.py --epochs 5 --batch_size 4 --lr 5e-5 --data data/train.txt --output output/model
"""

import argparse
import os
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    TextDataset,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_dataset(file_path: str, tokenizer, block_size: int = 128):
    return TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )


def get_data_collator(tokenizer):
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )


def train(
    data_path: str = "data/train.txt",
    output_dir: str = "output/model",
    model_name: str = "gpt2",
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 5e-5,
    block_size: int = 128,
    save_steps: int = 100,
    logging_steps: int = 50,
):
    print(f"Loading tokenizer and model: {model_name}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    model = GPT2LMHeadModel.from_pretrained(model_name)

    print(f"Loading dataset from: {data_path}")
    train_dataset = load_dataset(data_path, tokenizer, block_size=block_size)
    data_collator = get_data_collator(tokenizer)

    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        save_steps=save_steps,
        logging_steps=logging_steps,
        save_total_limit=2,
        prediction_loss_only=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("Starting fine-tuning...")
    trainer.train()

    print(f"Saving fine-tuned model to: {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Training complete!")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-2 on domain-specific data.")
    parser.add_argument("--data", type=str, default="data/train.txt", help="Path to training text file.")
    parser.add_argument("--output", type=str, default="output/model", help="Directory to save the fine-tuned model.")
    parser.add_argument("--model", type=str, default="gpt2", help="Base model name (e.g. gpt2, gpt2-medium).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--block_size", type=int, default=128, help="Token block size for dataset chunking.")
    args = parser.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        block_size=args.block_size,
    )


if __name__ == "__main__":
    main()
