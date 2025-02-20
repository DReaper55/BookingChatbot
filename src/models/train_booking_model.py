import itertools

from transformers import Trainer, TrainingArguments

from src.data.load_dataset import load_booking_dataset, load_rag_dataset
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to


def train_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer()

    # Load dataset
    train_dataloader, val_dataloader = load_booking_dataset()

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        # logging_dir="./logs",
        # logging_steps=100,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        # max_steps=10,
        fp16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
        # data_collator=data_collator
    )

    # Train
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def train_rag_based_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer(True, get_path_to(AssetPaths.T5_BOOKING_MODEL.value))

    # Load dataset
    train_dataloader, val_dataloader = load_rag_dataset()

    # Take the first 3 samples from the dataset
    # def take_samples(dataloader, n):
    #     return list(itertools.islice(dataloader, n))
    #
    # # Take the first 3 samples from the eval DataLoader
    # sample_batch = take_samples(val_dataloader, 3)
    #
    # # Print preprocessed examples
    # for i, sample in enumerate(sample_batch):
    #     decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    #     decoded_target = tokenizer.decode(sample["labels"], skip_special_tokens=True)
    #
    #     print(f"\nSample {i+1} --------------------")
    #     print("Decoded Input:", decoded_input)
    #     print("Decoded Target (Expected Response):", decoded_target)

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        fp16=True
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataloader,
        eval_dataset=val_dataloader,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # Train
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


# Train booking model
# train_model(get_path_to(AssetPaths.T5_BOOKING_MODEL.value))

# Fine-tune booking model for RAG-based responses
train_rag_based_model(get_path_to(AssetPaths.T5_BOOKING_MODEL.value))
