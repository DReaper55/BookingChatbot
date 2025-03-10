import itertools

from transformers import Trainer, TrainingArguments

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.load_dataset import load_slot_filler_dataset, load_rag_dataset
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to


def train_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer()

    # Load dataset
    train_dataloader, val_dataloader = load_slot_filler_dataset()

    # Take the first 3 samples from the dataset
    # sample_batch = list(train_dataloader.take(3))
    #
    # # Print preprocessed examples
    # for i, sample in enumerate(sample_batch):
    #     decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
    #     decoded_target = tokenizer.decode(sample["labels"], skip_special_tokens=True)
    #
    #     print(f"\nSample {i+1} --------------------")
    #     print("Decoded Input:", decoded_input)
    #     print("Decoded Target (Expected Response):", decoded_target)

    def take_samples(dataloader, n):
        return list(itertools.islice(dataloader, n))

    # Take the first 3 samples from the eval DataLoader
    sample_batch = take_samples(val_dataloader, 20)

    # Print preprocessed examples
    for i, sample in enumerate(sample_batch):
        decoded_input = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        decoded_target = tokenizer.decode(sample["labels"], skip_special_tokens=True)

        print(f"\nSample {i+1} --------------------")
        print("Decoded Input:", decoded_input)
        print("Decoded Target (Expected Response):", decoded_target)

    # Define Training Arguments
    # training_args = TrainingArguments(
    #     output_dir=output_dir,
    #     per_device_train_batch_size=8,
    #     per_device_eval_batch_size=8,
    #     num_train_epochs=5,
    #     learning_rate=3e-4,
    #     save_steps=500,
    #     save_total_limit=2,
    #     evaluation_strategy="epoch",
    #     save_strategy="epoch",
    #     push_to_hub=False,
    # )
    #
    # # Trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataloader,
    #     eval_dataset=val_dataloader,
    #     tokenizer=tokenizer,
    #     data_collator=data_collator
    # )
    #
    # # Train
    # trainer.train()
    # model.save_pretrained(output_dir)
    # tokenizer.save_pretrained(output_dir)


# train_model(get_path_to(AssetPaths.T5_SLOT_FILLER_MODEL.value))
