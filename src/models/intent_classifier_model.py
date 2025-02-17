from transformers import Trainer, TrainingArguments

from src.data.load_dataset import load_intent_classifier_dataset
from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to


def train_model(output_dir):
    # Load model
    model, tokenizer, data_collator = load_t5_model_and_tokenizer()

    # Load dataset
    train_dataloader, val_dataloader = load_intent_classifier_dataset()

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


train_model(get_path_to(AssetPaths.T5_INTENT_CLASSIFIER_MODEL.value))
