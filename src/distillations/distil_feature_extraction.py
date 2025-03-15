import json

from transformers import T5Config, T5ForConditionalGeneration, T5Tokenizer
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from datasets import load_dataset, Dataset
import torch.nn.functional as F

from src.utils.asset_paths import AssetPaths
from src.utils.helpers import load_t5_model_and_tokenizer, get_path_to, extract_text_and_intent, reformat_text

# Load teacher model (T5-small)
teacher_model, teacher_tokenizer, _ = load_t5_model_and_tokenizer(True, AssetPaths.T5_MULTITASK_FEATURE_EXTRACTION_MODEL.value)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_student_model(teacher_tokenizer):
    # Create a custom T5-Tiny configuration
    tiny_config = T5Config(
        d_model=256,       # Smaller hidden size (default in T5-small is 512)
        d_ff=1024,          # Reduce feed-forward network size
        num_layers=3,      # Fewer encoder layers (T5-small has 6)
        num_decoder_layers=3,  # Reduce decoder layers
        num_heads=6,       # Fewer attention heads (T5-small has 8)
        vocab_size=32128,  # Same as the teacher model (T5-small)
        dropout_rate=0.1,  # Standard dropout rate
        layer_norm_epsilon=1e-6,
        decoder_start_token_id=0,  # Set to the same as T5-small
        pad_token_id=0,  # T5 uses 0 as the padding token
    )

    # Initialize a custom student model
    student_model = T5ForConditionalGeneration(tiny_config)
    student_tokenizer = teacher_tokenizer  # Use the same tokenizer

    return student_model, student_tokenizer

def preprocess_data(example):
    """Tokenizes input text and generates teacher model outputs."""
    # Tokenize input text
    inputs = teacher_tokenizer(
        example["input"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )

    # Ensure `input_ids` is properly shaped
    input_ids = inputs["input_ids"].to(device)  # Shape: (1, seq_len)
    attention_mask = inputs["attention_mask"].to(device)

    # Tokenize target text (labels)
    labels = teacher_tokenizer(
        example["output"],  # Ensure dataset has output/target text
        padding="max_length",
        truncation=True,
        max_length=128,  # Adjust as needed
        return_tensors="pt"
    )["input_ids"].to(device)

    # Get teacher model outputs (soft labels)
    with torch.no_grad():
        teacher_outputs = teacher_model.generate(
            input_ids=input_ids,  # Make sure it's not squeezed
            attention_mask=attention_mask,
            output_scores=True,
            # max_length=1024,
            # min_length=1024,
            # do_sample=False,
            return_dict_in_generate=True
        )

    # Convert teacher logits to a tensor
    teacher_logits = torch.stack(teacher_outputs.scores, dim=1)  # Shape: (seq_len, vocab_size)

    return {
        "input_ids": input_ids.squeeze(0),
        "attention_mask": attention_mask.squeeze(0),
        "labels": labels.squeeze(0),  # Actual target labels
        "teacher_logits": teacher_logits.cpu()  # Move to CPU to avoid memory overflow
    }

def collate_fn(batch):
    """Collates batch and ensures tensors are properly formatted."""
    input_ids = torch.stack([torch.tensor(item["input_ids"], dtype=torch.long) for item in batch])
    attention_mask = torch.stack([torch.tensor(item["attention_mask"], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item["labels"], dtype=torch.long) for item in batch])

    # Convert teacher logits into tensors and ensure they're all 3D `[1, seq_len, vocab_size]`
    teacher_logits_list = [
        torch.tensor(item["teacher_logits"], dtype=torch.float).unsqueeze(0)  # Add batch dimension if missing
        if len(torch.tensor(item["teacher_logits"], dtype=torch.float).shape) == 2
        else torch.tensor(item["teacher_logits"], dtype=torch.float)  # Keep it as-is if already 3D
        for item in batch
    ]

    # Determine max sequence length in batch
    max_len = max(logits.shape[1] for logits in teacher_logits_list)  # Sequence length dimension

    # Pad teacher logits along the sequence length dimension
    teacher_logits_padded = torch.stack([
        F.pad(logits, (0, 0, 0, max_len - logits.shape[1]), value=-1e9)  # Padding on seq_len dimension
        for logits in teacher_logits_list
    ])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "teacher_logits": teacher_logits_padded
    }

def load_data():
    # Load dataset in streaming mode
    dataset = load_dataset("json", data_files=get_path_to(AssetPaths.FEATURE_EXTRACTION_DATASET.value), split="train", streaming=True)

    # On-the-fly preprocessing and conversion
    def generator():
        for item in dataset:
            text = item["input"]
            slots = json.dumps(item["slots"])  # Convert slots to JSON string

            # Yield one example at a time
            yield {"input": f"extract slot: {text}", "output": slots}
            yield {"input": f"retrieve category: {text}", "output": item["category"]}
            yield {"input": f"extract features: {text}", "output": f"{slots}, category: {item['category']}"}

    # Use generator for dataset creation
    processed_dataset = Dataset.from_generator(generator)

    # Apply preprocessing directly on processed_dataset
    processed_dataset = processed_dataset.map(preprocess_data, batched=True)

    # Convert to DataLoader
    train_dataloader = DataLoader(processed_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

    return train_dataloader


def train_student():
    student_model, student_tokenizer = get_student_model(teacher_tokenizer)

    teacher_model.to(device)
    student_model.to(device)

    train_dataloader = load_data()

    # Optimizer
    optimizer = AdamW(student_model.parameters(), lr=5e-5)

    # Training Loop
    num_epochs = 2
    student_model.train()

    for epoch in range(num_epochs):
        total_loss = 0

        for batch in train_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            teacher_logits = batch["teacher_logits"].to(device)  # KD Loss

            # Generate decoder_input_ids from labels
            decoder_input_ids = teacher_model.prepare_decoder_input_ids_from_labels(labels=labels).to(device)

            # Forward pass with student model
            outputs = student_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )

            # Compute Student Loss (Cross-Entropy)
            ce_loss = outputs.loss

            # Compute Knowledge Distillation Loss (KL Divergence)
            student_logits = outputs.logits  # Student model logits

            # Ensure both logits have the same shape
            student_logits = student_logits.view(-1, student_logits.shape[-1])  # Flatten to (batch_size*num_tokens, vocab_size)
            teacher_logits = teacher_logits.view(-1, teacher_logits.shape[-1])  # Flatten to match student

            # Pad teacher_logits to match student_logits in the first dimension
            teacher_logits_padded = F.pad(
                teacher_logits,  # (32, vocab_size)
                (0, 0, 0, student_logits.shape[0] - teacher_logits.shape[0]),  # Padding on dim 0
                "constant",
                -1e9  # -1e9 for masking
            )

            kd_loss = torch.nn.functional.kl_div(
                torch.log_softmax(student_logits, dim=-1),  # Student
                torch.softmax(teacher_logits_padded, dim=-1),  # Teacher (Soft labels)
                reduction="batchmean"
            )

            alpha = .5

            # Combine CE loss and KD loss
            loss = ce_loss + alpha * kd_loss  # Adjust `alpha` for balance

            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {total_loss / 9670}")

        print(f"Epoch {epoch + 1}, Loss: {total_loss / 9670}")

    # Save model locally
    student_model.save_pretrained(get_path_to(AssetPaths.T5_DISTIL_SLOT_FILLER_MODEL.value))
    student_tokenizer.save_pretrained(get_path_to(AssetPaths.T5_DISTIL_SLOT_FILLER_MODEL.value))


train_student()
