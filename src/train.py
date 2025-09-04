import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from data_loader import load_termium_data, prepare_training_data
from dataset import DefinitionTermDataset

# Load data
data = load_termium_data("../data/termium.json")
inputs, targets = prepare_training_data(data)

# Initialize tokenizers and models
mt5_tokenizer = AutoTokenizer.from_pretrained("google/mt5-small", legacy=False)
byt5_tokenizer = AutoTokenizer.from_pretrained("google/byt5-small", legacy=False)

mt5_model = AutoModelForSeq2SeqLM.from_pretrained("google/mt5-small")
byt5_model = AutoModelForSeq2SeqLM.from_pretrained("google/byt5-small")

# Create datasets
train_dataset_mt5 = DefinitionTermDataset(inputs, targets, mt5_tokenizer)
train_dataset_byt5 = DefinitionTermDataset(inputs, targets, byt5_tokenizer)

# Create data collators
mt5_data_collator = DataCollatorForSeq2Seq(tokenizer=mt5_tokenizer, model=mt5_model, padding=True)
byt5_data_collator = DataCollatorForSeq2Seq(tokenizer=byt5_tokenizer, model=byt5_model, padding=True)

# Training arguments
training_args_mt5 = TrainingArguments(
    output_dir="../models/mt5-results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=False,
    logging_dir="../logs",
)

training_args_byt5 = TrainingArguments(
    output_dir="../models/byt5-results",
    evaluation_strategy="epoch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    num_train_epochs=1,
    save_strategy="epoch",
    fp16=True,
    gradient_checkpointing=False,
    logging_dir="../logs",
)

# Create trainers
trainer_mt5 = Trainer(
    model=mt5_model,
    args=training_args_mt5,
    train_dataset=train_dataset_mt5,
    data_collator=mt5_data_collator,
)

trainer_byt5 = Trainer(
    model=byt5_model,
    args=training_args_byt5,
    train_dataset=train_dataset_byt5,
    data_collator=byt5_data_collator,
)

if __name__ == "__main__":
    print("Training mT5...")
    trainer_mt5.train()
    
    print("Training ByT5...")
    trainer_byt5.train()
