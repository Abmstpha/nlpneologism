import torch
from torch.utils.data import Dataset

class DefinitionTermDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, max_length=512):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        # Add a prefix to make it clear this is a definition-to-term task
        input_text = f"Generate term: {self.inputs[idx]}"
        target_text = self.targets[idx]

        # Tokenize inputs and targets
        model_inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding=False,  # Let the data collator handle padding
            truncation=True,
        )

        # Tokenize targets
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                target_text,
                max_length=self.max_length,
                padding=False,  # Let the data collator handle padding
                truncation=True,
            )

        model_inputs['labels'] = labels['input_ids']
        return model_inputs
