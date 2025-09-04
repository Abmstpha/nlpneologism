# Definition-to-Neologism Generation

This project implements definition-to-neologism generation using mT5 and ByT5 models, inspired by the research of Paul Lerner in his paper [Towards Machine Translation of Scientific Neologisms](https://aclanthology.org/2024.jeptalnrecital-taln.17/).

## Overview

The project focuses on the "DEF" setting: given a definition, generate the corresponding term. We compare two approaches:
- **mT5**: A multilingual T5 model using BPE tokenization
- **ByT5**: A byte-level T5 model using character-level tokenization

### Example
- **Input**: "Having to do with the ability to transmit data in either direction."
- **Expected Output**: "bidirectional"

## Project Structure

```
nlpneologism/
├── src/                      # Source code
│   ├── data_loader.py        # Data loading functions
│   ├── dataset.py            # PyTorch dataset class
│   └── train.py              # Training script
├── notebooks/                # Jupyter notebooks
│   └── Definition-to-Neologism.ipynb
├── data/                     # Dataset storage (place termium.json here)
├── models/                   # Trained model storage
├── logs/                     # Training logs
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Abmstpha/nlpneologism.git
cd nlpneologism
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the TERMIUM dataset and place `termium.json` in the `data/` folder

## Usage

Run the training script:
```bash
cd src
python train.py
```

## Dataset

The project uses the TERMIUM dataset, which provides English definitions and corresponding terms.

## Model Comparison

### mT5 (Multilingual T5)
- Uses BPE tokenization
- Example: "bidirectional" → `['▁bi', 'direction', 'al']`

### ByT5 (Byte-level T5)  
- Uses character-level tokenization
- Example: "bidirectional" → `['b', 'i', 'd', 'i', 'r', 'e', 'c', 't', 'i', 'o', 'n', 'a', 'l']`
