import json
import warnings

warnings.filterwarnings('ignore')

def load_termium_data(data_path="data/termium.json"):
    """Load the TERMIUM dataset from JSON file."""
    with open(data_path, "rt") as file:
        data = json.load(file)
    return data

def filter_data_with_definitions(data, split):
    """Filter entries that have English definitions."""
    filtered_data = []
    for entry in data[split]:
        if ("def" in entry["en"] and 
            entry["en"]["def"] and 
            "text" in entry["en"]["def"] and
            entry["en"]["def"]["text"] is not None):
            filtered_data.append(entry)
    return filtered_data

def prepare_training_data(data):
    """Prepare training data by extracting definitions and terms."""
    train_data = filter_data_with_definitions(data, "train")
    
    inputs = []
    targets = []
    
    for item in train_data:
        definition = item["en"]["def"]["text"]
        term = item["en"]["text"]
        inputs.append(definition)
        targets.append(term)
    
    return inputs, targets
