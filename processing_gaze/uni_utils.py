import json
import pickle
from pathlib import Path

def save_pickle(obj, filepath):
    """Save a Python object to a pickle file."""
    filepath = Path(filepath)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """Load a Python object from a pickle file."""
    filepath = Path(filepath)
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_json(obj, filepath, indent=4):
    """Save a Python object to a JSON file."""
    filepath = Path(filepath)
    with open(filepath, 'w') as f:
        json.dump(obj, f)

def load_json(filepath):
    """Load a Python object from a JSON file."""
    filepath = Path(filepath)
    with open(filepath, 'r') as f:
        return json.load(f)
