import os
import ssl
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset

ssl._create_default_https_context = ssl._create_unverified_context

class RealTextDataset(Dataset):
    """
    Real-world language dataset using HuggingFace 'salesforce/wikitext' and GPT2 BPE Tokenizer.
    Includes disk caching for instant dataset loading.
    """
    def __init__(self, split: str = "train", seq_len: int = 256, cache_dir: str = "results"):
        self.seq_len = seq_len
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"tokens_{split}_{seq_len}.pt")

        if os.path.exists(cache_path):
            print(f"Loading pre-tokenized dataset from cache '{cache_path}'...")
            self.data = torch.load(cache_path)
            print(f"Loaded {len(self.data)} text sequences of length {seq_len} from cache.")
            return

        print(f"Tokenizing HuggingFace 'salesforce/wikitext' ({split} split) with GPT2 Tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.vocab_size = self.tokenizer.vocab_size # 50257
        
        raw_data = load_dataset("salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        
        full_text = "\n".join([item["text"] for item in raw_data if item["text"].strip()])
        tokens = self.tokenizer.encode(full_text)
        
        total_tokens = len(tokens)
        chunk_size = seq_len + 1
        num_chunks = total_tokens // chunk_size
        
        tokens_tensor = torch.tensor(tokens[:num_chunks * chunk_size], dtype=torch.long)
        self.data = tokens_tensor.view(num_chunks, chunk_size)
        
        torch.save(self.data, cache_path)
        print(f"Cached {num_chunks} real text sequences to '{cache_path}'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunk = self.data[idx]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

def get_dataloaders(vocab_size: int = 50257, seq_len: int = 256, batch_size: int = 4):
    train_dataset = RealTextDataset(split="train", seq_len=seq_len)
    val_dataset = RealTextDataset(split="validation", seq_len=seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader
