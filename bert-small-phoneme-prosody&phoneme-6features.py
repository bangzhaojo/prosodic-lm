import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

import torch
from torch.utils.data import Dataset, DataLoader
import random
from torch.optim import AdamW
import torch.nn as nn
from bertPhoneme import BertEmbeddingsV2, BertModelV2, BertForMaskedLMV2, BertConfigV2, MaskedLMWithProsodyOutput
import json
from tqdm import tqdm
from datasets import load_dataset
from transformers import get_linear_schedule_with_warmup



with open("/shared/3/projects/bangzhao/prosodic_embeddings/bert_train/phoneme_vocab.json", "r") as f: # changed
    phoneme_vocab = json.load(f)

cluster_size = 100

phoneme_vocab_size = len(phoneme_vocab)
mask_token_id = phoneme_vocab["SIL"] # new
pad_token_id = 72 # changed
pad_cluster_id = cluster_size + 1
mask_prosody_id = cluster_size


class HuggingFacePhonemeDataset(Dataset):
    def __init__(self, hf_dataset, vocab, mask_prob=0.15, max_length=512):
        self.dataset = hf_dataset
        self.vocab = vocab
        self.mask_prob = mask_prob
        self.max_length = max_length

        # NEW: Build (row_idx, chunk_start) mapping
        self.index_map = []
        print("Indexing chunks from dataset...")
        for row_idx, sample in tqdm(enumerate(self.dataset), total=len(self.dataset), desc="Chunking"):
            length = len(sample["phoneme"])
            for start in range(0, length, max_length):
                self.index_map.append((row_idx, start))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        row_idx, start = self.index_map[idx]
        sample = self.dataset[row_idx]

        phonemes = sample["phoneme"][start:start + self.max_length]
        prosody_ids = sample["prosody_id_100"][start:start + self.max_length]  # change cluster size here

        # Tokenize
        input_ids = [self.vocab.get(p, self.vocab["UNK"]) for p in phonemes]
        
        labels = input_ids.copy() # new
        prosody_labels = prosody_ids.copy()

        # Mask prosody
        for i in range(len(prosody_ids)):
            if random.random() < self.mask_prob:
                labels[i] = input_ids[i] # new
                input_ids[i] = mask_token_id # changed 
                prosody_labels[i] = prosody_ids[i]
                prosody_ids[i] = mask_prosody_id
            else:
                labels[i] = -100 # new 
                prosody_labels[i] = -100

        # Padding
        pad_length = self.max_length - len(input_ids)
        input_ids.extend([pad_token_id] * pad_length)
        labels.extend([-100] * pad_length) # new
        prosody_ids.extend([pad_cluster_id] * pad_length)
        prosody_labels.extend([-100] * pad_length)
        attention_mask = [1] * (len(input_ids) - pad_length) + [0] * pad_length

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long), # new
            "prosody_ids": torch.tensor(prosody_ids, dtype=torch.long),
            "prosody_labels": torch.tensor(prosody_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


hf_dataset = load_dataset("json", data_files="/shared/3/projects/bangzhao/prosodic_embeddings/merge/training_data_6features/output_part_1_20kSample.jsonl", split="train")
hf_dataset_train = hf_dataset.select(range(19800))
hf_dataset_test = hf_dataset.select(range(19800, 20000))

train_dataset = HuggingFacePhonemeDataset(hf_dataset_train, phoneme_vocab)
test_dataset = HuggingFacePhonemeDataset(hf_dataset_test, phoneme_vocab)

train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

max_length = 512

model_config = BertConfigV2(
    vocab_size=phoneme_vocab_size + 1,
    pad_token_id=pad_token_id,
    pad_cluster_id=cluster_size + 1,
    hidden_size=512,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=2048,
    max_position_embeddings=max_length,
    prosody_cluster_size=cluster_size + 2
)

model = BertForMaskedLMV2(config=model_config)

# BERT-Base	768	12	12	3072
# BERT-Small 512	4	8	2048
# BERT-Mini	256	4	4	1024
# BERT-Tiny	128	2	2	512

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set paths
save_dir = "/shared/3/projects/bangzhao/prosodic_embeddings/bert_train/mlm_prosody&phoneme_20kSample_6features_100clu/" # changed
os.makedirs(save_dir, exist_ok=True)

# Always start from scratch
start_step = 0
total_epochs = 1

model.to(device)
optimizer = AdamW(model.parameters(), lr=5e-4)

# Define warmup scheduler
num_training_steps = len(train_loader) * total_epochs
num_warmup_steps = int(0.1 * num_training_steps)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps
)

mlm_loss_fn = nn.CrossEntropyLoss(ignore_index=-100) # new
prosody_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

model.train()
total_loss = 0

print(f"Training for {total_epochs} epoch(s)...")

for epoch in range(total_epochs):
    print(f"\nEpoch {epoch} starting...\n")
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}")
    
    for step, batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device) # new 
        prosody_ids = batch["prosody_ids"].to(device)
        prosody_labels = batch["prosody_labels"].to(device) # changed

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, prosody_ids=prosody_ids)

        mlm_loss = mlm_loss_fn(outputs.logits.view(-1, model.config.vocab_size), labels.view(-1)) # new
        prosody_loss = prosody_loss_fn(
            outputs.prosody_logits.view(-1, model.config.prosody_cluster_size),
            prosody_labels.view(-1) # changed
        )

        total_batch_loss = mlm_loss + prosody_loss # changed
        total_batch_loss.backward() 
        optimizer.step()
        scheduler.step()

        total_loss += total_batch_loss.item()

        # Save every 500 steps
        current_step = step + 1
        if current_step % 100 == 0:
            save_path = os.path.join(save_dir, f"mlm_prosody_step{current_step}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": model.config.to_dict(),
                "step": current_step,
                "epoch": epoch
            }, save_path)

        # Update progress bar
        avg_loss = total_loss / (step + 1)
        if step % 10 == 0:
            progress_bar.set_postfix({"Avg Loss": f"{avg_loss:.4f}"})


# Final save after training
torch.save({
    "model_state_dict": model.state_dict(),
    "config": model.config.to_dict()
}, os.path.join(save_dir, "mlm_prosody_final.pt"))
