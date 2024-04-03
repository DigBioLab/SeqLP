import numpy as np
import torch

def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling."""
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token_id
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.mask_token_id

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer.vocab), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels



from torch.utils.data import Dataset

class AminoAcidDataset(Dataset):
    def __init__(self, encodings, tokenizer):
        self.encodings = encodings
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        input_ids, labels = mask_tokens(item['input_ids'], self.tokenizer)
        item['input_ids'] = input_ids
        item['labels'] = labels
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])