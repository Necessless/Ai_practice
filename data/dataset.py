from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.nn.utils.rnn import pad_sequence
import torch
import re 


def simple_sent_tokenize(text):
    sentences = re.split(r'(?<=[.!?…])\s+', text.strip())
    return [s.strip() for s in sentences if s]


class ZaratustraDataset(Dataset):

    def __init__(self, stage = "train"):
        self.setup_dataset(stage)
        try:
            self.tokenizer: Tokenizer = Tokenizer.from_file("data/tokenizer.json")
        except Exception:
            self.tokenizer = self.setup_tokenizer()

    def setup_tokenizer(self):
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=30000, special_tokens=["<pad>", "<s>", "</s>"])
        tokenizer.train_from_iterator(self.sentences, trainer)
        tokenizer.save('data/tokenizer.json')
        return tokenizer

    def setup_dataset(self, stage):
        with open(f'data/Zaratustra_{stage}.txt', mode = "r", encoding="utf-8") as f:
            text = f.read()
            self.sentences = simple_sent_tokenize(text)

    def get_pad_token_id(self):
        return self.tokenizer.token_to_id('<pad>')

    def get_bos_token_id(self):
        return self.tokenizer.token_to_id('<s>')

    def get_eos_token_id(self):
        return self.tokenizer.token_to_id('</s>')

    def __len__(self):
        return len(self.sentences)
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    def __getitem__(self, index):
        bos_token_id = self.tokenizer.token_to_id('<s>')
        eos_token_id = self.tokenizer.token_to_id('</s>')
        sentence = self.sentences[index]
        tokens = [bos_token_id] + self.tokenizer.encode(sentence).ids + [eos_token_id]
        src_text = tokens[:-1]
        trgt_text = tokens[1:]
        return {
            "src_ids": torch.tensor(src_text, dtype=torch.long),
            "trgt_ids": torch.tensor(trgt_text, dtype=torch.long)
        }
    

class Collator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        src_ids = [item['src_ids'] for item in batch if item is not None]
        tgt_ids = [item['trgt_ids'] for item in batch if item is not None]
        src_ids = pad_sequence(src_ids, batch_first=True, padding_value=self.pad_token_id)
        tgt_ids = pad_sequence(tgt_ids, batch_first=True, padding_value=self.pad_token_id)
        return {
            'src_ids': src_ids,
            'trgt_ids': tgt_ids
        }


def create_dataloaders(
    batch_size=1, 
    max_length=128, 
    max_train_samples=None, 
    max_val_samples=None, 
    num_workers=0
):  
    train_dataset = ZaratustraDataset(
        stage="train"
    )

    val_dataset = ZaratustraDataset(
        stage="val"
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=Collator(train_dataset.tokenizer.token_to_id('<pad>')),
        num_workers=num_workers, 
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=Collator(val_dataset.tokenizer.token_to_id('<pad>')),
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset 
