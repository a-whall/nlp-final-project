"""
Purpose:
    This script provides a custom Dataset class to be used as a component of a DataLoader in
    train.py. It handles converting the AItAS data samples into the inputs expected by BERT 
    and the AItA_Classifier model defined in model.py. It also encodes the labels on the fly 
    based on the value assigned to ENCODING below, providing flexibility to easily implement 
    new encoding methods in the future.
"""
import torch
from torch import tensor
from torch.utils.data import Dataset
from transformers import BertTokenizer, DistilBertTokenizer



ENCODING = 'onehot'



def encode_label(original_label):

    if ENCODING == 'onehot':
        match original_label:
            case 'Not the A-hole':
                return [1, 0, 0, 0, 0]
            case 'Asshole':
                return [0, 1, 0, 0, 0]
            case 'No A-holes here':
                return [0, 0, 1, 0, 0]
            case 'Everyone Sucks':
                return [0, 0, 0, 1, 0]
            case 'Not enough info':
                return [0, 0, 0, 0, 1]
            case _:
                raise ValueError(f"Invalid label encountered: {original_label}")
            
    raise NotImplementedError(f"Encoding type {ENCODING} not implemented")



class BertTokenizedDataset(Dataset):

    def __init__(self, arrow_dataset_Dataset):
        self.data = arrow_dataset_Dataset
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        title_text = " ".join(str(item['title']).split()) # clean up irregular spaces

        body_text = " ".join(str(item['body']).split()) # also use str() in case of None

        encode_args = {
            'add_special_tokens': True,
            'truncation': True,
            'padding': 'max_length',
            'return_token_type_ids': True
        }

        title_inputs = self.tokenizer.encode_plus(title_text, max_length=32, **encode_args)

        body_inputs = self.tokenizer.encode_plus(body_text, max_length=512, **encode_args)

        target = encode_label(str(item['label']))

        return [
            tensor(title_inputs['input_ids'],      dtype=torch.long),
            tensor(title_inputs['attention_mask'], dtype=torch.long),
            tensor(title_inputs["token_type_ids"], dtype=torch.long),
            tensor(body_inputs['input_ids'],       dtype=torch.long),
            tensor(body_inputs['attention_mask'],  dtype=torch.long),
            tensor(body_inputs["token_type_ids"],  dtype=torch.long),
            tensor(target,                         dtype=torch.float)
        ]
    


class DistilBertTokenizedDataset(Dataset):

    def __init__(self, arrow_dataset_Dataset):
        self.data = arrow_dataset_Dataset
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        item = self.data[index]

        title_text = " ".join(str(item['title']).split()) # clean up irregular spaces

        body_text = " ".join(str(item['body']).split()) # also use str() in case of None

        encode_args = {
            'add_special_tokens': True,
            'truncation': True,
            'padding': 'max_length'
        }

        title_inputs = self.tokenizer(title_text, max_length=32, **encode_args)

        body_inputs = self.tokenizer(body_text, max_length=512, **encode_args)

        target = encode_label(str(item['label']))

        return [
            tensor(title_inputs['input_ids'],      dtype=torch.long),
            tensor(title_inputs['attention_mask'], dtype=torch.long),
            tensor(body_inputs['input_ids'],       dtype=torch.long),
            tensor(body_inputs['attention_mask'],  dtype=torch.long),
            tensor(target,                         dtype=torch.float)
        ]