"""
Purpose:
    This module defines several different torch.nn models to be trained in train.py
"""
import torch
import transformers
from dataset import BertTokenizedDataset, DistilBertTokenizedDataset



class Bert_A(torch.nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(Bert_A, self).__init__()
        self.dataLoaderType = BertTokenizedDataset
        
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(dropout_rate)
        self.feed = torch.nn.Linear(1536, 5)

    def forward(self, title_ids, title_mask, title_token_type_ids, body_ids, body_mask, body_token_type_ids):
        title_embedding = self.bert(title_ids, attention_mask=title_mask, token_type_ids=title_token_type_ids, return_dict=False)[1]
        body_embedding = self.bert(body_ids, attention_mask=body_mask, token_type_ids=body_token_type_ids, return_dict=False)[1]
        full_embedding = torch.cat((title_embedding, body_embedding), dim=1)
        dropout_embedding = self.drop(full_embedding)
        logits = self.feed(dropout_embedding)
        return logits



class DistilBert_A(torch.nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(DistilBert_A, self).__init__()
        self.dataLoaderType = DistilBertTokenizedDataset
        self.bert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', output_hidden_states=True)
        self.drop = torch.nn.Dropout(dropout_rate)
        self.feed = torch.nn.Linear(1536, 5)

    def forward(self, title_ids, title_mask, body_ids, body_mask):
        title_hs = self.bert(title_ids, attention_mask=title_mask).hidden_states
        title_cls_mean = torch.stack([layer[:,0,:] for layer in title_hs]).mean(dim=0)
        body_hs = self.bert(body_ids, attention_mask=body_mask).hidden_states
        body_cls_mean = torch.stack([layer[:,0,:] for layer in body_hs]).mean(dim=0)
        full_embedding = torch.cat((title_cls_mean, body_cls_mean), dim=1)
        dropout_embedding = self.drop(full_embedding)
        logits = self.feed(dropout_embedding)
        return logits



class DistilBert_B(torch.nn.Module):

    def __init__(self, dropout_rate=0.3):
        super(DistilBert_B, self).__init__()
        self.dataLoaderType = DistilBertTokenizedDataset

        self.bert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = torch.nn.Dropout(dropout_rate)
        self.feed1 = torch.nn.Linear(1536, 768)
        self.relu = torch.nn.ReLU()
        self.feed2 = torch.nn.Linear(768, 5)

    def forward(self, title_ids, title_mask, body_ids, body_mask):
        title_embedding = self.bert(title_ids, attention_mask=title_mask, return_dict=False)[0][:,0,:]
        body_embedding = self.bert(body_ids, attention_mask=body_mask, return_dict=False)[0][:,0,:]
        full_embedding = torch.cat((title_embedding, body_embedding), dim=1)
        dropout_embedding = self.drop(full_embedding)
        middle_layer = self.feed1(dropout_embedding)
        activated_layer = self.relu(middle_layer)
        logits = self.feed2(activated_layer)
        return logits



class DistilBert_SST_A(torch.nn.Module):
    """
    WIP
    https://huggingface.co/docs/transformers/main/en/model_doc/distilbert#transformers.DistilBertForSequenceClassification
    """

    def __init__(self, dropout_rate=0.3):
        super(DistilBert_SST_A, self).__init__()
        self.dataLoaderType = DistilBertTokenizedDataset

        self.bert = transformers.DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=5,
            output_attentions=False,
            output_hidden_states=True
        )
        self.drop = torch.nn.Dropout(dropout_rate)
        self.feed = torch.nn.Linear(1536, 5)

    def forward(self, title_ids, title_mask, body_ids, body_mask):
        title_embedding = self.bert(title_ids, attention_mask=title_mask, return_dict=False)
        print(title_embedding[0].size(), len(title_embedding[1]))
        
        body_embedding = self.bert(body_ids, attention_mask=body_mask, return_dict=False)[0][:,0,:]
        full_embedding = torch.cat((title_embedding, body_embedding), dim=1)
        dropout_embedding = self.drop(full_embedding)
        logits = self.feed(dropout_embedding)
        return logits