import torch
import transformers



class AItA_Classifier(torch.nn.Module):

    def __init__(self):
        super(AItA_Classifier, self).__init__()
        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.drop = torch.nn.Dropout(0.2)
        self.feed = torch.nn.Linear(1536, 5)

    def forward(self, title_ids, title_mask, title_token_type_ids, body_ids, body_mask, body_token_type_ids):
        _, title_embedding = self.bert(title_ids, attention_mask=title_mask, token_type_ids=title_token_type_ids, return_dict=False)
        _, body_embedding = self.bert(body_ids, attention_mask=body_mask, token_type_ids=body_token_type_ids, return_dict=False)
        full_embedding = torch.cat((title_embedding, body_embedding), dim=1)
        dropout_embedding = self.drop(full_embedding)
        logits = self.feed(dropout_embedding)
        return logits