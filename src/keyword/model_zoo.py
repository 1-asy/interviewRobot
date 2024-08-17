import torch.nn.functional as F 
from torchcrf import CRF
from torch import nn 

# 构建基于bilstm实现ner
class bilstm(nn.Module):
    def __init__(self, parameter):
        super(bilstm, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out.view(-1,out.size(-1))

# 构建基于bilstm+crf实现ner
class bilstm_crf(nn.Module):
    def __init__(self, parameter):
        super(bilstm_crf, self).__init__()
        word_size = parameter['word_size']
        embedding_dim = parameter['d_model']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)

        hidden_size = parameter['hid_dim']
        num_layers = parameter['n_layers']
        dropout = parameter['dropout']
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)

        output_size = parameter['output_size']
        self.fc = nn.Linear(hidden_size*2, output_size)
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, x):
        out = self.embedding(x)
        out,(h, c)= self.lstm(out)
        out = self.fc(out)
        return out
    
# 基于bert预训练模型
from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch

import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器

class bert(BertPreTrainedModel):
    def __init__(self, config,parameter):
        super(bert, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits.view(-1,logits.size(-1))
    
# 构建基于bert+crf实现ner
class bert_crf(BertPreTrainedModel):
    def __init__(self, config,parameter):
        super(bert_crf, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        embedding_dim = parameter['d_model']
        output_size = parameter['output_size']
        self.fc = nn.Linear(embedding_dim, output_size)
        self.init_weights()
        
        self.crf = CRF(output_size,batch_first=True)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        outputs = self.bert(input_ids = input_ids,attention_mask=attention_mask,token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.fc(sequence_output)
        return logits
    
config_class, model_class, tokenizer_class = BertConfig, bert, BertTokenizer
config = config_class.from_pretrained("prev_trained_model")
tokenizer = tokenizer_class.from_pretrained("prev_trained_model")
    

# config_class, = BertConfig,
# config = config_class.from_pretrained("bert-base-chinese")
# tokenizer = tokenizer_class.from_pretrained("bert-base-chinese")
# model = model_class.from_pretrained("bert-base-chinese",config=config)
# config = config_class.from_pretrained("prev_trained_model")
# tokenizer = tokenizer_class.from_pretrained("prev_trained_model")
# model = model_class.from_pretrained("prev_trained_model",config=config,parameter = parameter)