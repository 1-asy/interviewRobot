from transformers import WEIGHTS_NAME, BertConfig,get_linear_schedule_with_warmup,AdamW, BertTokenizer
from transformers import BertModel,BertPreTrainedModel
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch
from operator import itemgetter
import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn,optim # 构建模型和优化器
from torchcrf import CRF
import pickle as pk
import numpy as np

max_len = 512-25

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
    


def load_model(root_path = './'):
    parameter = pk.load(open(root_path+'parameter.pkl','rb'))
    model = bert_crf(config,parameter).to(parameter['device'])
    model.load_state_dict(torch.load(root_path+'bert_crf.h5'))
    model.eval()
    return model,parameter

def list2torch(ins):
    return torch.from_numpy(np.array(ins)).long().to(parameter['device'])

def keyword_predict(input):
    input = list(input)
    input_id = tokenizer.convert_tokens_to_ids(input)
    predict = model.crf.decode(model(list2torch([input_id])))[0]
    predict = itemgetter(*predict)(parameter['ind2key'])
    keys_list = []
    for ind,i in enumerate(predict):
        if i == 'O':
            continue
        if i[0] == 'S':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],True])
            continue
        if i[0] == 'B':
            if not(len(keys_list) == 0 or keys_list[-1][-1]):
                del keys_list[-1]
            keys_list.append([input[ind],[i],[ind],False])
            continue
        if i[0] == 'I':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
        if i[0] == 'E':
            if len(keys_list) > 0 and not keys_list[-1][-1] and \
            keys_list[-1][1][0].split('-')[1] == i.split('-')[1]:
                keys_list[-1][0] += input[ind]
                keys_list[-1][1] += [i]
                keys_list[-1][2] += [ind]
                keys_list[-1][3] = True
            else:
                if len(keys_list) > 0:
                    del keys_list[-1]
            continue
#     print(keys_list)
    keys_list = [i[0] for i in keys_list]
    return list(set(keys_list))

def keyword_predict_long_text(input):
    if len(input) < max_len:
        return keyword_predict(input)
    else:
        keys_list = []
        pad = 25
        for i in range(len(input)//max_len+1):
            if i == 0:
                input_slice = input[i*max_len:(i+1)*max_len]
            else:
                input_slice = input[i*max_len-pad:(i+1)*max_len]
            keys_list += keyword_predict(input_slice)
    return list(set(keys_list))

import sys
sys.path.append('../')
from path_config import KEYWORD_MODEL_PATH
config_class, bert_crf, tokenizer_class = BertConfig, bert_crf, BertTokenizer
config = config_class.from_pretrained(KEYWORD_MODEL_PATH+"prev_trained_model")
tokenizer = tokenizer_class.from_pretrained(KEYWORD_MODEL_PATH+"prev_trained_model")
model,parameter = load_model(KEYWORD_MODEL_PATH)