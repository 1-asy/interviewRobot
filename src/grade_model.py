import torch.nn.functional as F # pytorch 激活函数的类
from torch import nn # 构建模型和优化器
import torch
import pickle as pk
import numpy as np

class Grade(nn.Module):
    def __init__(self, parameter):
        super(Grade, self).__init__()
        embedding_dim = parameter['embedding_dim']
        hidden_size = parameter['hidden_size']
        num_layers = parameter['num_layers']
        dropout = parameter['dropout']
        word_size = parameter['word_size']
        self.embedding = nn.Embedding(word_size, embedding_dim, padding_idx=0)
        self.lstm_q = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.lstm_a = nn.LSTM(embedding_dim, hidden_size, num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, q, a1,a2 = None):
        q_emd = self.embedding(q)
        q_emd,(h, c)= self.lstm_q(q_emd)
        q_emd = torch.max(q_emd,1)[0]
        a1_emd = self.embedding(a1)
        a1_emd,(h, c)= self.lstm_a(a1_emd)
        a1_emd = torch.max(a1_emd,1)[0]
        if a2 is not None:
            a2_emd = self.embedding(a2)
            a2_emd,(h, c)= self.lstm_a(a2_emd)
            a2_emd = torch.max(a2_emd,1)[0]
            return q_emd,a1_emd,a2_emd
        return F.cosine_similarity(q_emd,a1_emd,1,1e-8)
    
def list2torch(a):
    return torch.from_numpy(np.array(a)).long().to(parameter['cuda'])

def load_model(root_path = './'):
    parameter = pk.load(open(root_path+'parameter.pkl','rb'))
    model = Grade(parameter).to(parameter['cuda'])
    model.load_state_dict(torch.load(root_path+'grade.h5'))
    model.eval()
    return model,parameter

def grade_predict(q,a):
    q = list(q)
    a = list(a)
    q_cut = []
    for i in q:
        if i in parameter['word2id']:
            q_cut.append(parameter['word2id'][i])
        else:
            q_cut.append(parameter['word2id']['<UNK>'])
    a_cut = []
    for i in a:
        if i in parameter['word2id']:
            a_cut.append(parameter['word2id'][i])
        else:
            a_cut.append(parameter['word2id']['<UNK>'])
    # print(q_cut,a_cut)
    q_cut,a_cut = [q_cut[:parameter['max_len']]],[a_cut[:parameter['max_len']]]
    prob = model(list2torch(q_cut),list2torch(a_cut))
    # print(prob)
    return prob.cpu().item()

import sys
sys.path.append('../')
from path_config import GRADE_MODEL_PATH
model,parameter = load_model(GRADE_MODEL_PATH)



