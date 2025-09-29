# coding=utf-8


import sys

import json

import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from transformers import BertTokenizer, BertPreTrainedModel, BertModel

from typing import List, Tuple



def load_srl_data(data_path, frame_filename) :
    with open(frame_filename, 'r', encoding='cp949') as f :
        framefiles = json.load(f)
    temp = []
    data = []
    with open(data_path, 'r') as f :
        for line in f :
            line = line.replace('\r\n', '')
            line = line.replace('\n', '')
            if len(line) < 1 :
                #print(line)
                #print(temp)
                sentence_org = temp[0].strip()
                text = []
                v_position = [] # 동사 위치 index
                verbs = []
                for ttt in temp[1].split('\t') :
                    if ttt != '_' :
                        verbs.append(ttt)
                for i, ttt in enumerate(temp[2:]) :
                    text.append(ttt.split('\t')[0])
                    if ttt.split('\t')[1] != '_' :
                        v_position.append(i)
                for ii, (v, idx) in enumerate(zip(verbs, v_position)) :
                    sentence = []
                    v_org = ""
                    for i, word in enumerate(text) :
                        if i == idx :
                            sentence.append("<predicate>")
                            sentence.append(word)
                            sentence.append("</predicate>")
                            v_org = word
                        else :
                            sentence.append(word)
                    gold_conll = ""
                    for ddd in temp[2:] :
                        gold_conll = gold_conll + ddd.split('\t')[0] + "\t" + ddd.split('\t')[ii+2] + "\n"
                    gold_conll = gold_conll.strip()
                    #print(gold_conll)
                    sentence = " ".join(sentence)
                    roles = []
                    if v in framefiles :
                        for rr in framefiles[v] :
                            roles.append(rr)
                    roles = "\n".join(roles)
                    #data.append([sentence_org, sentence, v_org, v, roles, gold_conll])
                    data.append({"sentence_org":sentence_org, "sentence":sentence, "v_org":v_org, "v":v, "roles":roles, "gold":gold_conll})
                temp = []
            else :
                temp.append(line)
    return data




class BertDB :
    def __init__(self, bert_model_name: str = "klue/bert-base", use_gpu: bool = True, metric: str = "euclidean", extraction_method: str = "last"):
        #self.tokenizer = BertTokenizer.from_pretrained("klue/bert-base", do_basic_tokenize=True)
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_basic_tokenize=True)
        self.bert_model = BertModel.from_pretrained(bert_model_name)
        self.vector_dim = self.bert_model.config.hidden_size
        self.num_layers = self.bert_model.config.num_hidden_layers
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.metric = metric
        self.extraction_method = extraction_method
        if self.use_gpu:
            self.bert_model = self.bert_model.cuda()
        self.data = []
        self.inv_covariance_matrices = [0]
    
    
    def encode_sentence(self, sentence1: str) -> np.ndarray:
        inputs = self.tokenizer(sentence1, return_tensors="pt", padding=True, truncation=True, max_length=512)
        #print(inputs)
        #for ttttt in inputs['input_ids'][0] :
        #    print(self.tokenizer.ids_to_tokens[ttttt.item()])
        #exit(1)
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.bert_model(**inputs, output_hidden_states=True)
        
        #vector = self.extract_bert_vector(outputs.hidden_states)
        '''
        if self.extraction_method == "last" :
            vector = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        elif self.extraction_method == "avr_all" :
            vector = np.concatenate([hidden_state[:, 0, :].cpu().numpy().flatten() for hidden_state in outputs.hidden_states[1:]])
        elif self.extraction_method == "norm_all" :
            #vector = np.concatenate([torch.tanh(hidden_state[:, 0, :]).cpu().numpy().flatten() for hidden_state in outputs.hidden_states[1:]])
            vector = np.concatenate([torch.tanh(hidden_state[:, 0, :]).cpu().numpy().flatten() if (ii<self.num_layers-1) else hidden_state[:, 0, :].cpu().numpy().flatten() for ii,hidden_state in enumerate(outputs.hidden_states[1:])])
        '''
        if self.extraction_method == "last" :
            vector = outputs.last_hidden_state[:, 0, :].flatten()
        elif self.extraction_method == "avr_all" :
            vector = torch.concatenate([hidden_state[:, 0, :].flatten() for hidden_state in outputs.hidden_states[1:]])
        elif self.extraction_method == "norm_all" :
            vector = torch.concatenate([torch.tanh(hidden_state[:, 0, :]).flatten() if (ii<self.num_layers-1) else hidden_state[:, 0, :].flatten() for ii,hidden_state in enumerate(outputs.hidden_states[1:])])
        
        if self.metric == "cosine" :
            vector = vector / torch.linalg.norm(vector)
        
        return vector

    def add_item(self, sentence1: str, index: int):
        vector = self.encode_sentence(sentence1)
        self.data.append((vector, index))

    def update_covariance_matrix(self):
        if self.metric == "mahalanobis" :
            #all_vectors = np.array([vec for vec, _ in self.data])
            all_vectors = []
            for vec, _ in self.data :
                all_vectors.append(vec)
            if self.extraction_method == "last" :
                self.inv_covariance_matrices = [0]
                all_vectors = torch.stack(all_vectors)
                covariance_matrices = torch.cov(all_vectors.T)
                self.inv_covariance_matrices[0] = torch.linalg.pinv(covariance_matrices)
            else:
                #self.covariance_matrices = []
                self.inv_covariance_matrices = []
                for i in range(self.num_layers):
                    start = i * self.vector_dim
                    end = (i + 1) * self.vector_dim
                    #layer_vectors = all_vectors[:, start:end]
                    all_vectors2 = []
                    for vv in all_vectors :
                        all_vectors2.append(vv[start:end])
                    layer_vectors = torch.stack(all_vectors2)
                    cov_matrix = torch.cov(layer_vectors.T)
                    #self.covariance_matrices.append(cov_matrix) #필요없는거
                    #self.inv_covariance_matrices.append(np.linalg.pinv(cov_matrix))
                    self.inv_covariance_matrices.append(torch.linalg.pinv(cov_matrix))

    def calculate_distance(self, xx: np.ndarray, yy: np.ndarray) -> float:
        if self.metric == "euclidean" :
            #x = torch.Tensor(xx).to("cuda")
            #y = torch.Tensor(yy).to("cuda")
            x = xx
            y = yy
            if self.extraction_method == "last" :
                return torch.linalg.norm(x - y).item()
            else:
                return np.mean([torch.linalg.norm(x[i*self.vector_dim:(i+1)*self.vector_dim] - 
                                               y[i*self.vector_dim:(i+1)*self.vector_dim]).item()
                                for i in range(self.num_layers)])
        elif self.metric == "cosine" :
            x = xx.flatten()
            y = yy.flatten()
            if self.extraction_method == "last" :
                return (1 - torch.dot(x, y) / (torch.linalg.norm(x) * torch.linalg.norm(y))).item()
            else:
                return np.mean([(1 - torch.dot(x[i*self.vector_dim:(i+1)*self.vector_dim], 
                                           y[i*self.vector_dim:(i+1)*self.vector_dim]) / 
                                (torch.linalg.norm(x[i*self.vector_dim:(i+1)*self.vector_dim]) * 
                                 torch.linalg.norm(y[i*self.vector_dim:(i+1)*self.vector_dim]))).item()
                                for i in range(self.num_layers)])
        elif self.metric == "mahalanobis" :
            x = xx
            y = yy
            if self.extraction_method == "last" :
                diff = x - y
                return torch.sqrt((diff @ self.inv_covariance_matrices[0] @ diff.T)).item()
            else:
                distances = []
                for i in range(self.num_layers):
                    start = i * self.vector_dim
                    end = (i + 1) * self.vector_dim
                    x_layer = x[start:end]
                    y_layer = y[start:end]
                    diff = x_layer - y_layer
                    #distance = torch.sqrt(diff.dot(self.inv_covariance_matrices[i]).dot(diff.T)).item()
                    distance = torch.sqrt((diff @ self.inv_covariance_matrices[i] @ diff.T)).item()
                    distances.append(distance)
                    #print(" ", i,":", distance)
                return np.mean(distances)


    def search(self, query_sentence1: str, k: int = 1) -> List[Tuple[float, int]]:
        query_vector = self.encode_sentence(query_sentence1)
        
        # 전체 데이터베이스에 대해 거리 계산
        distances = []
        for stored_vector, index in self.data:
            distance = self.calculate_distance(query_vector, stored_vector)
            distances.append((distance, index))
        
        # 거리에 따라 정렬하고 상위 k개 결과 반환
        distances.sort(key=lambda x: x[0])
        return distances[:k]









