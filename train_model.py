from transformers import BertTokenizer, BertModel
import warnings
from torch import nn
from utils import load_config
from torch.utils.data import DataLoader
import numpy as np
from torch.optim import Adam
import torch
from tqdm import tqdm
from loguru import logger
import time
import pandas as pd
import json

np.random.seed(42)
warnings.filterwarnings("ignore")
bert_chinese_model = "./pretrain_models/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(bert_chinese_model)
use_mps = torch.backends.mps.is_available()
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")

    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, df, LABELS):
        self.LABELS = LABELS
        self.labels = [self.LABELS[l] for l in df['label']]
        self.texts = [tokenizer(str(text),
                                padding='max_length',
                                max_length=64,
                                truncation=True,
                                return_tensors='pt')
                      for text in df['content']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        return self.texts[idx]
        
    def __getitem__(self, idx):
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_texts, batch_y


class BertClassifier(nn.Module):
    def __init__(self, dropout, out_dims):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_chinese_model)
        for param in self.bert.encoder.layer[:6].parameters():    
            param.requires_grad = False
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, out_dims)
        self.relu = nn.ReLU()

    def forward(self, input_id, mask):
        _, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        final_layer = self.relu(linear_output)
        return final_layer
    

def train(model, train_data, val_data, learning_rate, epochs, batch_size, save_path, num_classes, LABELS, onnx_path):
    train, val = Dataset(train_data, LABELS), Dataset(val_data, LABELS)
    train_dataloader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    BEST_VAL_ACCURACY = 0.0
    best_model = None
    for epoch_num in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        for train_input, train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc
            model.zero_grad()         
            batch_loss.backward()      
            optimizer.step()          
            total_acc_val = 0
            total_loss_val = 0
            with torch.no_grad():
                for val_input, val_label in val_dataloader:
                    val_label = val_label.to(device)
                    mask = val_input['attention_mask'].to(device)           
                    input_id = val_input['input_ids'].squeeze(1).to(device)  
                    output = model(input_id, mask)                           
                    batch_loss = criterion(output, val_label.long())
                    total_loss_val += batch_loss.item()                      
                    acc = (output.argmax(dim=1) == val_label).sum().item()  
                    total_acc_val += acc
                logger.info("Epochs: {} Train Loss: {}, Train Accuracy: {}, Val Loss: {}, Val Accuracy: {}.".format(
                      epoch_num + 1,
                      round(total_loss_train / len(train_data), 3), round(total_acc_train / len(train_data), 3),
                      round(total_loss_val / len(val_data), 3), round(total_acc_val / len(val_data), 3)))
            if round(total_acc_val / len(val_data), 3) > BEST_VAL_ACCURACY:
                BEST_VAL_ACCURACY = round(total_acc_val / len(val_data), 3)
                best_model = model
                # 保存模型
                #torch.save(model.state_dict(), save_path)
                logger.warning("Best validation accuracy updated.")
    if best_model is not None:
        text = "带宽超出接口GigabitEthernet0/0/2-R_GZ_XL_C2911_03_F0/0/0_CUC_8M_510JPW12465754配置的限制。当前流入流量是4.392 Mbps (配置的流入速度是4.096 Mbps) ，流出流量是376 Kbps (配置的流出速度是4.096 Mbps)。"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64).to(device)
        model = model.eval()
        outputs = model(inputs['input_ids'], inputs['attention_mask'])
        torch.onnx.export(
            model, 
            (inputs['input_ids'], inputs['attention_mask']),
            onnx_path, 
            verbose=True,
            input_names=["input_ids", "attention_mask"],
            output_names = ["output"],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                # 'output': {0: 'batch_size', 1: 'sequence'}
            }, 
        )

def evaluate(model, test_data, batch_size, LABELS):
    test = Dataset(test_data, LABELS)
    test_dataloader = DataLoader(test, batch_size=batch_size)
    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            output = model(input_id, mask)
            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc
    print(f'Test Accuracy: {total_acc_test / len(test_data): .3f}') 