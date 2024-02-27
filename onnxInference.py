from transformers import BertTokenizer
from utils import labels
import onnxruntime
import torch
import numpy as np
import os
import json
import time
from tqdm import tqdm
import pandas as pd

tokenizer = BertTokenizer.from_pretrained("./pretrain_models/bert-base-chinese")
provider = onnxruntime.get_available_providers()[1 if onnxruntime.get_device() == 'GPU' else 0]

def inferance(texts, model_name):
    tokens = tokenizer(
        texts, 
        return_tensors="np", 
        padding="max_length", 
        truncation=True, 
        max_length=64
    )
    ort_session = onnxruntime.InferenceSession(model_name, providers=[provider])                        # 不应该放到循环里
    ort_inputs = {
        'input_ids': tokens['input_ids'].astype(np.int64),
        'attention_mask': tokens['attention_mask'].astype(np.int64)
    }
#     print("ort_inputs 的维度：", ort_inputs)
    pred_logits = ort_session.run(['output'], ort_inputs)[0]
    pred_probs = torch.nn.functional.softmax(torch.tensor(pred_logits), dim=1)
    result = torch.max(pred_probs, dim=1)
    indexs = result.indices.tolist()
    probs = result.values.tolist()
    return indexs, probs

def main_inferance(texts, config_name, model_name):
    label_list = []
    LABELS = labels(config_name)
    label_dic = {v:k for k, v in LABELS.items()}
    inferance_list, pred_probs = inferance(texts, model_name)
    for i in inferance_list:
        label_list.append(label_dic[i])
    return label_list, pred_probs

def batch_inference(path_inference, batch_size, config_name, model_path):
    inference_df = pd.read_csv(path_inference)
    contents = inference_df.content.values.tolist()
    labels = []
    probs = []
    start_time = time.time()
    for i in tqdm(range(0, len(contents), batch_size)):
        input_contents = contents[i:i+batch_size]
        label_list, pred_probs = main_inferance(input_contents, config_name, model_path)
        labels += label_list
        probs += pred_probs
    print("Used Time is : ", time.time() - start_time)
    inference_df['model_label'] = labels
    inference_df['model_prob'] = probs
    # 计算准确度
    T = []
    origin_labels = inference_df.label.values.tolist()
    for i in range(len(origin_labels)):
        if origin_labels[i] == labels[i]:
            T.append(1)
        else:
            T.append(0)
    inference_df['T'] = T
    return inference_df

if __name__ == "__main__":
    
    texts = ["带宽超出接口GigabitEthernet0/0/2-R_GZ_XL_C2911_03_F0/0/0_CUC_。当前流入流量是4.392 Mbps (配置的流入速度是4.096 Mbps) ，流出流量是376 Kbps (配置的流出速度是4.096 Mbps)。"]  # 添加更多文本样本
    config_name = "./config/标签类别映射/config75.json"
    model_name = "./models/test_model.onnx"
    label_list, pred_probs = main_inferance(texts, config_name, model_name)
    print(label_list)
    print(pred_probs)