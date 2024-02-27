from ASParser import running
import pandas as pd
import warnings
import time
import json
warnings.filterwarnings("ignore")


def read_and_save(path, save_path, content_name):
    origin_df = pd.read_csv(path)
    print("原始数据量为:", origin_df.shape)
    df = origin_df.drop_duplicates(subset=[content_name])
    df = df.rename(columns={'告警内容':'content'})
    print("安装告警内容去重后数据量为：", df.shape)
    df.to_csv(save_path, index=False, encoding="utf-8")
    
def parser(file_name, sim, save_path):
    structured, templateId = running(file_name, sim=sim, input_path=save_path)
    return structured, templateId

def save_(structured, templateId, save_struc, save_temp):
    structured.to_csv(save_struc, index=False)
    templateId.to_csv(save_temp, index=False)
    
def load_config(config_name):
    with open(config_name, 'r') as config_json:
        config_data = json.load(config_json)
    return config_data
    
def labels(config_name):
    LABELS = load_config(config_name)
    return LABELS