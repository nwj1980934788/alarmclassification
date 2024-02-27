from train_model import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import jieba
import random
from tqdm import tqdm
import re
import os
import json
from utils import load_config
from sklearn.utils import resample


config_count_path = "./config/标签分布映射/origin_count_config.json"
def dump_config(save_file, file_name):
    with open(file_name, 'w') as config_json:
        return json.dump(save_file, config_json)

def save_config_map(df):
    config_file = dict(zip(list(df['label'].value_counts().keys()), [i for i in range(len(df['label'].unique()))]))
    # 将可能的int64类型转换为内置的int类型
    config_file = {key: int(value) for key, value in config_file.items()}
    file_name = "./config/标签类别映射/config{}.json".format(len(config_file))
    dump_config(config_file, file_name)

def save_config(df):
    config_file = dict(zip(list(df['label'].value_counts().keys()), list(df['label'].value_counts().values)))
    # 将可能的int64类型转换为内置的int类型
    config_file = {key: int(value) for key, value in config_file.items()}
    dump_config(config_file, config_count_path)

def read_df(alarm_path, temp_path):
    df = pd.read_csv(alarm_path)
    df_extract_temp_with_label = pd.read_csv(temp_path)
    event_label_map = dict(zip(
        list(df_extract_temp_with_label.EventId.values), 
        list(df_extract_temp_with_label.label.values)
    ))
    df['label'] = df['EventId'].map(event_label_map)
    # df = df.rename(columns={'Content':'content'})
    new_df = df[['content', 'label', 'EventId']]
    new_df = new_df.dropna()
    new_df.reset_index(drop=True, inplace=True)
    return new_df
             
def seaborn_plot(width, height):
    plt.rcParams['font.sans-serif'] = 'SimHei' 
    plt.rcParams['axes.unicode_minus'] = False
    map1 = load_config(config_count_path)
    data = {'Category': list(map1.keys()),
            'Count': list(map1.values())}
    df = pd.DataFrame(data)
    df = df.sort_values(by='Count', ascending=False)
    fig, ax = plt.subplots(figsize=(width, height))
    sns.barplot(x='Count', y='Category', data=df, palette='viridis', ax=ax)
    for p in ax.patches:
        ax.annotate(f'{p.get_width():.0f}', (p.get_width(), p.get_y() + p.get_height() / 2),
                    ha='left', va='center', xytext=(5, 0), textcoords='offset points')
    ax.set_xlabel('数量', fontsize=12)
    ax.set_ylabel('类别', fontsize=12)
    ax.set_title('类别数量统计（排序）', fontsize=12)
    plt.show()

def add_save_all_alarm(temp_alarm_path, all_alarm_path, save_alarm_path, col1, col2):
    df = pd.read_csv(all_alarm_path, encoding="utf8") 
    temp_df = pd.read_csv(temp_alarm_path, encoding="utf8")
    EventId_label_map = dict(zip(temp_df[col1].values.tolist(), temp_df[col2].values.tolist()))
    df['label'] = df[col1].map(EventId_label_map)
    new_df = df[['content', 'label']]
    new_df = new_df.dropna()
    new_df.reset_index(drop=True, inplace=True)
    new_df.to_csv(save_alarm_path, index=False)
    
def resample_lower(negative_samples, n_samples):
    # 下采样
    # n_samples: 采样样本数
    # label_class: 采样标签类别
    negative_samples_resampled = resample(
        negative_samples, replace=False, n_samples=n_samples, random_state=42
    )
    return negative_samples_resampled

def main_lower(df, resample_count_threshold):
    # resample_count_threshold: 下采样类别数量阈值，超过这个值即进行下采样
    label_count = df.label.value_counts().to_dict()
    filter_label = {k: v for k, v in label_count.items() if v > resample_count_threshold}
    for select_label in tqdm(filter_label):
        select_df = df[df.label == select_label]
        print("数据总量为:{} 选择下采样的数据量为:{}".format(df.shape, select_df.shape))
        df.drop(select_df.index, axis=0, inplace=True)
        lower_sample = resample_lower(select_df, n_samples=resample_count_threshold)
        print("下采样样本数量大小为:", lower_sample.shape)
        print("去除下采样类别后数据量:", df.shape)
        df = pd.concat([df, lower_sample], axis=0, ignore_index=False)
        print("合并之后总数据量为:", df.shape)
        print("*"*50)
    df.reset_index(drop=True, inplace=True)
    return df

def random_add_sample(df2, num_copies):
    # 随机过采样 + 添加噪音
    copied_data = df2.copy()
    for _ in range(num_copies):
        noise = np.random.normal(0, 0.1, len(df2['content']))
        copied_sample = df2.copy()
        copied_sample['content'] = [
            str(text) + str(noise_val) 
            for text, noise_val in zip(df2['content'], noise)
        ]
        copied_data = pd.concat([copied_data, copied_sample], ignore_index=True)
    return copied_data

def random_drop(df3, drop_prob):
    # 随机删除
    for i in range(len(df3['content'])):
        tokens = jieba.lcut(df3['content'][i])
        tokens_after_drop = [token for token in tokens if random.uniform(0, 1) > drop_prob]
        df3.loc[i, 'content'] = "".join(tokens_after_drop)
    return df3

def select_label_df(df1, label):
    select_df = df1[df1.label == label]
    df1.drop(select_df.index, axis=0, inplace=True)
    return df1, select_df
 
# [左区间，右区间，复制倍数]
# 比如[[0, 100, 20], [100, 200, 10], [200, 300, 6]]
# 表示：类别样本在0~100之间，复制20次，类别样本在100~200之间，复制10次...
# 具体如何分桶、设置，需观察上图，目标是---使得类别基本平衡即可
# (left, right] 左开右闭区间

# lrn = [
#     [0, 1, 800], [1, 2, 400], [2, 3, 266], [3,4,200], [4,5,160]
# ]
# def main(df, dic, left_right_num_copies):

#     add_df_list = []
#     for label, count in tqdm(dic.items()):
#         for left, right, num_copies in left_right_num_copies:
#             if left < count <= right:
#                 print("上采样前df.shape为: {}".format(df.shape))
#                 df, select_df = select_label_df(df, label)
#                 print("标签:{} 的count为:{} 上采样后df.shape为: {}".format(label, count, df.shape))
#                 print("-" * 50)
#                 copied_data = random_add_sample(select_df, num_copies)
#                 print('增加数据量:-----------------', copied_data.shape[0])
#                 copied_data_drop = random_drop(copied_data, drop_prob=0.05)
#                 add_df_list.append(copied_data_drop)
#     add_df_list.append(df)
#     new_df = pd.concat(add_df_list, axis=0, ignore_index=False)
#     print(new_df.shape)
#     return new_df

def main(df, dic, resample_count_threshold):
    add_df_list = []
    for label, count in tqdm(dic.items()):
        if resample_count_threshold / count == 1.0:
            continue
        num_copies = int(resample_count_threshold / count)
        
        print("上采样前df.shape为: {}".format(df.shape))
        df, select_df = select_label_df(df, label)
        print("标签:{} 的count为:{} 上采样后df.shape为: {}".format(label, count, df.shape))
        print("-" * 50)
        copied_data = random_add_sample(select_df, num_copies)
        print('增加数据量:-----------------', copied_data.shape[0])
        copied_data_drop = random_drop(copied_data, drop_prob=0.05)
        add_df_list.append(copied_data_drop)
    add_df_list.append(df)
    new_df = pd.concat(add_df_list, axis=0, ignore_index=False)
    print(new_df.shape)
    return new_df

def random_exchange_word(dataframe):
    # 随机交换句子中两个词
    def is_start_with_chinese(word):
        if not word:
            return False
        first_char_code = ord(word[0])
        if 0x4e00 <= first_char_code <= 0x9fff:
            return True 
        else:
            return False
    def contains_chinese(text):
        pattern = re.compile(r'[\u4e00-\u9fff]+')
        return bool(pattern.search(text))
    for i in range(len(dataframe['content'])):
        tokens = jieba.lcut(dataframe['content'][i])
        tokens = [t for t in tokens if t != ' ']
        if len(tokens) >= 2:
            swap_indices = random.sample(range(len(tokens)), min(2, len(tokens)))
            tokens[swap_indices[0]], tokens[swap_indices[1]] = tokens[swap_indices[1]], tokens[swap_indices[0]]
        if contains_chinese("".join(tokens)):
            dataframe.at[i, 'content'] = "".join(tokens)
        else:
            dataframe.at[i, 'content'] = "".join(tokens)
    return dataframe

def add_random_noise(sentence, noise_add, noise_level=0.1):
    # 加入随机噪音
    char_list = list(sentence)
    num_chars_to_replace = int(len(char_list) * noise_level)
    replace_indices = random.sample(range(len(char_list)), num_chars_to_replace)
    for index in replace_indices:
        char_list[index] = random.choice(noise_add)
    sentence_with_noise = ''.join(char_list)
    return sentence_with_noise

def add_label_alarm(df, label_dict, num_copies):
    label_list = list(label_dict.keys())
    label_df_list = []
    for label in tqdm(label_list):
        select_df = pd.DataFrame({'content':[label], 'label':[label]})
        copied_data = random_add_sample(select_df, num_copies=num_copies)
        # copied_data_drop = random_drop(copied_data, drop_prob=0.01)
        copied_data_exchange = random_exchange_word(copied_data)
        label_df_list.append(copied_data_exchange)
    result = pd.concat(label_df_list, axis=0, ignore_index=False)
    print(result.shape)
    df = pd.concat([df, result], axis=0, ignore_index=False)
    df = df.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    return df

def add_new_temp(df, content, label, num_copies, drop_prob):
    select_df = pd.DataFrame({'content': [content], 'label': [label]})
    copied_data = random_add_sample(select_df, num_copies=num_copies)
    print(select_df.shape, copied_data.shape)
    copied_data_drop = random_drop(copied_data, drop_prob=drop_prob)
    print("构造的新样本数量: ", copied_data_drop.shape)
    print("合并之前:", df.shape)
    df = pd.concat([df, copied_data_drop], axis=0, ignore_index=False)
    df.reset_index(drop=True, inplace=True)
    print("合并之后:", df.shape)
    return df

