import openai
import json
import pandas as pd
from tqdm import tqdm
from ipywidgets import widget

openai.api_base = "https://api.theb.ai/v1"
# openai.api_base = "https://api.baizhi.ai/v1"
openai.api_key = "sk-XsgkR7jwUtRV5yBxahebVGhlQi5BSQThqx4N8awhpPlYFUY3"


def gpt_request(messages):
    response = openai.ChatCompletion.create(
        model="gpt-4-turbo",
        messages=messages,
        stream=False,
        model_params={
            "temperature": 0.7
        }
    )
    response_json = response.choices[0].message
    result_str = json.dumps(response_json, ensure_ascii=False)
    result_json = json.loads(result_str)
    return result_json['content']

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as config_json:
        config_data = json.load(config_json)
        labels = tuple(config_data.keys())
    return labels

labels = load_config('./config/标签类别映射/config40.json')
def get_prompt(input_alarm):
    # Prompt:
    prompt1 = """
    告警标签是对告警数据的归纳总结和分类,我将输入样例，按照告警数据的序号返回标签，请学习这种为告警打标签的模式：
    1 531企业网银系统（境内）2019-11-01 13~13服务进程:182.251.48.53_CEBSSvr02,1小时交易量低持续52次,当前值0阈值1   
    2 19:38”19:39 业务返回码:PB521099(付款行填写的其他快捷支付失败说明)，1分钟交易量高，闻值:100笔，实际值:421笔  
    3 DTTDCGF NUCPNETepcc101001.0111 busTransCount 300秒内无数据推送]          
    以上三条样例的标签为：应用系统业务告警：[2023-04-23T18:51:00.000+08:00]财资管理系统_财资web（江北）的响应时间发生突增异常，当前值为:6.869377,高于动态基线值:1.884699,同时对应交易量和响应率发生突变异常，请注意! 
    1. 交易量持续低
    2. 交易量突增
    3. 交易量无推送
    请学习这种为告警打标签的方法，为后续告警打上正确的标签，要求只输出告警序号和标签，如（1. 交易量持续低），无需输出告警数据：
    """
    prompt2 = """
    请为下列告警打上正确的标签,告警数据: {};
    要求:1.输出格式为：告警标签(比如：交易量持续低)；2.仅输出告警数据和标签，不要输出其他任何内容。
    """.format(input_alarm)
    prompt3 = """3.标签范围包含在{}中:""".format(labels)
    return prompt1, prompt2, prompt3

def gpt_demo(input_alarm, prompt_nums):
    prompt1, prompt2, prompt3 = get_prompt(input_alarm)
    messages = [
            {"role": "system", "content": "您是一位智能运维专家，告警数据有着深刻的理解，能够通过告警标签生成不同类型的告警数据，下面请解决运维领域告警数据相关的问题:"},
            {"role": "user", "content": prompt1},
            {'role': "user", "content": prompt2},
            {'role': "user", "content": prompt3}
        ]
    result = gpt_request(messages[:prompt_nums])
    return result

def map_label(file_path):
    df = pd.read_csv(file_path)
    with_label = []
    without_label = []
    for i, per_alarm in tqdm(enumerate(df.content)):
        with_result = gpt_demo(input_alarm=per_alarm, prompt_nums=4)
        without_result = gpt_demo(input_alarm=per_alarm, prompt_nums=3)
        with_label.append(with_result)
        without_label.append(without_result)
        print("已完成调用次数:", i)
        print("告警数据:{}, 有范围标签{}, 不设定范围标签{}".format(per_alarm, with_result, without_result))
    df['label-in'] = with_label
    df['label-not-in'] = without_label
    return df


if __name__ == "__main__":
    
    contain = 3
    text_ = "基金代销响应时间连续2分钟大于200ms。"
    result = gpt_demo(input_alarm=text_, prompt_nums=contain)