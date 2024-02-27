import requests
import json

url = "http://172.16.139.194:9000/predict"

data = {
	"data": [
	"人行前置机A到人行2900A异常，172.18.81.37 BPH001072 人行前置机B到人行2900A异常", 
	"人行前置机A到人行2900A异常",
	"昆明分行串口服务器通信状态:断线报警",
	"128.20.143.215runmqsc-status on 5022900001_6:116runmqsc '5022900001_6' status is 116",
	"合肥支行串口服务器通信状态:断线报警",
	"2023年10月18日 16:12:26,东亚银行张江数据中心,二层主机房漏水监测,漏水定位,机房漏水报警,状态告警,值为17.115",
	"undefined172.18.80.2 FXPS001034 人行PMTS前置异常"
	]
}

response = requests.post(url, json=data)
print(response.json())
print("status_code", response.status_code)