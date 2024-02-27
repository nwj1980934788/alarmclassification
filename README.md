# 基于BERT模型的告警标签分类算法
## 1. 简介

**告警标签**是对告警数据的归纳、总结与简化，能够简单直观的反映告警信息。将告警数据简化成标签，可以降低告警风暴、过滤普通告警，不同的标签结合能够清晰反映故障发生的机理和影响范围，且具有通用、复刻性，从而能够更方便的发现根因、解决故障。告警标签分类算法，结合大模型技术，通过有监督的训练Bert模型，大小模型配合，高效精确的实现告警数据的分类打标，为故障的排查提供了方便高效的解决方案。

## 2. 运行流程

![相对路径](img/WechatIMG735.jpg)

## 3. 资源包

```
|--lab
    |--config
        ｜--标签分布映射
        ｜--标签类别映射
    |--dataset
        ｜--大模型标签
        ｜--聚类模版
        ｜--数据增强
        ｜--推理输出
        ｜--训练数据
        ｜--原始告警
    |--img
        ｜--llm.jpg
        ｜--WechatlMG735.jpg
    |--models
        ｜--model.onnx
    |--pretrain_models
        ｜--bert-base-chinese
    |--ASParser.py
    |--data_enhancement.py
    |--findAllDelimiters.py
    |--llm_api.py
    |--onnxInference.py
    |--sim.py
    |--train_model.py
    |--utils
    |--Part1-告警聚类.ipynb
    |--Part2-模型标签.ipynb
    |--Part3-数据扩展.ipynb
    |--Part4-训练模型.ipynb
    |--Part5-模型验证.ipynb
    |--Part6-场景验证.ipynb
```

各种模型用途

![相对路径](img/llm.jpg)

## 4. 环境

(1)新建项目路径，上传资源

- 新建一个文件夹D:\intelGiagn,上传miniconda.exe, dsw-win.zip  (基础资源包),lab.zip  (程序包)到D:\intelGiagn

(2)安装Miniconda3

- 双击安装Miniconda3-latest-Windows-x86.exe，默认下一步安装即可

(3)释放dsw基础资源包

- 上传dsw-win.zip 到路径 ～Miniconda3/envs/目录下，并解压，得dsw-win

(4)打开cmd命令窗口执行

- 激活python环境
  ``conda activate dsw-win``
- 启动jupyter
  ``jupyter lab --no-browser --ip=192.xx.xx.xx --port=18088 (自定义) --allow-root``
- 随后即可跳转到网页jupyter notebook
