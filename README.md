## PyTorch 文本分类

TextCNN

LSTM BiLSTM

GRU BiGRU

TransformerEncoder



**环境**


**使用方法**

1. 安装环境
```shell
pip install requirements.txt
```
2. 运行代码
```shell
python main.py
```
### 注意事项

####更换模型请在main.py代码中修改以下部分

***from model.<模型名称> import \<模型名称\>***

***model = \<模型名称\> (len(vocab), config).to(config.device)***
### 这是我第一个开源的项目，喜欢请多多star