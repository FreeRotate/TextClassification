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

#### 更换模型请在main.py中修改default部分，例如使用GRU/BiGRU模型，使用下面代码
```python
parser.add_argument('--model', type=str, default='GRU', help='CNN, GRU, LSTM, TransformerEncoder')
```

### 这是我第一个开源的项目，喜欢请多多star