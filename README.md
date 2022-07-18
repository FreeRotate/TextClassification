## PyTorch 文本分类

### 包含的模型如下

TextCNN

LSTM BiLSTM

GRU BiGRU

TransformerEncoder



**环境**
```text
torch==1.10.1+cu113
argparse==1.4.0
numpy==1.22.3
sklearn==0.0
scikit-learn==1.0.2
pandas==1.1.1
tqdm==4.62.3
```

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

#### 更换模型请在main.py中修改default部分，例如GRU/BiGRU模型，使用下面代码
```python
parser.add_argument('--model', type=str, default='GRU', help='CNN, GRU, LSTM, TransformerEncoder')
```

### 这是我第一个开源的项目，欢迎star