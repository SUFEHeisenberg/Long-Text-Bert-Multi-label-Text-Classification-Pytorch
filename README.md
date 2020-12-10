# Long-Text-Bert-Multi-label-Text-Classification-Pytorch
基于Pytorch预训练模型上的中文长文本多标签分类。

[BERT](https://github.com/google-research/bert), [ERNIE](https://github.com/PaddlePaddle/ERNIE/tree/repro), [RoBERTa](https://github.com/ymcui/Chinese-BERT-wwm),[RBT3](https://github.com/ymcui/Chinese-BERT-wwm), [RBTL3](https://github.com/ymcui/Chinese-BERT-wwm), [NEZHA](https://github.com/huawei-noah/Pretrained-Language-Model), [ZEN](https://github.com/sinovation/ZEN)皆可。可自行下载pytroch版本模型。

## 环境
机器：Quadro P2000

python 3.7  
pytorch 1.6.0+cu101  

## 数据格式

data文件下目录格式为：

- data
  - class.txt（类别）
  - dev.txt
  - test.txt
  - train.txt
  - vocab.pkl（字典）
- log
- saved_dict（保存模型参数）

具体文本与标签排列方式为text\tlabel_1,label2,...。举例如下：

```
新华社北京12月9日电（记者汪子旭）12月8日，中国银保监会主席郭树清在2020年新加坡金融科技节上发表题为“金融科技发展、挑战与监管”的主题演讲。	1,3
```

例中ChnSentiCorp本身非多标签数据集，在原本0,1标签后都加了2这个标签。

## Pretrained Language Model
bert模型放在 bert_pretain目录下，ERNIE模型放在ERNIE_pretrain目录下，每个目录下都是三个文件：
 - pytorch_model.bin  
 - bert_config.json  
 - vocab.txt  

其中ZEN多了ngrams.txt。

## 使用说明

```python
# 训练并测试：
python run.py --model bert
python run.py --model ERNIE
python run.py --model roberta_wwm
python run.py --model ZEN_pretrain
```

### 参数说明
```python
class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt' # 训练集
        self.dev_path = dataset + '/data/dev.txt'  # 验证集
        self.test_path = dataset + '/data/test.txt' # 测试集
        self.class_list = [x.strip() for x in open(dataset +'/data/class.txt',encoding='UTF-8').readlines()]# 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt' # 模型训练结果
        self.output_path = 'output/ChnSentiCorp'# 评估结果输出路径
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 设备
        self.require_improvement = 1000# 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)# 类别数
        self.num_epochs = 5# epoch数
        self.batch_size = 16# mini-batch大小
        self.pad_size = 20# 每句话处理成的长度(短填长切)
        self.learning_rate = 3e-5# 学习率
        self.threshold = 0.5 #模型预测多标签概率阈值
        self.bert_path = './pretrained_models/bert_pretrain'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.trunc_medium = -2 #长文本截断形式:
        					   #-2表示从文本头部选取pad_size个token
            				   #-1表示从文本尾部选取pad_size个token
            				   #0表示从文本头、尾各选取1/2*pad_size个token
            				   #1表示从文本头、中、尾部选取1/3*pad_size个token
```

## 评价方法

```python
#以CAIL_2018法条预测为例：
#===============clf_evaluation=====================
This is result of bert is:
{'hamming_loss': 0.009702766991278183, 'micro_f1': 0.10417644298451431, 'micro_precision': 0.9487179487179487, 'micro_recall': 0.055114200595829194, 'instance_f1': 0.0745037645448323, 'instance_precision': 0.11396303901437371, 'instance_recall': 0.055527036276522924}
when the bs = 16 pad_size = 300  lr = 3e-05 epoch = 5 training time= 0:13:17
#===============classification_report=====================
This is result of RBTL3 is:#部分标签
                                  precision    recall  f1-score   support

                            妨害公务     0.0000    0.0000    0.0000        86
                            寻衅滋事     0.0000    0.0000    0.0000       137
                         盗窃、侮辱尸体     0.0000    0.0000    0.0000         1
                          危险物品肇事     0.0000    0.0000    0.0000         0

                       micro avg     0.8826    0.3640    0.5154      4028
                       macro avg     0.0396    0.0348    0.0355      4028
                    weighted avg     0.4380    0.3640    0.3821      4028
                     samples avg     0.5167    0.3708    0.4181      4028

when the bs = 16 pad_size = 512  lr = 3e-05 epoch = 5 training time= 0:08:52
#===============multi-label confusion matrix(One-vs-Rest)=====================
The confusion matrix for Label "妨害公务" is:
[[1862    0]
 [  86    0]]
The confusion matrix for Label "寻衅滋事" is:
[[1811    0]
 [ 137    0]]
The confusion matrix for Label "盗窃、侮辱尸体" is:
[[1947    0]
 [   1    0]]

```

## To-Do

 - Config类放到 run.py中，提高解耦性；
 - albert, xlnet, gpt-2的适配；
 - 长文本的Hierarchy分句聚合文本表示。


## 对应论文
[1] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding  

[2] ERNIE: Enhanced Representation through Knowledge Integration  

[3] Pre-Training with Whole Word Masking for Chinese BERT

[4] NEZHA: Neural Contextualized Representation for Chinese Language Understanding

[5] ZEN: Pre-training Chinese Text Encoder Enhanced by N-gram Representations
