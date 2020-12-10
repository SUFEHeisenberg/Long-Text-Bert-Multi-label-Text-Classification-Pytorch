# coding: UTF-8
import torch
import os
from tqdm import tqdm
import time
from datetime import timedelta
from sklearn.preprocessing import MultiLabelBinarizer

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, pad_size=32, trunc_medium=0):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            for line in tqdm(f):
                lin = line.strip()
                if not lin:
                    continue
                content, label_str = lin.split('\t')
                token = config.tokenizer.tokenize(content)
                token = [CLS] + token
                seq_len = len(token)
                mask = []
                token_ids = config.tokenizer.convert_tokens_to_ids(token)

                # which should be tested:
                if pad_size:
                    if len(token) < pad_size:
                        mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                        token_ids += ([0] * (pad_size - len(token)))
                    else:
                        if trunc_medium == -2:
                            mask = [1] * pad_size
                            token_ids = token_ids[0:pad_size]
                        elif trunc_medium == -1:
                            mask = [1] * pad_size
                            token_ids = token_ids[-pad_size:]
                        elif trunc_medium == 0:
                            mask = [1] * pad_size
                            token_ids = token_ids[:pad_size // 2] + token_ids[-(pad_size) // 2:]
                        elif trunc_medium == 1:
                            mask = [1] * pad_size
                            tokens_ids = token_ids[:(pad_size) // 3] + tokens_ids[pad_size // 3 + (
                                        len(tokens_ids) - pad_size) // 2:2 * (pad_size) // 3 + (
                                        len(tokens_ids) - pad_size) // 2] + tokens_ids[-(pad_size // 3):]
                        elif trunc_medium > 1:
                            mask = [1] * pad_size
                            token_ids = token_ids[:trunc_medium] + token_ids[-(pad_size - trunc_medium):]
                label = [int(x) for x in label_str.split(',')]
                mlb = MultiLabelBinarizer(classes=range(config.num_classes))
                label_one_hot = mlb.fit_transform([label]).tolist()[0]
                contents.append((token_ids, label_one_hot, seq_len, mask))
        return contents

    train = load_dataset(config.train_path, config.pad_size)
    dev = load_dataset(config.dev_path, config.pad_size)
    test = load_dataset(config.test_path, config.pad_size)
    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def output_records(config, results, test_report, test_confusion, train_time_dif):
    with open(os.path.join(config.output_path, 'clf_evaluation.txt'), 'a', encoding='UTF-8') as f:
        print('This is result of %s is:\n%s' % (config.model_name,
                                                results) + '\n' + 'when the bs = %d' % config.batch_size + ' pad_size = %d' % config.pad_size + '  lr = %s' % config.learning_rate + ' threshold = %s' % config.threshold + ' epoch = %s' % config.num_epochs + ' training time= %s\n' % train_time_dif,
              file=f)
    with open(os.path.join(config.output_path, 'label_evaluation.txt'), 'a') as f:
        print('This is result of %s is:\n%s' % (config.model_name,
                                                test_report) + '\n' + 'when the bs = %d' % config.batch_size + ' pad_size = %d' % config.pad_size + '  lr = %s' % config.learning_rate + ' threshold = %s' % config.threshold + ' epoch = %s\n' % config.num_epochs + ' training time= %s\n' % train_time_dif,
              file=f)
    with open(os.path.join(config.output_path, 'label_evaluation.txt'), 'a') as f:
        for i in range(len(test_confusion)):
            key = config.class_list[i]
            print('The confusion matrix for Label "%s"' % key + ' is:\n%s' % test_confusion[i], file=f)
