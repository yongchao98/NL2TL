import copy
import json
import os

from typing import Dict, Any

from transformers import (BertConfig, RobertaConfig, XLMRobertaConfig,
                          PretrainedConfig)

class Config(object):
    def __init__(self, **kwargs):
        self.coref = kwargs.pop('coref', True)
        # bert
        self.bert_model_name = kwargs.pop('bert_model_name', 'bert-base-uncased')
        self.bert_cache_dir = kwargs.pop('bert_cache_dir', None)
        self.bert_dropout = kwargs.pop('bert_dropout', .5)
        
        self.linear_dropout = kwargs.pop('linear_dropout', .4)
        self.linear_bias = kwargs.pop('linear_bias', True)
        self.linear_activation = kwargs.pop('linear_activation', 'relu')
        
        self.train_file = kwargs.pop('train_file', None)
        self.dev_file = kwargs.pop('dev_file', None)
        self.test_file = kwargs.pop('test_file', None)
        self.log_path = kwargs.pop('log_path', None)
        # training
        self.accumulate_step = kwargs.pop('accumulate_step', 1)
        self.batch_size = kwargs.pop('batch_size', 10)
        self.eval_batch_size = kwargs.pop('eval_batch_size', 5)
        self.max_epoch = kwargs.pop('max_epoch', 50)
        self.learning_rate = kwargs.pop('learning_rate', 1e-4)
        self.bert_learning_rate = kwargs.pop('bert_learning_rate', 1e-5)
        self.weight_decay = kwargs.pop('weight_decay', 1e-4)
        self.bert_weight_decay = kwargs.pop('bert_weight_decay', 1e-5)
        self.warmup_epoch = kwargs.pop('warmup_epoch', 5)
        self.grad_clipping = kwargs.pop('grad_clipping', 5.0)
        # others
        #self.gpu_device = kwargs.pop('gpu_device', 0)
        




    @classmethod
    def from_dict(cls, dict_obj):
        """Creates a Config object from a dictionary.
        Args:
            dict_obj (Dict[str, Any]): a dict where keys are
        """
        config = cls()
        for k, v in dict_obj.items():
            setattr(config, k, v)
        return config

    @classmethod
    def from_json_file(cls, path):
        with open(path, 'r', encoding='utf-8') as r:
            return cls.from_dict(json.load(r))

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output

    def save_config(self, path):
        """Save a configuration object to a file.
        :param path (str): path to the output file or its parent directory.
        """
        if os.path.isdir(path):
            path = os.path.join(path, 'config.json')
        print('Save config to {}'.format(path))
        with open(path, 'w', encoding='utf-8') as w:
            w.write(json.dumps(self.to_dict(), indent=2,
                               sort_keys=True))
    @property
    def bert_config(self):
        if self.bert_model_name.startswith('bert-'):
            return BertConfig.from_pretrained(self.bert_model_name,
                                              cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('roberta-'):
            return RobertaConfig.from_pretrained(self.bert_model_name,
                                                 cache_dir=self.bert_cache_dir)
        elif self.bert_model_name.startswith('xlm-roberta-'):
            return XLMRobertaConfig.from_pretrained(self.bert_model_name,
                                                    cache_dir=self.bert_cache_dir)
        else:
            raise ValueError('Unknown model: {}'.format(self.bert_model_name))
