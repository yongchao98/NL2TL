import copy
import itertools
import json
import torch
from torch.utils.data import Dataset
from collections import Counter, namedtuple, defaultdict
from tqdm import tqdm

from transformers import (BertTokenizer,
                          RobertaTokenizer,
                          XLMRobertaTokenizer,
                          PreTrainedTokenizer)

instance_fields = [
    'sent_id', 'tokens', 'ltl', 'nl_pieces', 'ltl_pieces', 'ltl_idxs', 'ltl_mask', 'nl_label_idxs',
    'nl_label_mask', 'nl_label', 'nl_idxs', 'nl_mask', 'ltl_label_idxs', 'ltl_label_mask', 'ltl_label', 'vocab_ltl_label_idxs', 'vocab_ltl_label_mask', 'vocab_ltl_label',
    'pieces', 'piece_idxs','attention_mask', 'token_lens', 'label_idxs', 'label_mask', 'labels', 'prop_label_idxs', 'prop_labels'
]

batch_fields = [
    'sent_ids', 'tokens', 'ltl', 'nl_pieces', 'ltl_pieces', 'ltl_idxs', 'ltl_mask', 'nl_label_idxs',
    'nl_label_mask', 'nl_label', 'nl_idxs', 'nl_mask', 'ltl_label_idxs', 'ltl_label_mask', 'ltl_label', 'vocab_ltl_label_idxs', 'vocab_ltl_label_mask', 'vocab_ltl_label',
    'pieces', 'piece_idxs','attention_mask', 'token_lens', 'label_idxs', 'label_mask', 'labels',
    'attention_masks', 'label_masks', 'token_nums', 'ltls', 'prop_labels', 'prop_label_idxs'
]

Instance = namedtuple('Instance', field_names=instance_fields,
                      defaults=[None] * len(instance_fields))
Batch = namedtuple('Batch', field_names=batch_fields,
                   defaults=[None] * len(batch_fields))

def generate_vocabs(datasets):
    operation_set = set()
    object_set = set()
    action_set = set()

    for dataset in datasets:
        operation_set.update(dataset.operation_set)
        object_set.update(dataset.object_set)
        action_set.update(dataset.action_set)
    
    print('operation: ',len(operation_set),operation_set)
    print('object: ',len(object_set),object_set)
    print('action: ',len(action_set),action_set)

    type_set = set.union(operation_set,object_set,action_set)
    # type_set.add('<start>')
    type_set.add('<end>')
    type_stoi = {k:i for i, k in enumerate(type_set, 1)}
    type_stoi['<start>'] = 0

    return {
        'type': type_stoi
    }


class LTLDataset(Dataset):
    def __init__(self, path, tokenizer, max_length=64):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = max_length
        self.tokenizer = tokenizer

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @property
    def operation_set(self):
        type_set = set()
        for inst in self.data:
            for trg in inst['ltl']:
                if not '_v' in trg and not '_n' in trg:
                    type_set.add(trg)
        return type_set

    @property
    def object_set(self):
        type_set = set()
        for inst in self.data:
            for trg in inst['ltl']:
                if '_n' in trg:
                    type_set.add(trg)
        return type_set

    @property
    def action_set(self):
        type_set = set()
        for inst in self.data:
            for trg in inst['ltl']:
                if '_v' in trg:
                    type_set.add(trg)
        return type_set

    
    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
                tokens = inst['sentence']
                pieces = [self.tokenizer.tokenize(t) for t in tokens]
                token_lens = [len(x) for x in pieces]
                if 0 in token_lens:
                    overlength_num += 1
                    continue 
                pieces = [p for ps in pieces for p in ps]
                inst_len = max(len(pieces),len(inst['ltl']))
                if self.max_length != -1 and inst_len > self.max_length - 2:
                    overlength_num += 1
                    continue
                inst = {'inst_id':inst['id'],'tokens':tokens,'pieces':pieces,'ltl':inst['ltl'],'token_lens':token_lens}
                self.data.append(inst)
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        print('Loaded {} instances from {}'.format(len(self), self.path))

    def numberize(self, vocabs):
        """Numberize word pieces, labels, etcs.
        :param tokenizer: Bert tokenizer.
        :param vocabs (dict): a dict of vocabularies.
        """
        type_stoi = vocabs['type']

        data = []
        for inst in self.data:
            tokens = inst['tokens']
            pieces = inst['pieces']
            sent_id = inst['inst_id']
            token_num = len(tokens)
            token_lens = inst['token_lens']
            ltl= inst['ltl']

            # Pad word pieces with special tokens
            piece_idxs = self.tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
            pad_num = self.max_length - len(piece_idxs)
            attn_mask = [1] * len(piece_idxs) + [0] * pad_num
            piece_idxs = piece_idxs + [0] * pad_num

            ltl_idxs = [0] + [type_stoi[op] for op in ltl]
            pad_len = self.max_length - len(ltl_idxs)
            ltl_len = len(ltl_idxs)
            # ltl_idxs = ltl_idxs
            ltl_mask = []
            for i in range(self.max_length):
                if ltl_len - (i+1) > 0:
                    temp = (i+1) * [1] + (ltl_len - (i+1))*[0] + [0] * pad_len
                else:
                    temp = ltl_len * [1] + [0] * pad_len
                # print(temp)
                ltl_mask.append(temp)
            # print('ltl',len(ltl_mask))
            ltl_idxs = ltl_idxs + [0] * pad_len
            # ltl_mask = [1] * len(ltl_idxs) + [0] * pad_len
            ltl_labels = [type_stoi[op] for op in ltl] + [type_stoi['<end>']]
            # pad_len = self.max_length - len(ltl_idxs)
            ltl_labels = ltl_labels + [-100] * pad_len

            instance = Instance(
                sent_id=sent_id,
                tokens=tokens,
                ltl=ltl,
                pieces=pieces,
                piece_idxs=piece_idxs,
                attention_mask=attn_mask,
                token_lens=token_lens,
                label_idxs=ltl_idxs,
                label_mask=ltl_mask,
                labels=ltl_labels
            )
            data.append(instance)
        self.data = data

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_token_lens = []
        batch_attention_masks = []
        batch_label_idxs = []
        batch_label_masks = []
        batch_labels = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)

        for inst in batch:
            token_num = len(inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_label_idxs.append(inst.label_idxs)
            batch_label_masks.append(inst.label_mask)
            batch_labels.append(inst.labels)
                    
        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)
        batch_label_idxs = torch.cuda.LongTensor(batch_label_idxs)
        batch_label_masks = torch.cuda.FloatTensor(batch_label_masks)
        batch_labels = torch.cuda.LongTensor(batch_labels)
        token_nums = torch.cuda.LongTensor(token_nums)
        # print('piece idx: ',batch_piece_idxs.shape)
        # print('batch_attention_masks: ',batch_attention_masks.shape)
        # print('batch_label_idxs: ',batch_label_idxs.shape)
        # print('batch_label_masks: ',batch_label_masks.shape)
        # print('batch_labels: ',batch_labels.shape)
        # print('token_nums: ',token_nums.shape)


        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            ltls=[inst.ltl for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            label_idxs=batch_label_idxs,
            label_masks=batch_label_masks,
            labels=batch_labels,
            token_nums=token_nums
        )

class LTL2EngT5Dataset(Dataset):
    def __init__(self, path, tokenizer, config, model_config, ltl_vocabs = None, max_size = -1):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = config.max_length
        self.max_generate_length = config.max_generate_length
        self.tokenizer = tokenizer
        self.ltl_vocabs = ltl_vocabs
        self.decoder_start_token_id = model_config.decoder_start_token_id
        self.pad_token_id = model_config.pad_token_id
        self.ignore_label_id = -100
        self.max_size = max_size

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        data = []
        all_tokens = []
        all_ltls = []
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in tqdm(r):
                inst = json.loads(line)
                sent_id = inst['id']
                if 'logic_sentence' in inst:
                    tokens = ' '.join(inst['logic_sentence'])
                else:
                    tokens = ' '.join(inst['sentence'])
                # ltl_input = ['translate LTL to English:']
                ltl_input = []
                ltl_label = []
                if 'logic_ltl' in inst:
                    inst_ltl = inst['logic_ltl']
                else:
                    inst_ltl = inst['ltl']
                for ob in inst_ltl:
                    if "_v" in ob:
                        ob = ob.replace('_v','')
                        ob = ob.split('_')
                    elif "_n" in ob:
                        ob = ob.replace('_n','')
                        ob = ob.split('_')
                    if not isinstance(ob, list):
                        ob = [ob]
                    ltl_input.extend(ob)
                    ltl_label.extend(ob)
                # vocab_ltl_input = copy.deepcopy(ltl_input)
                vocab_ltl_label = copy.deepcopy(ltl_label)
                ltl_input = ' '.join(ltl_input)
                ltl_label = ' '.join(ltl_label)
                ltl = copy.deepcopy(ltl_label)
                # nl_input =  'translate English to LTL: ' + tokens
                nl_input = tokens

                nl_label = self.tokenizer(tokens).input_ids
                ltl_label = self.tokenizer(ltl_label).input_ids #['▁(', '▁F', '▁(', '▁grab', '▁(', '▁orange', '▁', ')', '▁', ')', '▁', ')', '</s>']
                vocab_ltl_label = [self.ltl_vocabs[vo] for vo in vocab_ltl_label] + [self.ltl_vocabs['<end>']]
                nl_idxs = self.tokenizer(nl_input).input_ids
                ltl_idxs = self.tokenizer(ltl_input).input_ids
                nl_label_idxs = [self.decoder_start_token_id] + nl_label[:-1]
                ltl_label_idxs = [self.decoder_start_token_id] + ltl_label[:-1]
                vocab_ltl_label_idxs = [self.ltl_vocabs['<begin>']] + vocab_ltl_label[:-1]
                nl_idxs_len = len(nl_idxs)
                ltl_idxs_len = len(ltl_idxs)
                if self.max_length != -1 and max(nl_idxs_len,ltl_idxs_len) > self.max_length:
                    overlength_num += 1
                    continue
                nl_idxs = nl_idxs + [self.pad_token_id]*(self.max_length-nl_idxs_len)
                ltl_idxs = ltl_idxs + [self.pad_token_id]*(self.max_length-ltl_idxs_len)
                nl_mask = [1] * nl_idxs_len + [0] *  (self.max_length-nl_idxs_len)
                ltl_mask = [1] * ltl_idxs_len + [0] *  (self.max_length-ltl_idxs_len)
                nl_label_len = len(nl_label_idxs)
                ltl_label_len = len(ltl_label_idxs)
                vocab_ltl_label_len = len(vocab_ltl_label_idxs)
                nl_label_pad_len = self.max_length - nl_label_len
                ltl_label_pad_len = self.max_length - ltl_label_len
                vocab_ltl_label_pad_len = self.max_length - vocab_ltl_label_len
                # nl label mask
                nl_label_mask = []
                for i in range(self.max_length):
                    if nl_label_len - (i+1) > 0:
                        temp = (i+1) * [1] + (nl_label_len - (i+1))*[0] + [0] * nl_label_pad_len
                    else:
                        temp = nl_label_len * [1] + [0] * nl_label_pad_len
                    nl_label_mask.append(temp)
                nl_label_idxs = nl_label_idxs + [self.pad_token_id] * (self.max_length-nl_label_len)
                nl_label = nl_label + [-100] * (self.max_length-nl_label_len)
                # ltl label mask
                ltl_label_mask = []
                for i in range(self.max_length):
                    if ltl_label_len - (i+1) > 0:
                        temp = (i+1) * [1] + (ltl_label_len - (i+1))*[0] + [0] * ltl_label_pad_len
                    else:
                        temp = ltl_label_len * [1] + [0] * ltl_label_pad_len
                    ltl_label_mask.append(temp)
                ltl_label_idxs = ltl_label_idxs + [self.pad_token_id] * (self.max_length-ltl_label_len)
                ltl_label = ltl_label + [-100] * (self.max_length-ltl_label_len)
                # vocab ltl label mask
                vocab_ltl_label_mask = []
                for i in range(self.max_length):
                    if vocab_ltl_label_len - (i+1) > 0:
                        temp = (i+1) * [1] + (vocab_ltl_label_len - (i+1))*[0] + [0] * vocab_ltl_label_pad_len
                    else:
                        temp = vocab_ltl_label_len * [1] + [0] * vocab_ltl_label_pad_len
                    vocab_ltl_label_mask.append(temp)
                vocab_ltl_label_idxs = vocab_ltl_label_idxs + [self.ltl_vocabs['<pad>']] * (self.max_length-vocab_ltl_label_len)
                vocab_ltl_label = vocab_ltl_label + [-100] * (self.max_length-vocab_ltl_label_len)

                nl_pieces = self.tokenizer.tokenize(tokens)
                ltl_pieces = self.tokenizer.tokenize(ltl)
                tokens = tokens.split(' ')
                ltl = ltl.split(' ')
                # print('--------------------------------------------')
                # print('ltl_idxs',len(ltl_idxs))
                # print('ltl_mask',len(ltl_mask))
                # print('nl_label_idxs',len(nl_label_idxs))
                # print('nl_label_mask',nl_label_mask)
                # print('nl_label',len(nl_label))
                # print('nl_mask',len(nl_mask))
                # print('ltl_label_idxs',len(ltl_label_idxs))
                # print('ltl_label_mask',ltl_label_mask)
                # print('ltl_label',len(ltl_label))
                # print(labels)
                # print(piece_mask)
                # print(len(ltl_idxs),len(piece_idxs))
                
                instance = Instance(
                    sent_id=sent_id,
                    tokens=tokens,
                    ltl=ltl,
                    nl_pieces=nl_pieces,
                    ltl_pieces=ltl_pieces,
                    ltl_idxs=ltl_idxs, # this is the input to encoder of ltl2eng model
                    ltl_mask=ltl_mask,
                    nl_label_idxs=nl_label_idxs, # this is the input to decoder of ltl2eng model
                    nl_label_mask=nl_label_mask,
                    nl_label=nl_label, # this is the label for ltl2eng model 
                    nl_idxs=nl_idxs, # this is the input to encoder of eng2ltl model
                    nl_mask=nl_mask,
                    ltl_label_idxs=ltl_label_idxs, # this is the input to decoder of eng2ltl model
                    ltl_label_mask=ltl_label_mask,
                    ltl_label=ltl_label, # this is the label for eng2ltl model
                    vocab_ltl_label_idxs=vocab_ltl_label_idxs,
                    vocab_ltl_label_mask=vocab_ltl_label_mask,
                    vocab_ltl_label=vocab_ltl_label
                )
                data.append(instance)
                if not self.max_size == -1:
                    if len(data) >=self.max_size:
                        break
        self.data = data
        print('Loaded {} instances from {}'.format(len(self), self.path))
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        
    def collate_fn(self, batch):
        batch_ltl_idxs = []
        batch_ltl_mask = []
        batch_nl_label_idxs = []
        batch_nl_label_mask = []
        batch_nl_label = []

        batch_nl_idxs = []
        batch_nl_mask = []
        batch_ltl_label_idxs = []
        batch_ltl_label_mask = []
        batch_ltl_label = []

        batch_vocab_ltl_label_idxs = []
        batch_vocab_ltl_label_mask = []
        batch_vocab_ltl_label = []
        
        for inst in batch:
            batch_ltl_idxs.append(inst.ltl_idxs)
            batch_ltl_mask.append(inst.ltl_mask)
            batch_nl_label_idxs.append(inst.nl_label_idxs)
            batch_nl_label_mask.append(inst.nl_label_mask)
            batch_nl_label.append(inst.nl_label)

            batch_nl_idxs.append(inst.nl_idxs)
            batch_nl_mask.append(inst.nl_mask)
            batch_ltl_label_idxs.append(inst.ltl_label_idxs)
            batch_ltl_label_mask.append(inst.ltl_label_mask)
            batch_ltl_label.append(inst.ltl_label)

            batch_vocab_ltl_label_idxs.append(inst.vocab_ltl_label_idxs)
            batch_vocab_ltl_label_mask.append(inst.vocab_ltl_label_mask)
            batch_vocab_ltl_label.append(inst.vocab_ltl_label)


                    
        batch_ltl_idxs = torch.cuda.LongTensor(batch_ltl_idxs)
        batch_ltl_mask = torch.cuda.FloatTensor(batch_ltl_mask)
        batch_nl_label_idxs = torch.cuda.LongTensor(batch_nl_label_idxs)
        batch_nl_label_mask = torch.cuda.FloatTensor(batch_nl_label_mask)
        batch_nl_label = torch.cuda.LongTensor(batch_nl_label)

        batch_nl_idxs = torch.cuda.LongTensor(batch_nl_idxs)
        batch_nl_mask = torch.cuda.FloatTensor(batch_nl_mask)
        batch_ltl_label_idxs = torch.cuda.LongTensor(batch_ltl_label_idxs)
        batch_ltl_label_mask = torch.cuda.FloatTensor(batch_ltl_label_mask)
        batch_ltl_label = torch.cuda.LongTensor(batch_ltl_label)

        batch_vocab_ltl_label_idxs = torch.cuda.LongTensor(batch_vocab_ltl_label_idxs)
        batch_vocab_ltl_label_mask = torch.cuda.LongTensor(batch_vocab_ltl_label_mask)
        batch_vocab_ltl_label = torch.cuda.LongTensor(batch_vocab_ltl_label)
        
        # print('batch_ltl_idxs: ',batch_ltl_idxs.shape)
        # print('batch_ltl_mask: ',batch_ltl_mask.shape)
        # print('batch_nl_label_idxs: ',batch_nl_label_idxs.shape)
        # print('batch_nl_label_mask: ',batch_nl_label_mask.shape)
        # print('batch_nl_label: ',batch_nl_label.shape)

        # print('batch_nl_idxs: ',batch_nl_idxs.shape)
        # print('batch_nl_mask: ',batch_nl_mask.shape)
        # print('batch_ltl_label_idxs: ',batch_ltl_label_idxs.shape)
        # print('batch_ltl_label_mask: ',batch_ltl_label_mask.shape)
        # print('batch_ltl_label: ',batch_ltl_label.shape)

        return Batch(
            sent_ids = [inst.sent_id for inst in batch],
            tokens=[inst.tokens for inst in batch],
            ltl=[inst.ltl for inst in batch],
            nl_pieces=[inst.nl_pieces for inst in batch],
            ltl_pieces=[inst.ltl_pieces for inst in batch],
            ltl_idxs=batch_ltl_idxs, # this is the input to encoder of ltl2eng model
            ltl_mask=batch_ltl_mask,
            nl_label_idxs=batch_nl_label_idxs, # this is the input to decoder of ltl2eng model
            nl_label_mask=batch_nl_label_mask,
            nl_label=batch_nl_label, # this is the label for ltl2eng model 
            nl_idxs=batch_nl_idxs, # this is the input to encoder of eng2ltl model
            nl_mask=batch_nl_mask,
            ltl_label_idxs=batch_ltl_label_idxs, # this is the input to decoder of eng2ltl model
            ltl_label_mask=batch_ltl_label_mask,
            ltl_label=batch_ltl_label,
            vocab_ltl_label_idxs=batch_vocab_ltl_label_idxs,
            vocab_ltl_label_mask=batch_vocab_ltl_label_mask,
            vocab_ltl_label=batch_vocab_ltl_label
        )

class UnlabeledLTLDataset(Dataset):
    def __init__(self, path, tokenizer, config, max_length=128):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.decoder_start_token_id = config.decoder_start_token_id
        self.pad_token_id = config.pad_token_id
        self.ignore_label_id = -100

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        data = []
        all_tokens = []
        all_ltls = []
        with open(self.path, 'r', encoding='utf-8') as r:
            for inst_id, line in enumerate(r):
                inst = json.loads(line)
                sent_id = inst_id
                ltl = ['translate LTL to English:']
                for ob in inst:
                    if "_v" in ob:
                        ob = ob.replace('_v','')
                    elif "_n" in ob:
                        ob = ob.replace('_n','')
                    ob = ob.split('_')
                    ltl.extend(ob)
                ltl = ' '.join(ltl)
                # print(ltl)
                # inst = {'inst_id':inst['id'],'tokens':tokens,'ltl':ltl}
                
                ltl_idxs = self.tokenizer(ltl).input_ids
                ltl_len = len(ltl_idxs)
                if self.max_length != -1 and ltl_len > self.max_length:
                    overlength_num += 1
                    continue
                ltl_idxs = ltl_idxs + [self.pad_token_id]*(self.max_length-ltl_len)
                ltl_mask = [1] * ltl_len + [0] *  (self.max_length-ltl_len)
                assert len(ltl_idxs) == len(ltl_mask)
                ltl_pieces = self.tokenizer.tokenize(ltl)
                ltl = ltl.split(' ')

                instance = Instance(
                    sent_id=sent_id,
                    ltl=ltl,
                    ltl_pieces=ltl_pieces,
                    piece_idxs=ltl_idxs,
                    attention_mask=ltl_mask
                )
                data.append(instance)
        self.data = data
        print('Loaded {} instances from {}'.format(len(self), self.path))
        if overlength_num:
            print('Discarded {} overlength instances'.format(overlength_num))
        

    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_attention_masks = []
        for inst in batch:
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)
        return Batch(
            sent_ids = [inst.sent_id for inst in batch],
            ltls=[inst.ltl for inst in batch],
            ltl_pieces = [inst.ltl_pieces for inst in batch],
            piece_idxs=batch_piece_idxs,
            attention_masks=batch_attention_masks
        )


def get_prop_labels(props, token_num):
    """Convert event mentions in a sentence to a trigger label sequence with the
    length of token_num.
    :param events (list): a list of event mentions.
    :param token_num (int): the number of tokens.
    :return: a sequence of BIO format labels.
    """
    labels = ['O'] * token_num
    for id_, prop in props.items():
        # trigger = event['trigger']
        start, end = prop['span']
        labels[start] = 'B-{}'.format('prop')
        for i in range(start + 1, end):
            labels[i] = 'I-{}'.format('prop')
    return labels

class SeqLogicDataset(Dataset):
    def __init__(self, path, vocabs, tokenizer, config, model_config, max_size = 1500):
        """
        :param path (str): path to the data file.
        :param max_length (int): max sentence length.
        :param gpu (bool): use GPU (default=False).
        :param ignore_title (bool): Ignore sentences that are titles (default=False).
        """
        self.path = path
        self.data = []
        self.max_length = config.max_length
        # self.max_generate_length = config.max_generate_length
        self.tokenizer = tokenizer
        # self.ltl_vocabs = ltl_vocabs
        self.decoder_start_token_id = model_config.decoder_start_token_id
        self.pad_token_id = model_config.pad_token_id
        self.ignore_label_id = -100
        self.max_size = max_size
        self.vocab = vocabs
        self.prop_label_stoi = self.vocab['prop_label']

        self.load_data()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    def load_data(self):
        """Load data from file."""
        overlength_num = 0
        data = []
        all_tokens = []
        all_ltls = []
        with open(self.path, 'r', encoding='utf-8') as r:
            for line in tqdm(r):
                inst = json.loads(line)
                sent_id = inst['id']
                # tokens = ' '.join(inst['sentence'])
                # ltl_input = ['translate LTL to English:']
                tokens = inst['sentence']
                sent_len = len(tokens)
                props = inst['propositions']
                prop_labels = get_prop_labels(props, sent_len)
                pieces = [self.tokenizer.tokenize(t) for t in tokens]
                if len(pieces)>self.max_length:
                    skip_num += 1
                    continue
                token_lens = [len(x) for x in pieces]
                if 0 in token_lens:
                    skip_num += 1
                    continue
                pieces = [p for ps in pieces for p in ps]
                if len(pieces) == 0:
                    skip_num += 1
                    continue
                piece_idxs = self.tokenizer.encode(pieces,
                                          add_special_tokens=True,
                                          max_length=self.max_length,
                                          truncation=True)
                pad_num = self.max_length - len(piece_idxs)
                attn_mask = [1] * len(piece_idxs) + [0] * pad_num
                piece_idxs = piece_idxs + [0] * pad_num
                prop_label_idxs = [self.prop_label_stoi[l] for l in prop_labels]
                
                instance = Instance(
                    sent_id=sent_id,
                    tokens=tokens,
                    pieces=pieces,
                    piece_idxs=piece_idxs,
                    token_lens=token_lens,
                    attention_mask=attn_mask,
                    prop_labels=prop_labels,
                    prop_label_idxs=prop_label_idxs
                )
                data.append(instance)
                if not self.max_size == -1:
                    if len(data) >=self.max_size:
                        break
        self.data = data
        print('Loaded {} instances from {}'.format(len(self), self.path))
        if overlength_num:
            print('Discarded {} overlength instances'.format(skip_num))
        
    def collate_fn(self, batch):
        batch_piece_idxs = []
        batch_tokens = []
        batch_token_lens = []
        batch_attention_masks = []
        batch_prop_label_idxs = []

        sent_ids = [inst.sent_id for inst in batch]
        token_nums = [len(inst.tokens) for inst in batch]
        max_token_num = max(token_nums)
        
        for inst in batch:
            token_num = len(inst.tokens)
            assert token_num == len(inst.prop_label_idxs)
            # print(inst.prop_labels, inst.tokens)
            batch_piece_idxs.append(inst.piece_idxs)
            batch_attention_masks.append(inst.attention_mask)
            batch_token_lens.append(inst.token_lens)
            batch_tokens.append(inst.tokens)
            batch_prop_label_idxs.append(inst.prop_label_idxs +
                                        [0] * (max_token_num - token_num))

        batch_piece_idxs = torch.cuda.LongTensor(batch_piece_idxs)
        batch_attention_masks = torch.cuda.FloatTensor(batch_attention_masks)
        batch_prop_label_idxs = torch.cuda.LongTensor(batch_prop_label_idxs)
        token_nums = torch.cuda.LongTensor(token_nums)

                    
        return Batch(
            sent_ids=sent_ids,
            tokens=[inst.tokens for inst in batch],
            piece_idxs=batch_piece_idxs,
            token_lens=batch_token_lens,
            attention_masks=batch_attention_masks,
            prop_labels=[inst.prop_labels for inst in batch],
            token_nums=token_nums,
            prop_label_idxs=batch_prop_label_idxs
        )
