import os
import json
import time
from argparse import ArgumentParser
import tqdm
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from transformers import T5Tokenizer, AdamW, get_linear_schedule_with_warmup
from seq2seq import LTL2Eng
from data import LTLDataset, generate_vocabs, LTL2EngT5Dataset, UnlabeledLTLDataset
from config import Config

from util import collect_ltl_vocabs

parser = ArgumentParser()
parser.add_argument('-c', '--config', default='config/example.json')
parser.add_argument('-n', '--name', default='temp')
parser.add_argument('-o', '--outputname', default='para1.txt')
parser.add_argument('-g', '--gpu', type=int, default=0)
args = parser.parse_args()
config = Config.from_json_file(args.config)
config.gpu_device = args.gpu
print("Run training on GPU " + str(config.gpu_device))

if config.gpu_device >= 0:
    torch.cuda.set_device(config.gpu_device)
input_dir = os.path.join(config.log_path, args.name)
output_dir = os.path.join(config.output_path, args.outputname)
#if not os.path.exists(output_dir):
#    os.mkdir(output_dir)

model_name = config.bert_model_name
tokenizer = T5Tokenizer.from_pretrained(model_name,cache_dir=config.bert_cache_dir)

ltl_vocabs = collect_ltl_vocabs([config.train_file,config.dev_file,config.test_file])
model = LTL2Eng(config,ltl_vocabs,tokenizer)
model.load_state_dict(torch.load(input_dir+'/model_state.pt'))
test_set = LTL2EngT5Dataset(config.test_file,tokenizer,config,model.ltl2eng.config,ltl_vocabs=ltl_vocabs)

test_batch_num = len(test_set) // config.eval_batch_size + \
    (len(test_set) % config.eval_batch_size != 0)

model.cuda(device=config.gpu_device)
model.eval()

# test
progress = tqdm.tqdm(total=test_batch_num, ncols=75,
                    desc='Test {}'.format(1))
# calculate the one on one match accuracy
count_3 = 0
total = 0
test_predictions = []
for batch in DataLoader(test_set, batch_size=config.eval_batch_size,
                        shuffle=False, collate_fn=test_set.collate_fn):
    progress.update(1)
    with torch.no_grad():
        outputs = model.predict_eng2ltl(batch, max_length=config.max_generate_length)
        pred_ltls = tokenizer.batch_decode(outputs['output_idxs'], skip_special_tokens=True)

        for pred_ltl, gold_sent, gold_ltl in zip(pred_ltls, outputs['tokens'], outputs['ltl']):
            print('-----------------------------------')
            print('sent: ', ' '.join(gold_sent))
            print('gold: ',' '.join(gold_ltl))
            print('pred: ',pred_ltl)

            item_pred = pred_ltl.strip().split(' ')
            if item_pred == gold_ltl:
                count_3 += 1
            total += 1
            test_predictions.append('ltl: '+' '.join(gold_ltl)+'\ngold: '+' '.join(gold_sent)+'\npred ltl: '+pred_ltl)
test_accu = float(count_3)/total
progress.close()
print('test exact match accuracy: ', round(test_accu,6))

with open(output_dir, 'w') as fout:
    fout.write("epoch: {}, test accu: {}".format(1, test_accu) + '\n')
    for line in test_predictions:
        fout.write(line + '\n\n')
fout.close()
