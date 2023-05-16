import nltk
import spacy_transformers
import benepar, spacy
from nltk.tree import Tree
from tqdm import tqdm
import re
import copy
import json
from spacy.language import Language
import random

benepar.download('benepar_en3')
data_portion = 'total'
#data_size = 1500


nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

# --------------------------- customize functions --------------------------  #
@Language.component("prevent_sbd")
def prevent_sbd(doc):
    for token in doc:
        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False
    return doc
nlp.add_pipe('prevent_sbd', before='parser')
# -------------------------------------------------------------------------- #

logic_set = set(['G', 'F', '(' , ')', '&', '|', 'U', '[', ']', ':', 'hist|ically', 'U', '->'])
operation_set = set(['>=', '<=', '>', '<', '->', '==', 'rise', 'fall', '=', '-'])

def get_proposition(ltl):
    """
    this function finds the proposition  (action object) and automaton rise action object 
    """
    ltl = re.sub('\[ [0-9]+ : [0-9]+ \]', '', ltl)
    while '  ' in ltl:
        ltl = ltl.replace('  ',' ')
    ltl = ltl.split(' ')
    # print(ltl)
    propositions = []
    proposition = []
    autos = []
    auto = []
    for op in ltl:
        # if op == 'rise' or op == 'fall':
        #     auto.append(op)
        if not op in logic_set:
            auto.append(op)
            if not op in operation_set:
                proposition.append(op)
        if len(proposition) == 2:
            propositions.append(tuple(proposition))
            proposition = []
            autos.append(auto)
            auto = []
    return propositions, autos

def get_prop_idx(propositions, tree):
    """
    Find the index of corresponding proposition in original sentence
    """
    prop2idx = [] 
    tokens = tree.leaves()
    for prop in propositions:
        act_pos_list = []
        obj_pos_list = []
        for pos, t in enumerate(tokens):
            # if not pos in used_index: 
            if t == prop[0]:
                act_pos_list.append(pos)
            elif t == prop[1]:
                obj_pos_list.append(pos)
        assert len(act_pos_list) > 0 and len(obj_pos_list) > 0
        min_dis = 10000000

        for p1 in act_pos_list:
            for p2 in obj_pos_list:
                if abs(p1-p2) < min_dis:
                    min_dis = abs(p1-p2)
                    act_pos = p1
                    obj_pos = p2
        # used_index.add(act_pos)
        # used_index.add(obj_pos)
        prop2idx.append({'prop':prop, 'pos':[act_pos, obj_pos] if act_pos < obj_pos else [obj_pos, act_pos]})

    return prop2idx

def find_prop_spans(prop2idx, tree):
    """
    find the span of proposition in sentence
    """
    indexes = []
    for prop in prop2idx:
        pos = prop['pos']
        tree_pos_s = tree.leaf_treeposition(pos[0])
        tree_pos_e = tree.leaf_treeposition(pos[1])
        parent = []
        branch = []
        for s, e in zip(tree_pos_s, tree_pos_e):
            if s == e:
                parent.append(s)
            else:
                branch.append(s)
                branch.append(e)
                break
        token_seq = []
        for i in range(branch[0], branch[1]+1):
            t_index = copy.copy(parent)
            t_index.append(i)
            subtree = tree[t_index]
            token_seq.extend(subtree.leaves())
        for i in range(len(tokens)):
            if tokens[i:i+len(token_seq)] == token_seq:
                indexes.append((i, i+len(token_seq)))
    return indexes

def get_new_ltl(auto, ltl, prop_id):
    ltl = ltl.split(' ')
    start_idxs = []
    end_idxs = []
    for i, op in enumerate(ltl):
        if op == auto[0]:
            start_idxs.append(i)
        if op == auto[-1]:
            end_idxs.append(i+1)
    dist = 100000000
    s_idx = 0
    e_idx = len(ltl)
    for s in start_idxs:
        for e in end_idxs:
            if e - s < dist and e - s > 0:
                s_idx = s
                e_idx = e
                dist = e - s
    sub_ltl = ' '.join(ltl[s_idx:e_idx])
    ltl = ' '.join(ltl)
    num_left = sub_ltl.count('(')
    num_right = sub_ltl.count(')')
    assert num_left >= num_right and num_left - num_right <= 1
    if num_left > num_right:
         sub_ltl += ' )'
    ltl = ltl.replace(sub_ltl, prop_id)
    while '  ' in ltl:
        ltl = ltl.replace('  ', ' ') 
    return ltl, sub_ltl
           

input_path = '/raw_data/circuit_{}.jsonl'.format(data_portion)
corpus = []
sentences = []
ltls = []
with open(input_path,'r') as fin:
    for line in fin:
        line = json.loads(line)
        sentences.append(' '.join(line['sentence']).replace('hist|ically','histically'))
        ltls.append(' '.join(line['ltl']))
# random.shuffle(sentences)
# sentences = ['When signal_1_n is greater than or equal to 48.9 and the transition action that the value of signal_2_n decreases below 55.2 occurs, then for each moment within the following 23 to 50 time units the value of signal signal_3_n should be 20.4 ultimately at a certain moment in the future before the execution ends.']

# ltls = 'G ( signal_1_n >= 48.9 & rise ( signal_2_n < 55.2 ) -> G [ 23 : 50 ] ( F ( signal_3_n == 20.4 ) ) )'


ltl_idx = 0
errors = 0
corpus = []
for doc in nlp.pipe(sentences, disable=["tok2vec", "tagger", "attribute_ruler", "lemmatizer"]):
    # print(ltl_idx)
    # doc = nlp(sentence)
    ltl = ltls[ltl_idx]
    ltl_idx += 1
    # sentence = sentences[ltl_idx]
    sent = list(doc.sents)[0]
    tree = Tree.fromstring(sent._.parse_string)
    tokens = tree.leaves()
    sentence = ' '.join(tokens)
    ori_ltl = copy.deepcopy(ltl)
    ori_sent = copy.deepcopy(sentence)
    propositions, autos = get_proposition(ltl)
    try:
        prop2idx = get_prop_idx(propositions, tree)
    except:
        # print('-------------------- finding index error ---------------------------')
        # print(ltl, tokens)
        # print(propositions)
        # print(sentences[ltl_idx])
        errors += 1
        continue
    indexes = find_prop_spans(prop2idx, tree)

    # print('# --------------------------------- #')
    # print(sentence)
    # print(ltl)
    sorted_auto = []
    for i, (auto, index) in enumerate(zip(autos, indexes)):
        sorted_auto.append([auto, index, index[1]-index[0]])
    sorted_auto = sorted(sorted_auto, key=lambda x: x[2], reverse=True)
    # print(sorted_auto)
    index2auto = {}
    for auto in sorted_auto:
        added = False
        for index in index2auto:
            if auto[1][0] >= index[0] and auto[1][1] <= index[1]: # this means the longer span contain the shorter span
                index2auto[index].append(auto)
                added = True
                break
        if not added:
            index2auto[auto[1]] = [auto]
    # print('----------------------------------')
    # print(sentence)
    # print(ltl)
    # print(index2auto)
    prop2sub_ltl = {}
    for i, index in enumerate(sorted(index2auto.keys())):
        prop_idx =' prop_{} '.format(str(i+1))
        _autos = index2auto[index]
        # print('( prop_{} )'.format(str(i+1)), 'auto',autos, 'tokens', tokens[index[0]:index[1]])
        sentence = sentence.replace(' '.join(tokens[index[0]:index[1]]), '({})'.format(prop_idx))
        prop2sub_ltl[prop_idx]={'span':[index[0],index[1]]}
        
        for auto in _autos:
            auto = auto[0]
            # print(auto)
            try:
                ltl, sub_ltl = get_new_ltl(auto, ltl, prop_idx)
                if 'prop' in prop2sub_ltl[prop_idx]:
                    prop2sub_ltl[prop_idx]['prop'].append(sub_ltl.split(' '))
                else:
                    prop2sub_ltl[prop_idx]['prop'] = [sub_ltl.split(' ')]
            except:
                # print('# --------------- bracket error ----------------- #')
                # print(auto)
                # print(sentence)
                # print(ltl)
                continue
    if not ltl.count('(') == ltl.count(')'):
        # print('# --------------- bracket error ----------------- #')
        # print(sentence)
        # print(ltl)
        continue
    if 'signal_' in sentence:
        # print('# --------------- auto error ----------------- #')
        # print(sentence)
        # print(ltl)
        continue
    if 'signal_' in ltl:
        # print('# --------------- ltl error ----------------- #')
        # print(propositions, autos)
        # print(sentence)
        # print(ltl)
        continue
    # print('#----------------- good instance -------------- # ')
    # print('ori sent',ori_sent)
    # print('ori ltl', ori_ltl)
    # print(propositions, autos)
    # print(sentence)
    # print(ltl)
    # print(prop2sub_ltl)
    corpus.append({'id':ltl_idx, 'sentence':ori_sent.split(' '), 'ltl': ori_ltl.split(' '),'logic_sentence':sentence.split(' '), 'logic_ltl':ltl.split(), 'propositions': prop2sub_ltl})
    print(len(corpus))
    #if len(corpus) > data_size:
    #    break

with open('/raw_data/circuit_{}_span_2.jsonl'.format(data_portion),'w') as fout:
    for line in corpus:
        fout.write(json.dumps(line)+'\n')


    
