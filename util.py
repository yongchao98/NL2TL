import json


def collect_ltl_vocabs(path_list):
    vocabs = set([])
    for path in path_list:
        with open(path, 'r', encoding='utf-8') as r:
            for line in r:
                inst = json.loads(line)
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
                    for vo in ob:
                        vocabs.add(vo)
    vocabs = ['<pad>'] + ['<begin>'] + ['<end>'] + list(vocabs)
    vocabs_dic = {}
    for i, vob in enumerate(vocabs):
        vocabs_dic[vob] = i
    return vocabs_dic

