import os
import json
import re
import pickle
from transformers import BertTokenizer
import random
from sumy.nlp.tokenizers import Tokenizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.kl import KLSummarizer
tokenizer = BertTokenizer.from_pretrained('../../scibert_scivocab_uncased')


def func():
    n = 0
    pkl_list = os.listdir('iclr2019/')
    rs = []
    decs = []
    for i in pkl_list:
        if i.split('.')[-1] != 'json':
            continue
        ix = i.split('.')[0]
        reviews = []
        text = ""
        file = "iclr2019/" + str(ix) + ".json"
        js = json.loads(open(file).read())
        rvs = js["Reviews"]
        dec = False
        if "Recommendation" not in js:
            continue
        if js["Recommendation"].split()[0]=="Accept":
             dec = True
        elif js["Recommendation"].split()[0]=="Reject":
             dec = False
        else:
             continue
        for j in range(len(rvs)):
            if rvs[j]["IS_META_REVIEW"]:
                mr = rvs[j]["Review"].replace("\n", " ")
            else:
                text = text + rvs[j]["Review"].replace("\n", " ") + " "
        summarizer = KLSummarizer()
        parser = PlaintextParser.from_string(mr, Tokenizer("english"))
        summary = summarizer(parser.document, 10)
        if dec:
            n = n + 1
        text_summary = ""
        for i in summary:
            text_summary = text_summary + str(i) + " "
        r1 = tokenizer.encode(text_summary, max_length=256, pad_to_max_length=True)
        rs.append(r1)
        decs.append(dec)
    print(n/len(decs))
    o = [(rs[i], decs[i]) for i in range(len(rs))]
    random.shuffle(o)
    rs = [i[0] for i in o]
    decs = [i[1] for i in o]
    f = open('dec_fol/dec2019mks.pkl', 'wb')
    pickle.dump(rs, f)
    f.close()
    f = open('dec_fol/dec2019mkt.pkl', 'wb')
    pickle.dump(decs, f)
    f.close()
