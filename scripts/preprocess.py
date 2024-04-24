import sys
import string  
import pandas as pd
from tqdm import tqdm

import spacy
from spacy.tokens import Doc, DocBin, Token

def read_file(fname, outname):
    print(f"Loading {fname}")
    df = pd.read_json(fname, lines = True)
    
    print("Loading model")
    nlp = spacy.load("uk_core_news_lg", disable=["ner", "attribute_ruler", "lemmatizer"])

    db = DocBin()

    print("Processing data...")
    for i in tqdm(df.index):
        sent_starts = []
        for s in df['sentences'][i]:
            sent_starts += [True] + [False for _ in range(len(s)-1)]
        doc = spacy.tokens.doc.Doc(nlp.vocab, words=df['tokens'][i], sent_starts = sent_starts)

        for cl in df['clusters'][i]:
            spans = [doc[ss:ee] for ss, ee in cl]
            skey = f"coref_clusters_{len(doc.spans)}"
            doc.spans[skey] = spans
        
        doc = nlp(doc)
        headc = 0
        for ii, cl in enumerate(df['clusters'][i]):
            heads = [doc[ss:ee].root.i for ss, ee in cl]
            heads = list(set(heads))
            if len(heads) == 1:
                continue
            headc += 1
            spans = [doc[hh : hh + 1] for hh in heads]
            doc.spans[f"coref_head_clusters_{headc}"] = spans

        db.add(doc)
    
    print(f"Serializing {len(db)} documents")
    db.to_disk(outname)

if __name__ == "__main__":
    read_file(sys.argv[1], sys.argv[2])