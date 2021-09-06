import sys
from copy import deepcopy

filename = sys.argv[1]
splitted_fname = filename.split("/")
outfile = '/'.join(splitted_fname[:-1])+'/truncated_'+splitted_fname[-1]

with open(filename, 'r') as fin:
    with open(outfile, 'w') as fout:
        for i, line in enumerate(fin):
            idx = line.rfind('</s>')
            assertion = line[idx+5:].strip()
            idx2 = line.find('If')
            facts = line[:idx2]
            facts = facts.split('. ')
            splitted = assertion.split()
            subject = splitted[0]
            adj = splitted[2][:-1]
            for fact in facts:
                if adj in fact and subject in fact:
                    gold_fact = fact
            fout.write(f"{gold_fact}. </s> {assertion}\n")

