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
            rules = line[idx2:idx].split('. ') 
            splitted = assertion.split()
            subject = splitted[0]
            adj = splitted[2][:-1]
            filtered_rules = []
            for rule in rules:
                if f"chance X is {adj}" in rule or f"chance {subject} is {adj}" in rule:
                    filtered_rules.append(rule)
            #print(filtered_rules)
            rule_adjs = []
            for rule in filtered_rules:
                rulecopy = deepcopy(rule)
                rule = rule.split()
                if 'and' in rule:
                    rule_adjs.append([rule[3], rule[7], rulecopy])
                else:
                    rule_adjs.append([rule[3], rulecopy])
            #print(rule_adjs)
            gold_fact = None
            a0, a1 = False, False
            gold_fact = []
            for elm in rule_adjs:
                if len(elm) == 2:
                    for fact in facts:
                        if elm[0] in fact and subject in fact:
                            gold_fact.append(fact)
                            gold_rule = elm[-1]
                else:
                    for fact in facts:
                        if elm[0] in fact and subject in fact:
                            a0 = True
                            gold_fact.append(fact)
                        elif elm[1] in fact and subject in fact:
                            a1 = True
                            gold_fact.append(fact)
                    if a0 and a1:
                        gold_rule = elm[-1]
            if gold_fact is None or gold_rule is None:
                raise Exception("truncation fails")
            fout.write(f"{gold_fact[0]}. {gold_rule}. </s> {assertion}\n")
            #print(f"{gold_fact[0]}. {gold_rule}. </s> {assertion}\n")
