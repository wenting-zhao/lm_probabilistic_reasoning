import json
import sys
import os
import math


def process_data(f, mode):
    content = open(f, 'r')
    result = [json.loads(jline) for jline in content.read().splitlines()]
    ret_inputs = []
    ret_labels = []
    for i, item in enumerate(result):
        context = item['english']['theory_statements']
        context = ' '.join(context)
        q = item['english']['assertion_statement']
        label = item['theory_assertion_instance']['label']
        if mode == "3cls":
            tmp = float(label)
            if tmp < 1/3:
                label = "0"
            elif 1/3 <= tmp < 2/3:
                label = "1"
            else:
                label = "2"
        elif mode == "10cls":
            label = float(label)
            label = math.floor(label * 10)
            assert 0 <= label < 10
            label = str(label)
        ret_inputs.append(context+' </s> '+q)
        ret_labels.append(label)
    return ret_inputs, ret_labels

def main():
    infile = sys.argv[1]
    outdir = sys.argv[2]
    mode = sys.argv[3]
    inputs, labels = process_data(infile, mode)
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    f = open(outdir+"/input0", "w")
    f2 = open(outdir+"/label", "w")
    for text in inputs: f.write(text+'\n')
    for label in labels: f2.write(str(label)+'\n')
    f.close()
    f2.close()

if __name__ == "__main__":
    main()

