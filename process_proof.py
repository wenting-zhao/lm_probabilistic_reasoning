import json
import sys
import os


def process_data(f):
    content = open(f, 'r')
    result = [json.loads(jline) for jline in content.read().splitlines()]
    ret_inputs = []
    ret_labels = []
    for i, item in enumerate(result):
        context = item['english']['theory_statements']
        context = ' '.join(context)
        q = item['english']['assertion_statement']
        label = item['theory_assertion_instance']['label']
        ret_inputs.append(context+' <sep> '+q)
        ret_labels.append(label)
    return ret_inputs, ret_labels

def main():
    infile = sys.argv[1]
    outdir = sys.argv[2]
    inputs, labels = process_data(infile)
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

