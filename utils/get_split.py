import sys

feat_fn = sys.argv[1]
label_fn = sys.argv[2]
num_files = int(sys.argv[3])
train_ratio = float(sys.argv[4])
dev_ratio = float(sys.argv[5])
assert train_ratio + dev_ratio < 1
train_end = int(num_files * train_ratio)
dev_end = train_end + int(num_files * dev_ratio)
print(train_end, dev_end)

def split_file(filename):
    with open(filename, 'r') as fin:
        with open(filename+'_train', 'w') as train_out:
            with open(filename+'_dev', 'w') as dev_out:
                with open(filename+'_test', 'w') as test_out:
                    for i, line in enumerate(fin):
                        if i <= train_end:
                            train_out.write(line+'\n')
                        elif train_end < i <= dev_end:
                            dev_out.write(line+'\n')
                        else:
                            test_out.write(line+'\n')

split_file(feat_fn)
split_file(label_fn)
