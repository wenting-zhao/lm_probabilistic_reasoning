import sys

directory = sys.argv[1]
num_files = int(sys.argv[2])
train_ratio = float(sys.argv[3])
dev_ratio = float(sys.argv[4])
assert train_ratio + dev_ratio < 1
train_end = int(num_files * train_ratio)
dev_end = train_end + int(num_files * dev_ratio)
print(train_end, dev_end)

def split_file(path, fn):
    with open(f'{path}/{fn}', 'r') as fin:
        with open(f'train.{fn}', 'w') as train_out:
            with open(f'valid.{fn}', 'w') as dev_out:
                with open('test.{fn}', 'w') as test_out:
                    for i, line in enumerate(fin):
                        if i <= train_end:
                            train_out.write(line)
                        elif train_end < i <= dev_end:
                            dev_out.write(line)
                        else:
                            test_out.write(line)

split_file(directory, 'input0')
split_file(directory, 'label')
