import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, mode):
        self.encodings = encodings
        self.labels = labels
        self.mode = mode

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.mode == "regression":
            item['labels'] = torch.tensor(float(self.labels[idx]))
        else:
            item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)
