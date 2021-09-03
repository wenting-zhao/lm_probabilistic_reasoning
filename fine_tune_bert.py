import argparse
import pickle
import os
from datasets import load_metric
import torch
from transformers import RobertaTokenizer
from transformers import RobertaForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import r2_score
from utils import dataset
from utils import utils
from ray.tune.schedulers import AsyncHyperBandScheduler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", default="./results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", '-b', type=int, required=True,
                        help="batch size per gpu.")
    parser.add_argument("--epoch", '-epoch', type=int, required=True,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--learning_rate", '-lr', default=5e-5, type=float,
                        help="The learning rate for fine-tuning.")
    parser.add_argument("--model_dir", default="LIAMF-USP/roberta-large-finetuned-race", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--data_dir", default="data/random", type=str,
                        help="data directory")
    parser.add_argument("--mode", "-m", default="regression", type=str, choices=["3cls", "10cls", "regression"],
                        help="data directory")
    parser.add_argument("--test_only", action='store_true',
                        help="test only mode")
    parser.add_argument("--hyperparameter_search", "-hs", action='store_true',
                        help="hyperparameter search")
    parser.add_argument("--not_special_tokens", "-st", action='store_true',
                        help="whether not to use special tokens")


    args = parser.parse_args()
    return args
 
def get_data(directory, split, mode):
    feats = []
    with open(f"{directory}/input0_{split}", 'r') as fin:
         for line in fin:
             line = line.strip()
             if line == '': continue
             feats.append(line)
    labels = []
    with open(f"{directory}/label_{split}", 'r') as fin:
         for line in fin:
             line = line.strip()
             if line == '': continue
             if mode == "regression":
                 label = line
             elif mode == "3cls":
                 label = float(line)
                 if label < 1/3:
                     label = "0"
                 elif 1/3 <= label < 2/3:
                     label = "1"
                 else:
                     label = "2"
             elif mode == "10cls":
                 label = line[line.find('.')+1]
             labels.append(label)
    return feats, labels

def get_encodings(directory, split):
    if not os.path.isfile(f"{directory}/{split}_encodings.p"):
        encodings = tokenizer(train_texts, truncation=True, padding=True)
        pickle.dump( encodings, open( f"{directory}/{split}_encodings.p", "wb" ) )
    else:
        encodings = pickle.load(open( f"{directory}/{split}_encodings.p", "rb" ))
    return encodings

args = get_args()
directory = args.data_dir
train_texts, train_labels = get_data(directory, 'train', args.mode)
valid_texts, valid_labels = get_data(directory, 'dev', args.mode)
test_texts, test_labels = get_data(directory, 'test', args.mode)

tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
if args.mode == "regression":
    num_labels = 1
elif args.mode == "3cls":
    num_labels = 3
elif args.mode == "10cls":
    num_labels = 10
model = RobertaForSequenceClassification.from_pretrained(args.model_dir, num_labels=num_labels)

if (args.model_dir == "LIAMF-USP/roberta-large-finetuned-race" or args.model_dir == "roberta-large") and not args.not_special_tokens:
    new_toks = [str(round(i*0.1, 1)) for i in range(1, 10)]
    tokenizer.add_tokens(new_toks)
    print("added tokens:", new_toks)
    model.resize_token_embeddings(len(tokenizer))

train_encodings = get_encodings(directory, 'train') 
val_encodings = get_encodings(directory, 'valid') 
test_encodings = get_encodings(directory, 'test') 

train_dataset = dataset.Dataset(train_encodings, train_labels, args.mode)
val_dataset = dataset.Dataset(val_encodings, valid_labels, args.mode)
test_dataset = dataset.Dataset(test_encodings, test_labels, args.mode)

training_args = TrainingArguments(
    learning_rate=args.learning_rate,
    output_dir=args.output_model_dir,
    num_train_epochs=args.epoch,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=64,
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=500
)

if args.mode == "regression":
    metric = load_metric("utils/metrics/regress.py")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.squeeze()
        return metric.compute(predictions=logits, references=labels)
else:
    metric = load_metric("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

if args.hyperparameter_search:
    best_trial = trainer.hyperparameter_search(
        direction="maximize",
        backend="ray",
        n_trials=20)  # number of hyperparameter samples
else:
    if not args.test_only:
        trainer.train()
    res = trainer.predict(test_dataset)
    print(res[-1])
