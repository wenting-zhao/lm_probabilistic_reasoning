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


class RegressionTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.MSELoss()
        #loss_fct = torch.nn.L1Loss()
        #loss_fct = utils.r2_loss
        loss = loss_fct(logits.view(-1), labels.float().view(-1))
        return (loss, outputs) if return_outputs else loss


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_model_dir", default="./results", type=str,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--batch_size", '-b', type=int, required=True,
                        help="batch size per gpu.")
    parser.add_argument("--epoch", '-epoch', type=int, required=True,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--learning_rate", '-lr', default=1e-5, type=float,
                        help="The learning rate for fine-tuning.")
    parser.add_argument("--model_dir", default="LIAMF-USP/roberta-large-finetuned-race", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--data_dir", default="data/random", type=str,
                        help="data directory")
    parser.add_argument("--test_only", action='store_true',
                        help="test only mode")


    args = parser.parse_args()
    return args
 
def get_data(directory, split):
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
             labels.append(line)
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
train_texts, train_labels = get_data(directory, 'train')
valid_texts, valid_labels = get_data(directory, 'dev')
test_texts, test_labels = get_data(directory, 'test')

tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)
model = RobertaForSequenceClassification.from_pretrained(args.model_dir, num_labels=1)
if args.model_dir == "LIAMF-USP/roberta-large-finetuned-race":
    new_toks = [str(round(i*0.1, 1)) for i in range(1, 10)]
    tokenizer.add_tokens(new_toks)
    print("added tokens:", new_toks)
    model.resize_token_embeddings(len(tokenizer))

train_encodings = get_encodings(directory, 'train') 
val_encodings = get_encodings(directory, 'valid') 
test_encodings = get_encodings(directory, 'test') 

train_dataset = dataset.Dataset(train_encodings, train_labels)
val_dataset = dataset.Dataset(val_encodings, valid_labels)
test_dataset = dataset.Dataset(test_encodings, test_labels)

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

metric = load_metric("/mnt/beegfs/bulk/mirror/wz346/probabilistic_reasoning/utils/metrics/regress.py")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    logits = logits.squeeze()
    return metric.compute(predictions=logits, references=labels)

trainer = RegressionTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    tokenizer=tokenizer,
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=val_dataset,            # evaluation dataset
    compute_metrics=compute_metrics
)

if not args.test_only:
    trainer.train()
res = trainer.predict(test_dataset)
#res = trainer.predict(train_dataset)
print(res[-1])
