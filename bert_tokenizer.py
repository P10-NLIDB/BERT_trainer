from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
import torch, json

questions, db_ids = [], []
with open('questions.jsonl','r') as f:
    for line in f:
        rec = json.loads(line)
        questions.append(rec['question'])
        db_ids.append(rec['db_id'])


db_list = sorted(set(db_ids))
label2id = {db:i for i, db in enumerate(db_list)}
labels = [label2id[db] for db in db_ids]

with open("db_classifier/label2id.json","w") as f:
    json.dump(label2id, f)

tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
encodings = tok(questions, truncation=True, padding=True)

class QDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) 
                for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

ds = QDataset(encodings, labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', 
                                                       num_labels=len(db_ids))
args = TrainingArguments(
    output_dir='out', per_device_train_batch_size=16,
    num_train_epochs=3, logging_steps=50
)
trainer = Trainer(model=model, args=args, train_dataset=ds)
trainer.train()

model.save_pretrained('db_classifier')
tok.save_pretrained('db_classifier')
