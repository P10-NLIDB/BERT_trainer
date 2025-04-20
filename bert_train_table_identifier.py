import json
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

def load_global_schema(tables_json_path):
    with open(tables_json_path, 'r') as f:
        all_dbs = json.load(f)
    schema = {}
    for db in all_dbs:
        db_id = db['db_id']
        tables = db['table_names']
        cols   = db['column_names'] 
        tbl2cols = defaultdict(list)
        for tbl_idx, col_name in cols:
            if tbl_idx >= 0:
                tbl2cols[tables[tbl_idx]].append(col_name)
        schema[db_id] = dict(tbl2cols)
    return schema

def parse_schema(schema_str):
    """
    Input: schema string like
      "department : creation , name , ... | head : head_id , years_old , ... | management : ..."
    Output: 
      tables = ["department","head","management"]
      columns = {
        "department": ["creation","name",...],
        "head":       ["head_id","years_old",...],
        "management":["department_id","head_id",...]
      }
    """
    tables, columns = [], {}
    for part in schema_str.split('|'):
        if ':' not in part:
            continue
        table, collist = part.split(':', 1)
        table = table.strip()
        cols = [c.strip() for c in collist.split(',') if c.strip()]
        tables.append(table)
        columns[table] = cols
    return tables, columns



def load_questions(jsonl_path):
    recs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

def build_linking_examples(recs, global_schema, neg_ratio=1):
    examples = []
    for r in recs:
        q   = r['question']
        db  = r['db_id']
        local_t, local_c = parse_schema(r['schema'])

        pos = local_t + [f"{t}.{c}" for t in local_t for c in local_c[t]]

        all_tbls = list(global_schema[db].keys())
        all_cols = [f"{t}.{c}" for t in all_tbls for c in global_schema[db][t]]
        all_elems = all_tbls + all_cols

        neg_cand = [e for e in all_elems if e not in pos]

        target_neg = len(pos) * neg_ratio
        k = min(len(neg_cand), target_neg)

        neg = random.sample(neg_cand, k=k) if k > 0 else []

        for e in pos:
            examples.append({'question': q, 'element': e, 'label': 1})
        for e in neg:
            examples.append({'question': q, 'element': e, 'label': 0})

    return examples



class LinkDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.ex = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        q  = self.ex[idx]['question']
        e  = self.ex[idx]['element']
        lbl = float(self.ex[idx]['label'])
        enc = self.tok(
            q, e,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels':         torch.tensor(lbl)
        }
    

def train_linker(examples, output_dir='linker_out'):
    tok   = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ds    = LinkDataset(examples, tok)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1,
        problem_type='regression'
    )
    args  = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=32,
        num_train_epochs=3,
        learning_rate=3e-5,
        logging_steps=100
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tok.save_pretrained(output_dir)
    return model, tok

def prune_elements(question, candidate_elements, model, tokenizer, theta=0.5):
    enc = tokenizer(
        [question]*len(candidate_elements),
        candidate_elements,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    logits = model(**enc).logits.squeeze(-1)
    probs  = torch.sigmoid(logits).tolist()
    return {
        e: p for e, p in zip(candidate_elements, probs)
        if p >= theta
    }


if __name__ == '__main__':
    TABLES_JSON = './tables.json'
    QUESTIONS_JL= './questions.jsonl'

    global_schema = load_global_schema(TABLES_JSON)
    recs          = load_questions(QUESTIONS_JL)

    examples = build_linking_examples(recs, global_schema, neg_ratio=1)
    model, tok = train_linker(examples, output_dir='linker_out')

    rec0 = recs[0]
    local_t, local_c = parse_schema(rec0['schema'])
    elems = local_t + [f"{t}.{c}" for t in local_t for c in local_c[t]]

    pruned = prune_elements(rec0['question'], elems, model, tok, theta=0.5)
    
    print("Question", rec0["question"], "Kept edges:", pruned)