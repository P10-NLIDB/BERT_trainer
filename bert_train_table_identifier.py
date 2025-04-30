from __future__ import annotations
import json
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset
import sqlparse
from sqlparse.sql import IdentifierList, Identifier, Function, Token, TokenList
from sqlparse.tokens import DML, Keyword, Punctuation
from typing import List, Tuple, Dict, Iterable
import re
from transformers import (
    BertTokenizerFast,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)

# Load all tables and columns for all databses in Spider


def load_global_schema(tables_json_path):
    with open(tables_json_path, 'r') as f:
        all_dbs = json.load(f)
    schema = {}
    for db in all_dbs:
        db_id = db['db_id']
        tables = db['table_names']
        cols = db['column_names']
        tbl2cols = defaultdict(list)
        for tbl_idx, col_name in cols:
            if tbl_idx >= 0:
                tbl2cols[tables[tbl_idx]].append(col_name)
        schema[db_id] = dict(tbl2cols)
    return schema

# Parse schema elements from SQL statement - First identify all tables af From
# Then all columns after select (Might not encompass everything and be somewhat fucking dumb
# But itll do for now)
# Right now it does not allow columns to be added to the second table found - and i cannot for the life of
# me get it to work - so this is what it is for now!


def _safe_name(ident: Identifier | None) -> str | None:
    """Return the real identifier name or None if it cannot be resolved."""
    if ident is None:
        return None
    try:
        return ident.get_real_name()
    except AttributeError:
        return None


def _collect_tables(tokens: TokenList) -> list[str]:
    """Return every table that appears after FROM or JOIN (recurses into groups)."""
    tables: list[str] = []
    for tok in tokens.tokens:
        if tok.is_group:
            tables.extend(_collect_tables(tok))
        if tok.ttype is Keyword and tok.value.upper() in {"FROM", "JOIN"}:
            nxt = tokens.token_next(tokens.token_index(tok), skip_ws=True)[1]
            if isinstance(nxt, IdentifierList):
                tables.extend(i.get_real_name() for i in nxt.get_identifiers())
            elif isinstance(nxt, Identifier):
                tables.append(nxt.get_real_name())
    return tables


def _add_column(tbl: str | None, col: str | None, acc: dict[str, list[str]]) -> None:
    if tbl and col:
        if col not in acc[tbl]:
            acc[tbl].append(col)


def _extract_columns(stmt: sqlparse.sql.Statement, tables: list[str]) -> dict[str, list[str]]:
    cols: dict[str, list[str]] = defaultdict(list)

    # Locate SELECT … FROM slice.
    try:
        select_idx = next(i for i, t in enumerate(stmt.tokens)
                          if t.ttype is DML and t.value.upper() == "SELECT")
        from_idx = next(i for i, t in enumerate(stmt.tokens)
                        if t.ttype is Keyword and t.value.upper() == "FROM")
    except StopIteration:
        return cols  # malformed; bail out

    slice_tokens = stmt.tokens[select_idx + 1: from_idx]

    for tok in slice_tokens:
        if tok.is_whitespace or tok.match(Punctuation, ","):
            continue

        # Drill into nested structures.
        targets = tok if isinstance(tok, (IdentifierList, Identifier, Function)) else tok.flatten()
        for sub in (targets if isinstance(targets, list) else [targets]):
            if isinstance(sub, IdentifierList):
                for ident in sub.get_identifiers():
                    _add_column(
                        ident.get_parent_name() if isinstance(ident, sqlparse.sql.Identifier) else ident._get_repr_name() if isinstance(ident, sqlparse.sql.Token) else (tables[0] if len(set(tables)) == 1 else None),
                        _safe_name(ident),
                        cols,
                    )
            elif isinstance(sub, Identifier):
                _add_column(
                    sub.get_parent_name() or (tables[0] if len(set(tables)) == 1 else None),
                    _safe_name(sub),
                    cols,
                )
            elif isinstance(sub, Function):
                inside = re.search(r"\(([^)]+)\)", sub.value)
                if not inside:
                    continue
                for arg in inside.group(1).split(","):
                    arg = arg.strip()
                    tbl, col = (arg.split(".", 1) if "." in arg else
                                ((tables[0] if len(set(tables)) == 1 else None), arg))
                    _add_column(tbl, col, cols)
    return cols


def parse_schema(sql: Iterable[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Extract tables and columns from a collection of SQL strings.

    Returns
    -------
    tables : list[str]
        Unique table names in first-appearance order.
    columns : dict[str, list[str]]
        Mapping table -> list(unique column names in order).
    """
    tables_ordered: list[str] = []
    columns: dict[str, list[str]] = defaultdict(list)

    for raw in sql:
        if not raw:
            continue
        try:
            stmt = sqlparse.parse(raw)[0]
        except Exception:
            continue  # garbage in, ignore

        tbls = _collect_tables(stmt)
        for t in tbls:
            if t not in tables_ordered:
                tables_ordered.append(t)

        for t, cols in _extract_columns(stmt, tbls).items():
            for c in cols:
                if c not in columns[t]:
                    columns[t].append(c)

    return tables_ordered, dict(columns)



def load_questions(jsonl_path):
    recs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            recs.append(json.loads(line))
    return recs

# Build positive and negative examples for training - ration controls the ratio of negative examples to
# positives


def build_linking_examples(recs, global_schema, neg_ratio=1):
    examples = []
    for r in recs:
        q = r['question']
        db = r['db_id']
        local_t, local_c = parse_schema(r['queries'])

        pos = local_t + [
            f"{t}.{c}"
            for t in local_t
            for c in local_c.get(t, [])
        ]

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


# Preproccesing of dataset for training
class LinkDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length=64):
        self.ex = examples
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ex)

    def __getitem__(self, idx):
        q = self.ex[idx]['question']
        e = self.ex[idx]['element']
        lbl = float(self.ex[idx]['label'])
        enc = self.tok(
            q, e,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'labels': torch.tensor(lbl)
        }

# Link model trainer with dataset


def train_linker(examples, output_dir='linker_out'):
    tok = BertTokenizerFast.from_pretrained('bert-base-uncased')
    ds = LinkDataset(examples, tok)
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=1,
        problem_type='regression'
    )
    args = TrainingArguments(
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

# Test if model works - theta is just pulled from thin air  - do not put to much thought into this


def prune_elements(question, candidate_elements, model, tokenizer, theta=0.5):
    enc = tokenizer(
        [question]*len(candidate_elements),
        candidate_elements,
        truncation=True,
        padding=True,
        return_tensors='pt'
    )
    logits = model(**enc).logits.squeeze(-1)
    probs = torch.sigmoid(logits).tolist()
    return {
        e: p for e, p in zip(candidate_elements, probs)
        if p >= theta
    }


if __name__ == '__main__':
    TABLES_JSON = './tables.json'
    QUESTIONS_JL = './questions.jsonl'

    global_schema = load_global_schema(TABLES_JSON)
    recs = load_questions(QUESTIONS_JL)

    examples = build_linking_examples(recs, global_schema, neg_ratio=1)
    # model, tok = train_linker(examples, output_dir='linker_out')
    model = BertForSequenceClassification.from_pretrained("./linker_out/")
    tokenizer = BertTokenizerFast.from_pretrained("./linker_out/")
    rec0 = recs[0]
    local_t, local_c = parse_schema(rec0['queries'])
    elems = local_t + [f"{t}.{c}" for t in local_t for c in local_c[t]]
    all_tbls = list(global_schema["department_management"].keys())
    all_cols = [
        f"{t}.{c}" for t in all_tbls for c in global_schema["department_management"][t]]
    all_elems = all_tbls + all_cols

    pruned = prune_elements(rec0['question'], all_elems,
                            model, tokenizer, theta=0.72)

    print("Question", rec0["question"], "Kept edges:",
          pruned, "Start Edges:", elems)
