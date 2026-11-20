"""Offline Ragas evaluation. Input JSONL rows: question, answer, contexts, ground_truth."""
import json, argparse
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

def main(path):
    rows=[json.loads(x) for x in open(path,encoding="utf-8") if x.strip()]
    ds=Dataset.from_list(rows)
    result=evaluate(ds,metrics=[faithfulness,answer_relevancy,context_precision,context_recall])
    print(result)
    result.to_pandas().to_csv("ragas-report.csv",index=False)

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("dataset"); main(p.parse_args().dataset)
