import json
from typing import List, Dict
import csv
import random


def load_json_by_file(filename: str) -> List[Dict]:
    with open(filename) as fin:
        return json.load(fin)


def load_json_by_line(filename: str) -> List[Dict]:
    with open(filename) as fin:
        return [json.loads(line.strip()) for line in fin if line]


def save_json_by_line(filename, data):
    with open(filename, 'w') as fout:
        for line in data:
            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

def save_json_by_file(filename, data):
    with open(filename, 'w') as fout:
        json.dump(data, fout, ensure_ascii=False, indent=4)

def save_json_to_csv(filename, data):
    with open(filename, "w") as csvFile:
        writer = csv.writer(csvFile)
        head = data[0].keys()
        writer.writerow(head)

        for case in data:
            writer.writerow(case.values())

def save_list_to_csv(filename, data):
    with open(filename, "w") as csvFile:
        writer = csv.writer(csvFile)
        for d in data:
            writer.writerow(d)

def load_csv_to_list(filename):
    csv_reader = csv.reader(open(filename))
    header = next(csv_reader)
    data = []
    for line in csv_reader:
        data.append(line)
    return data

if __name__ == '__main__':
    filename = "information_extract/data/nyt/train_data.json"
    data = load_json_by_file(filename)
    save_json_by_line("information_extract/data/nyt/train.json", data)

