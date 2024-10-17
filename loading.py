import csv
import os

training_path = path = os.path.join(os.getcwd(), 'data', 'train_rel_2.tsv')

def get_essay_sets(filename = training_path):
    return sorted(set(x["EssaySet"] for x in essay_reader(filename)))

def essays_by_set(essay_set, filename = training_path):
    return (essay for essay in essay_reader(filename)
            if essay["EssaySet"]==essay_set)

def essay_reader(filename = training_path):
    f = open(filename)
    reader = csv.reader(f, delimiter = '\t')
    header = next(reader)
    for row in reader:
        elem = dict(zip(header, row))
        for col in ["EssayScore", "Id", "EssaySet"]:
            if col in elem:
                elem[col] = float(elem[col])
        yield elem
    f.close()