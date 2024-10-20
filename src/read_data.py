import csv


def get_essay_sets(filename = "data/train_rel_2.tsv"):
    return sorted(set(x["EssaySet"] for x in essay_reader(filename)))

def essays_by_set(essay_set, filename = "data/train_rel_2.tsv"):
    return (essay for essay in essay_reader(filename)
            if essay["EssaySet"]==essay_set)

def essay_reader(filename = "data/train_rel_2.tsv"):
    f = open(filename)
    reader = csv.reader(f, delimiter='\t' if filename.endswith('.tsv') else ',')
    header = next(reader)
    for row in reader:
        elem = dict(zip(header, row))
        for col in ["Score1", "Id", "EssaySet"]:
            if col in elem:
                elem[col] = float(elem[col])
        yield elem
    f.close()