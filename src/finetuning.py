import read_data
import re
from sklearn.ensemble import RandomForestRegressor

def main():    
    predictions = {}

    for essay_set in read_data.get_essay_sets():
        
        print(f"Making Predictions for Essay Set {essay_set}")
        
        train = list(read_data.essays_by_set(essay_set))
        
        # train logic...

        test = list(read_data.essays_by_set(essay_set, "../Data/public_leaderboard_rel_2.tsv"))
        # test logic...
        predicted_scores = []
        
        for essay_id, pred_score in zip([x["Id"] for x in test], predicted_scores):
            predictions[essay_id] = round(pred_score)
    
    output_file = "../Submissions/bag_of_words_benchmark.csv"
    print("Writing submission to %s" % output_file)
    f = open(output_file, "w")
    f.write("id,essay_score\n")
    for key in sorted(predictions.keys()):
        f.write("%d,%d\n" % (key,predictions[key]))
    f.close()
    
if __name__=="__main__":
    main()