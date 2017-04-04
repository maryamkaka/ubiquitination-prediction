from train import read_csv_data, read_model, preprocess_samples
import  pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from train import custom_scorer
import numpy as np

def main():
    ids, samples = read_csv_data("blindTest_feature.csv", isLabeled=False)
    X = preprocess_samples(samples)
    model = read_model('./backup/ga_svm_clf_model_4.pkl')
    y = model.predict(X)
    y = np.asarray(y, dtype=np.object)
    y[y == 1] = 'P'
    y[y == 0] = 'N'

    counts = Counter(y)
    df = pd.DataFrame.from_dict(counts, orient='index')
    df.plot(kind='bar')

    #plt.show()

    ids= np.array(ids, dtype=np.int)

    prediction_map = {}
    for i, id in enumerate(ids):
        prediction_map[id[0]] = y[i]

    all_ids = read_csv_data("blindTest_feature.csv", isLabeled=False, dropNa=False)[0]
    all_ids = np.array(all_ids, dtype=np.int)
    with open('prediction_result', 'w') as f:
        for i, id in enumerate(all_ids):
            if id[0] in prediction_map:
                f.write(str(id[0])+': '+prediction_map[id[0]] + '\n')
            else:
                f.write(str(id[0])+': ' + 'P' + '\n')

if __name__ == '__main__':
    main()