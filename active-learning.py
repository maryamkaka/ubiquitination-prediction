from train import read_csv_data, read_model, preprocess_samples
import numpy as np
import matplotlib.pyplot as plt
from train import custom_scorer


def main():

    ids, samples = read_csv_data("UnlabelledTrain.csv", isLabeled=False)
    model = read_model()

    X = preprocess_samples(samples)

    decisions = np.array(np.abs(model.decision_function(X)))

    oracle_idx = decisions.argsort()[:200]
    oracle_ids = np.array(ids[oracle_idx],dtype=np.int)

    y = model.predict(X[oracle_idx])


    plt.hist(decisions)

    plt.show()

    #y = model.predict(X)

    #counts = Counter(y)
    #df = pd.DataFrame.from_dict(counts, orient='index')
    #df.plot(kind='bar')

    #plt.show()

    print("Oracle ids: " + str(oracle_ids))
    print("Done")





if __name__ == '__main__':
    main()