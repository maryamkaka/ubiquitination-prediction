import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.naive_bayes import GaussianNB
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import AllKNN, RandomUnderSampler

from genetic_selection import GeneticSelectionCV

# Control the number of samples to use for training for performance reasons
NUM_SAMPLES_TO_USE = 5000


def read_csv_data(file, isLabeled=True):
    data = pd.read_csv(file)
    data = data.where(data != -9999).dropna().as_matrix()

    if isLabeled:
        ids = data[::, 0:1]
        labels = data[::, -1:]
        samples = data[::, 1:-1]
        return ids, labels, samples
    else:
        ids = data[::, 0:1]
        samples = data[::, 1:]
        return ids, samples


def main():
    siteIds, labels, samples = read_csv_data("LabelledTrain.csv")

    X = np.array(samples[0:NUM_SAMPLES_TO_USE, ::])
    # Normalize feature values, use L1 norm
    X = normalize(X, axis=0, norm='l1')
    y = np.array(labels[0:NUM_SAMPLES_TO_USE, ::])
    c, r = y.shape
    y = y.reshape(c, )

    # sm = SMOTE(random_state=42)
    # X, y = sm.fit_sample(X,y)

    # allKnn = AllKNN(random_state=42)
    # X, y = allKnn.fit_sample(X, y)

    randomUndersampler = RandomUnderSampler(random_state=42)
    X, y = randomUndersampler.fit_sample(X, y)

    print("Baseline with dummy classifier: \n")
    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
    dummy_clf.fit(X, y)

    print("Dummy classifier: accuracy for classes " + str(dummy_clf.classes_) +
          " " + str(dummy_clf.class_prior_) + "\n")
    # print(dummy_clf.score(X, y))

    # Naive Bayes:
    naive_bayes_clf = GaussianNB()
    print("Naive Bayes only score: ")
    print(cross_val_score(naive_bayes_clf, X, y))


    # SVM with out feature selection
    svm_clf = SVC(class_weight='balanced')
    print("SVM only score: ")
    print(cross_val_score(svm_clf, X, y))


    ga_naive_bayes_clf = GaussianNB()
    print("Naive Bayes with GA feature selection: \n")
    ga_naive_bayes_clf = SVC(class_weight='balanced')
    ga_naive_bayes_clf = GeneticSelectionCV(ga_naive_bayes_clf,
                                  cv=3,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=int(0.1 * y.shape[0]),
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=20,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    ga_naive_bayes_clf = ga_naive_bayes_clf.fit(X, y)
    print(ga_naive_bayes_clf.support_)


    print("SVM with GA feature selection: \n")
    ga_svm_clf = SVC(class_weight='balanced')
    ga_svm_clf = GeneticSelectionCV(ga_svm_clf,
                                  cv=3,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=int(0.1 * y.shape[0]),
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=50,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    ga_svm_clf = ga_svm_clf.fit(X, y)
    print(ga_svm_clf.support_)
    joblib.dump(ga_svm_clf, 'ga_svm_clf.pkl')


if __name__ == "__main__":
    main()