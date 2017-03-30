import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.externals import joblib
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score


from genetic_selection import GeneticSelectionCV

def main():
    labelled_data = pd.read_csv('LabelledTrain.csv')
    labelled_data= labelled_data.where(labelled_data != -9999)
    labelled_data = labelled_data.dropna()

    labelled_data = labelled_data.as_matrix()
    labels = labelled_data[::, -1:]
    features = labelled_data[::, 1:-1]
    siteIds = labelled_data[::, 0:1]

    X = np.array(features[0:1000,::])
    y = np.array(labels[0:1000,::])

    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
    dummy_clf.fit(X, y)

    print("Dummy classifier: accuracy for classes " + str(dummy_clf.classes_) + " " + str(dummy_clf.class_prior_))
    #print(dummy_clf.score(X, y))

    estimator = SVC(class_weight='balanced')

    ga_clf = GeneticSelectionCV(estimator,
                                  cv=5,
                                  verbose=1,
                                  scoring="accuracy",
                                  n_population=50,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=10,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    ga_clf = ga_clf.fit(X, y)

    print(ga_clf.support_)
    joblib.dump(ga_clf, 'model.pkl')

if __name__ == "__main__":
    main()