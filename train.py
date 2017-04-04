import numpy as np
import pandas as pd
import pylab as pl
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
from sklearn.model_selection import train_test_split
from genetic_selection import GeneticSelectionCV
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc, precision_score, recall_score, make_scorer


def read_model(path='ga_svm_clf.pkl'):
    return joblib.load(path)

def read_csv_data(file, isLabeled=True, dropNa = True):
    data = pd.read_csv(file)
    if dropNa:
        data = data.where(data != -9999).dropna().as_matrix()
    else:
        data = data.as_matrix()

    if isLabeled:
        ids = data[::, 0:1]
        labels = data[::, -1:]
        samples = data[::, 1:-1]
        return ids, labels, samples
    else:
        ids = data[::, 0:1]
        samples = data[::, 1:]
        return ids, samples

def preprocess_samples(samples, labels = None):
    X = np.array(samples)
    # Normalize feature values, use L2 norm
    X = normalize(X, axis=0, norm='l2')

    if labels is not None:
        y = np.array(labels)
        y[y == 'P'] = 1
        y[y == 'N'] = 0
        y = np.array(y, dtype=np.int)
        c, r = y.shape
        y = y.reshape(c, )
        return X,y
    else:
        return X


'''
def update_with_oracle_labels(X_train, y_train):
    ids, unlabeled_samples = read_csv_data("UnlabelledTrain.csv", isLabeled=False)
    oracle_data = pd.read_csv('oracle_labels')
    oracle_ids = np.array(oracle_data[::,0:1])
    oracle_labels = np.array(oracle_data[::,1:-1])

    labels_map = {}
    for idx,id in enumerate(oracle_ids):
        labels_map[id] = oracle_labels[idx]

    ids_indices = np.asarray([np.nonzero(ids == oracle_ids)[0][0] for id in oracle_ids])
    for i in ids_indices:
        id = ids[i]
        sample = unlabeled_samples[i]
        label = labels_map[id]
        np.append(X_train,sample)
        np.append(y_train, label)

'''

def custom_scorer(y, y_pred):
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    return min(precision,recall)

TEST_PERCENTAGE = 0.7

def main():

    siteIds, labels, samples = read_csv_data("LabelledTrain.csv")

    X, y = preprocess_samples(samples, labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_PERCENTAGE, random_state = 0)


    #sm = SMOTE(random_state=42)
    #X, y = sm.fit_sample(X,y)

    #allKnn = AllKNN(random_state=42)
    #X, y = allKnn.fit_sample(X, y)

    randomUndersampler = RandomUnderSampler(random_state=42)
    X_train,y_train = randomUndersampler.fit_sample(X_train,y_train)

    #X_train, y_train = update_with_oracle_labels(X_train, y_train)

    '''

    print("Baseline with dummy classifier: \n")
    dummy_clf = DummyClassifier(strategy='most_frequent', random_state=0)
    dummy_clf.fit(X, y)

    print("Dummy classifier: accuracy for classes " + str(dummy_clf.classes_) + " " + str(dummy_clf.class_prior_) + "\n")
    #print(dummy_clf.score(X, y))

    #Naive Bayes:
    naive_bayes_clf = GaussianNB()
    print("Naive Bayes only score: ")
    print(cross_val_score(naive_bayes_clf, X, y))


    #SVM with out feature selection
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

    '''

    '''
    print("SVM with GA feature selection: \n")
    ga_svm_clf = SVC(class_weight='balanced', probability=True)
    ga_svm_clf = GeneticSelectionCV(ga_svm_clf,
                                  cv=3,
                                  verbose=1,
                                  scoring = make_scorer(custom_scorer),
                                  n_population=100,
                                  crossover_proba=0.5,
                                  mutation_proba=0.2,
                                  n_generations=2,
                                  crossover_independent_proba=0.5,
                                  mutation_independent_proba=0.05,
                                  tournament_size=3,
                                  caching=True,
                                  n_jobs=-1)
    ga_svm_clf = ga_svm_clf.fit(X_train, y_train)
    #print(ga_svm_clf.support_)
    joblib.dump(ga_svm_clf, 'ga_svm_clf.pkl')
    print("Done")
    '''


    ga_svm_clf = read_model('./backup/ga_svm_clf_model_5.pkl')
    prediction_proba = ga_svm_clf.predict_proba(X_train)
    precision, recall, threshold = precision_recall_curve(y_train, prediction_proba[:, 1])
    prediction_labels = ga_svm_clf.predict(X_train)

    pl.clf()
    # #pl.plot(recall_3, precision_3, 'r', recall_4, precision_4, 'g', recall_5, precision_4, 'b', label='Precision-Recall curve')
    pl.plot(recall, precision, 'r', label='Precision-Recall curve')
    pl.xlabel('Recall')
    pl.ylabel('Precision')
    pl.ylim([0.0, 1.05])
    pl.xlim([0.0, 1.0])
    pl.title('Precision-Recall curves')
    pl.legend(loc="lower left")
    pl.show()
    print("Done")


    # ga_svm_clf_5 = read_model(path='./backup/ga_svm_clf_model_5.pkl')
    # mask = ga_svm_clf_5._get_support_mask()
    #
    # mask = np.asarray(mask)
    # X_train = X_train[:, mask]
    # X_test = X_test[:, mask]
    #
    # ga_svm_clf_2 =  SVC(class_weight='balanced', probability=True)
    # ga_svm_clf_2.fit(X_train, y_train)
    #
    #
    # score = custom_scorer(ga_svm_clf_2, X_test, y_test)
    #
    # prediction_proba_2 = ga_svm_clf_2.predict_proba(X_test)
    # precision_2, recall_2, threshold_2 = precision_recall_curve(y_test, prediction_proba_2[:, 1])

    '''
    ga_svm_clf_3 = read_model(path='./backup/ga_svm_clf_model_3.pkl')
    ga_svm_clf_4 = read_model(path='./backup/ga_svm_clf_model_4.pkl')
    ga_svm_clf_5 = read_model(path='./backup/ga_svm_clf_model_5.pkl')

    print("Testing with remaining labeled data. \n")
    prediction_proba_3 = ga_svm_clf_3.predict_proba(X_test)
    precision_3, recall_3, threshold_3 = precision_recall_curve(y_test, prediction_proba_3[:, 1])
    prediction_labels_3 = ga_svm_clf_3.predict(X_test)
    area_3 = auc(recall_3, precision_3)

    prediction_proba_4 = ga_svm_clf_4.predict_proba(X_test)
    precision_4, recall_4, threshold_4 = precision_recall_curve(y_test, prediction_proba_4[:, 1])
    prediction_labels_4 = ga_svm_clf_4.predict(X_test)
    area_4 = auc(recall_4, precision_4)

    prediction_proba_5 = ga_svm_clf_5.predict_proba(X_test)
    precision_5, recall_5, threshold_5 = precision_recall_curve(y_test, prediction_proba_5[:, 1])
    prediction_labels_5 = ga_svm_clf_5.predict(X_test)
    area_5 = auc(recall_5, precision_5)
    '''

    # pl.clf()
    # #pl.plot(recall_3, precision_3, 'r', recall_4, precision_4, 'g', recall_5, precision_4, 'b', label='Precision-Recall curve')
    # pl.plot(recall_2, precision_2, 'r', label='Precision-Recall curve')
    # pl.xlabel('Recall')
    # pl.ylabel('Precision')
    # pl.ylim([0.0, 1.05])
    # pl.xlim([0.0, 1.0])
    # pl.title('Precision-Recall curves')
    # pl.legend(loc="lower left")
    # pl.show()
    # print("Done")

if __name__ == "__main__":
    main()