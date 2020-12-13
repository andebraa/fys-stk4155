import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def adaboost(filename, lamda):
    data = pd.read_csv('archive/' +filename,
        usecols=['label', 'tweet']
    )

    # data = pd.read_csv(filename, usecols=['tweet', 'label'])
    corpus = data['tweet']
    labels = data['label']

    # create bag of words
    vectorizer = TfidfVectorizer()
    bow = vectorizer.fit_transform(corpus)

    # split in train and test data
    bow_train, bow_test, labels_train, labels_test = train_test_split(bow, labels,
                                                                      test_size=0.3)

    # AdaBoost
    lamda = 1
    eta = [0.9]



    ada_clf = AdaBoostClassifier(DecisionTreeClassifier(criterion='gini',
                                                        max_depth=1),
                                 n_estimators=100,
                                 algorithm='SAMME.R',
                                 learning_rate=eta[0])
    ada_clf.fit(bow_train, labels_train)

    # predict sentiment of tweetss
    ada_pred_train = ada_clf.predict(bow_train)
    ada_pred_test = ada_clf.predict(bow_test)

    # calculate accuracy score
    test_acc = accuracy_score(labels_test, ada_pred_test)
    train_acc = accuracy_score(labels_train , ada_pred_train)
    print(test_acc)
    print(train_acc)



if __name__ == '__main__':
    adaboost('data_trim_edit_1E5.csv', 1)
