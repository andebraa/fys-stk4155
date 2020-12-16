import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC, LinearSVC
from playsound import playsound

from sklearn.metrics import plot_confusion_matrix, accuracy_score

def pca_svm(filename):

    data = pd.read_csv('archive/' +filename,
        usecols=['label', 'tweet']
    )

    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(data['tweet'])
    vectorized=vectorized.todense()

    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)


    pca = SparsePCA()
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    clf = SVC(kernel = 'rbf')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_pred_tr = clf.predict(X_tr)


    accuracy = accuracy_score(y_te, y_pred)
    accuracy_train = accuracy_score(y_tr, y_pred_tr)


    plot_confusion_matrix(clf, X_te, y_te)
    plt.show()


if __name__ == '__main__':
    pca_svm('data_trimmed_edited_200.csv')
