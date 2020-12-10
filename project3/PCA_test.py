import numpy as np
import pandas as pd
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from playsound import playsound

from sklearn.metrics import plot_confusion_matrix, accuracy_score

def pca_svm(filename):

    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )
    print(data)



    vectorizer = TfidfVectorizer( min_df = 1, max_df = 0.8)
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)


    pca = TruncatedSVD(n_components = 69)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    clf = SVC(kernel = 'linear')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    sounds = ['Not Gay Sex.mp3', 'Objection Hearsay.mp3', 'Rock Flag and Eagle.mp3', 'The good lords going down on me.mp3']

    playsound(sounds[np.random.randint(0,4)])
    plot_confusion_matrix(clf, X_te, y_te)
    accuracy = accuracy_score(y_te, y_pred)


if __name__ == '__main__':
    pca_svm('data_trim_processed.csv')
