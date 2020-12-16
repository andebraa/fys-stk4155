from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def random_forest(filename):
    """
    applies the sklearn Random forest algorithm to the dataset specified in filename.

    args:
        filename (string): location of csv datafile to be read
    returns:
        test_acc (float): accuracy of test dataset
        train_acc (float): accuracy of train dataset
    """

    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )

    vectorizer = TfidfVectorizer( min_df = 1, max_df = 0.8)
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())

    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    clf = RandomForestClassifier(n_estimators = 300, criterion="gini", n_jobs=-1)
    clf.fit(X_tr, y_tr)

    pred_te = clf.predict(X_te)
    pred_train = clf.predict(X_tr)

    # calculate accuracy score
    test_acc = accuracy_score(y_te, pred_test)
    train_acc = accuracy_score(y_tr , pred_train)
    print(test_acc)
    print(train_acc)

    return test_acc, train_acc


if __name__=='__main__':
    random_forest('archive/data_trim_edit_1E4.csv')
