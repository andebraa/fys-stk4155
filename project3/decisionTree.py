import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


def decision_tree(filename, depth=100):
    """
    applies the sklearn decision tree algorithm to the dataset specified in filename.

    args:
        filename (string): location of csv datafile to be read
        depth (int): size of decision tree
    returns:
        test_acc (float): accuracy of test dataset
        train_acc (float): accuracy of train dataset
    """
    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )

    vectorizer = TfidfVectorizer() #0.8 removes all words that occur in more than 0.8 percent of tweets, which is a LOT
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)

    clf = DecisionTreeClassifier(criterion="gini", splitter='best', max_depth=depth)


    clf.fit(X_tr, y_tr)


    pred_train = clf.predict(X_tr)
    pred_test = clf.predict(X_te)

    # calculate accuracy score
    test_acc = accuracy_score(y_te, pred_test)
    train_acc = accuracy_score(y_tr , pred_train)
    print(test_acc)
    print(train_acc)

    return test_acc, train_acc




if __name__=='__main__':
    decision_tree('archive/data_trim_processed_1E5.csv')
