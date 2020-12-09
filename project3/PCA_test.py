import numpy as np
from sklearn.decomposition import PCA

def pca_svm(filename):

    data = pd.read_csv(filename,
        usecols=['label', 'tweet']
    )
    print(data)



    vectorizer = TfidfVectorizer( min_df = 1, max_df = 0.8)
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))


    X_tr, X_te, y_tr, y_te = train_test_split(vectorized, data['label'],test_size = 0.2)


    print(np.shape(X_tr))
    print(np.shape(y_tr))
    print(np.shape(X_te))
    print(np.shape(y_te))
    print(type(X_te))
    clf.fit(X_tr, y_tr)

    pred_te = clf.predict(X_te)
    pred_train = clf.predict(X_tr)
    print("-----------------------------------")
    print(confusion_matrix(y_te, pred_te))
    print(classification_report(y_te, pred_te))
    print(accuracy_score(y_te, pred_te))
    print("-----------------------------------")
    print(confusion_matrix(y_tr, pred_train))
    print(classification_report(y_tr, pred_train))
    print(accuracy_score(y_tr, pred_train))

    pca = PCA(n_components = 69)
