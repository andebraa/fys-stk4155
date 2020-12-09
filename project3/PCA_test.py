import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
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


    pca = PCA(n_components = 69)
    X_tr = pca.fit_transform(X_tr)
    X_te = pca.transform(X_te)
    explained_variance = pca.explained_variance_raio_
    clf = SVC(kernel = 'linear')
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)

    plot_confusion_matrix(clf, X_te, y_te)
    accuracy = accuracy_score(y_te, y_pred)
