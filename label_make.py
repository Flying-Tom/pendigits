import numpy as np
from algorithms.SVM import SVM
from imblearn.over_sampling import RandomOverSampler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import metrics


def get_data(train=True):
    if train:
        data_file = 'data/anonymous.tra'
        all_data = np.loadtxt(data_file, dtype=int, delimiter=',')
        inputs, targets = all_data[:, :-1], all_data[:, -1]
        return inputs, targets
    else:
        data_file = 'data/anonymous.tes'
        all_data = np.loadtxt(data_file, dtype=int, delimiter=',')
        inputs = all_data[:, :]
        return inputs


ros = RandomOverSampler(random_state=0)


X_train, y_train = get_data(train=True)
# X_train, y_train = ros.fit_resample(X_train, y_train)
X_test = get_data(train=False)


clf = make_pipeline(StandardScaler(), SVC(C=0.8, kernel='rbf', decision_function_shape='ovr', class_weight='balanced'))
# clf = SVC(gamma="auto")
# clf = SVR(kernel="poly",  gamma="auto", degree=3, epsilon=0.1, coef0=1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_train)
print(f"Train Error|| Accuracy: {100*metrics.accuracy_score(y_train, y_pred):>0.1f}% \n")
np.savetxt('output_191250004.txt', clf.predict(X_test), fmt='%d')
