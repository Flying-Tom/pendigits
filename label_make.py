import numpy as np
from algorithms.SVM import SVM
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
smo = SMOTE(random_state=42)


# X_train, y_train = smo.fit_sample(X, y)
X, y = get_data(train=True)
X_train, y_train = smo.fit_resample(X, y)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)
X_tocal = get_data(train=False)


#clf = make_pipeline(StandardScaler(), SVC(C=0.5, kernel='rbf',  class_weight='balanced'))
clf = SVC(C=0.4, kernel='rbf',  class_weight='balanced')
clf.fit(X_train, y_train)

# y_pred = clf.predict(X_test)
# print(f"Train Error|| Accuracy: {100*metrics.accuracy_score(y_test, y_pred):>0.1f}% \n")
np.savetxt('output_191250004.txt', clf.predict(X_tocal), fmt='%d')
