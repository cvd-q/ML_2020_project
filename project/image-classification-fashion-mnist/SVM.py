from ML_project_2020 import *
from sklearn import svm

DATA_BASE_FOLDER = '../input/ml-project-2020-dataset'
x_train = np.load(os.path.join(DATA_BASE_FOLDER, 'train.npy'))
x_valid = np.load(os.path.join(DATA_BASE_FOLDER, 'validation.npy'))
x_test = np.load(os.path.join(DATA_BASE_FOLDER, 'test.npy'))
y_train = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'train.csv'))['class'].values
y_valid = pd.read_csv(os.path.join(DATA_BASE_FOLDER, 'validation.csv'))['class'].values

clf = svm.SVC(verbose=1)
print("# training")
clf.fit(x_train,y_train)
print("# evaluating")
accuracy = clf.score(x_valid,y_valid)
print("accuracy", accuracy)

from sklearn.preprocessing import OneHotEncoder
N_CLASS = 10
enc = OneHotEncoder(handle_unknown='ignore')
cl = np.arange(N_CLASS).reshape((N_CLASS,1))
enc.fit(cl)

from sklearn.metrics import roc_auc_score
y_valid = y_valid.reshape((y_valid.shape[0], 1))
y = clf.predict(x_valid)
y = y.reshape((y.shape[0], 1))
roc_auc_score(enc.transform(y_valid).toarray(), enc.transform(y).toarray(), multi_class='ovo')