# サマリ

# メモ
- sklearn での SVM実装サンプル
```
from sklearn import datasets
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split

M = datasets.load_digits()
X = M.data
y = M.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
model = svm.SVC(gamma='auto',C=100.)
model.fit(X_train, y_train)

predict = model.predict(X_test)
print(metrics.accuracy_score(y_test, predict))
```