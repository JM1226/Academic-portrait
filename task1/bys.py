# coding=utf-8


from sklearn.metrics import classification_report


from sklearn.externals import joblib
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingClassifier


data = []
target = []
dir = ['fuyb.txt','zyb.txt']
for path in dir:
    with open(path,'r') as f:
        for line in f:
            line = line.strip('\n')
            data.append(line)
            if path == 'zyb.txt':
                target.append(1)
            else:
                target.append(0)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)
vec = CountVectorizer()
X_train = vec.fit_transform(X_train)
X_test = vec.transform(X_test)
# mnb = RandomForestClassifier()
# mnb.fit(X_train, y_train)
# joblib.dump(mnb, 'bys.model')
bys = joblib.load('bys.model')
print(bys.predict(vec.transform(['http shenghuan.shnu.edu.cn 6c 67 c3532a355431 page htm'])))
print( 'The accuracy is',bys.score(X_test, y_test))

