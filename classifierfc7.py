# Using SVM OneVsRestClassifier
# Each class label F1 score is also obtained 
# We split the training data into 80, 20 , This is done to test our model f! score. Each class F1 score is also obtained.

import numpy as np
import pandas as pd 
import time
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score

path_to_data = '/Users/samantha/yelp/'

photo_train = pd.read_csv(path_to_data+'train_photo_to_biz_ids.csv')
train_photo_to_biz = pd.read_csv(path_to_data+'train_photo_to_biz_ids.csv', index_col='photo_id')

train_features = pd.read_csv(path_to_data+"train_biz_fc7features.csv")
test_features  = pd.read_csv(path_to_data+"test_biz_fc7features.csv")

y_train = train_features['label'].values
X_train = train_features['feature vector'].values
X_test = test_features['feature vector'].values

def getLabelArray(str_label):
    str_label = str_label[1:-1]
    str_label = str_label.split(',')
    return [int(x) for x in str_label if len(x)>0]

def printstat():
	statistics = pd.DataFrame(columns=[ "attribuite "+str(i) for i in range(9)]+['num_biz'], index = ["biz count", "biz ratio"])
	statistics.loc["biz count"] = np.append(np.sum(y_ppredict, axis=0), len(y_ppredict))
	pd.options.display.float_format = '{:.0f}%'.format
	statistics.loc["biz ratio"] = statistics.loc["biz count"]*100/len(y_ppredict)
	statistics

def getFeatureVector(str_feature):
    str_feature = str_feature[1:-1]
    str_feature = str_feature.split(',')
    return [float(x) for x in str_feature]

y_train = np.array([getLabelArray(y) for y in train_features['label']])
X_train = np.array([getFeatureVector(x) for x in train_features['feature vector']])
X_test = np.array([getFeatureVector(x) for x in test_features['feature vector']])
t=time.time()

binarizer = MultiLabelBinarizer()
y_ptrain= binarizer.fit_transform(y_train)  #Convert list of labels to binary matrix

random_state = np.random.RandomState(0)
#Splitting the training data into train and test (80 and 20)
X_ptrain, X_ptest, y_ptrain, y_ptest = train_test_split(X_train, y_ptrain, test_size=.2,random_state=random_state)
svmclassifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
svmclassifier.fit(X_ptrain, y_ptrain)

y_ppredict = svmclassifier.predict(X_ptest)

def printF1scores():
	print "Total F1 score: ", f1_score(y_ptest, y_ppredict, average='micro') 
	print "Individual F1 score: ", f1_score(y_ptest, y_ppredict, average=None)

print "Elapsed time: ", "{0:.1f}".format(time.time()-t), "sec"

printstat()

printF1scores()

t = time.time()

binarizer = MultiLabelBinarizer()
#labels list is converted to binary matrix
y_train= binarizer.fit_transform(y_train) 

random_state = np.random.RandomState(0)
svmclassifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))
svmclassifier.fit(X_train, y_train)

y_predict = svmclassifier.predict(X_test)

#Binary matrix is converted to labels
y_predict_label = binarizer.inverse_transform(y_predict) 

print "Elaspsed Time: ", "{0:.1f}".format(time.time()-t), "sec"

tdf  = pd.read_csv(path_to_data+"test_biz_fc7features.csv")
df = pd.DataFrame(columns=['business_id','labels'])

for i in range(len(tdf)):
    biz = tdf.loc[i]['business']
    label = y_predict_label[i]
    label = str(label)[1:-1].replace(",", " ")
    df.loc[i] = [str(biz), label]

with open(path_to_data+"submission_fc7.csv",'w') as file67:
    df.to_csv(file67, index=False)   

printstat() 
