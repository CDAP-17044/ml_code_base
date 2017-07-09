#import numpy as np
#import matplotlib.pyplot as plt
import csv
import sys
#from sklearn import svm
#from sklearn.multiclass import OneVsRestClassifier
#from sklearn.preprocessing import MultiLabelBinarizer
#from sklearn.feature_extraction import DictVectorizer

train_data = dict()
train_data_labels = list()
train_data_list = []
train_data_labels_list = []

with open('C:\\Users\\my\\Documents\\Research\\dataset-full.csv', 'r') as f:
	reader = csv.reader(f) 
	for row in reader:
		print(row)
		for idx in range(len(row)):
			if idx == 0:
				train_data['Time'] = int(row[idx])
			if idx == 1:
				train_data['Current'] = float(row[idx])
			if idx == 2:
				train_data['Voltage'] = float(row[idx])
			if idx == 3:
				train_data_labels.append(row[idx])
				
		train_data_list.append(train_data)
		train_data_labels_list.append(train_data_labels)
		train_data = dict()
		train_data_labels = list()
		
C = 0.8
dict_vectorizer = DictVectorizer(sparse=False)
train_data_trasformed = dict_vectorizer.fit_transform(train_data_list)
test_vector_transformed = dict_vectorizer.transform(test_vector)

y_enc = dict_vectorizer.fit_transform(train_data_labels_list)			
train_vector = OneVsRestClassifier(svm.SVC(probability=True))
classifier_rbf = train_vector.fit(train_data_trasformed, y_enc)

prediction = classifier_rbf.predict(test_vector_transformed)
print("Predicted line: \n")

users = self.parse_prediction(mlb.inverse_transform(prediction))
print(users)		
