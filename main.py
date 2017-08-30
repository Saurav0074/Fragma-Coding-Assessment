## Author : Saurav Jha, CSED Undergrad, MNNIT Allahabad ##
## 10th Aug, 2017 ##

# import necessary modules and packages
import numpy as np 
import pandas as pd 
import math
import csv 

from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import  RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV

from collections import Counter
from imblearn.over_sampling import SMOTE

# function to encode non-numeric values into set of numeric labels
def encoding_columns(X):
	encoder = LabelEncoder()
	encoder.fit(X)
	X = encoder.transform(X) # transform X based on distinct labels found
	#print(format(Counter(X))) # print count of each label
	return X

############### Grid Search based parameter tuning ###################
def tune_parameters(alg, X, y):
	'''
	# Parameter sets for Gradient Boosting Classifier
	parameters = {'loss': ['deviance', 'exponential'],
					'learning_rate': [0.1, 0.3, 0.5, 0.08],
					'n_estimators': [100, 125, 150, 200],
					'min_samples_split': [2, 3, 5],
					'min_samples_leaf': [1, 5, 8]
					}
	
	# Parameter sets for Random Forest and Extra Trees Classifier
	parameters = {'n_estimators':[10, 12, 15, 20],
				'max_features':['log2', 'sqrt', 'auto'],
				'criterion': ['entropy', 'gini'], 
				'max_depth':[2, 3, 5, 10],
				'min_samples_split': [2, 3, 5],
				'min_samples_leaf': [1,5,8]
				}
	'''
	# Parameter sets for SVM
	parameters = {'C':[0.6, 0.8, 1.0, 1.2, 1.4, 1.6]}

	grid = GridSearchCV(alg, parameters, cv = 10, 
						verbose = 0, scoring = 'f1_weighted')
	grid.fit(X, y) # check the combination of parameters on train set

	print("Best F1 score while running Grid Search:", grid.best_score_) # best score obtained
	print("Best parameters found by Grid Search:\n", grid.best_params_) # parameters delivering the best score

# function to train the algorithms and test the prediction accuracy
def algo_evaluation(alg, X_train, X_test, y_train, y_test):
	alg = alg.fit(X_train, y_train)

	# calculate accuracy score 
	scores = cross_val_score(alg, X_test, y_test, cv = 10)
	print("10 fold Cross-validation Accuracy: ", sum(scores)/ len(scores))

    # predict the output and calculate the precision-recall score
	y_pred = alg.predict(X_test)
	print("Precision-Recall-Fscore ######## Each array has two elements \
		representing scores each for negative and positive classes:\n", 
		precision_recall_fscore_support(y_test, y_pred))

# Main function here
def main():
	# seeding for obtaining the same result if program is re-ran
	seed = 7
	np.random.seed(seed)

	# load the data set
	data = pd.read_csv('marketing-data.csv', delimiter=',')
	data = data.values # create numpy array of values of each column

	# encode all the columns with non-numeric values
	data[:,1] = encoding_columns(data[:,1])
	data[:,2] = encoding_columns(data[:,2])
	data[:,3] = encoding_columns(data[:,3])
	data[:,4] = encoding_columns(data[:,4])
	data[:,6] = encoding_columns(data[:,6])
	data[:,7] = encoding_columns(data[:,7])
	data[:,8] = encoding_columns(data[:,8])
	data[:,10] = encoding_columns(data[:,10])
	data[:,15] = encoding_columns(data[:,15])
	data[:,16] = encoding_columns(data[:,16])
	#print(data[0])

	# separate the features and labels
	X = data[:, 0:15].astype(float)
	y = data[:, 16]

	#print(X)
	#print(y)

	# Scale the features to 0-1 range
	scaler = MinMaxScaler()
	X = scaler.fit_transform(X)

	# Principal Component Analysis for find top 14 uncorrelated features
	pca = PCA(n_components=14)
	X = pca.fit_transform(X)

	# split the data set with train:test size = 75:25
	X_train, X_test, y_train, y_test = train_test_split(X,y)

	#print(format(Counter(y_train))) # shows that class-imbalance problem persists

	# SMOTE based oversampling of minority label, i.e. 'yes'
	sm = SMOTE(random_state=42)
	X_train, y_train = sm.fit_sample(X_train, y_train)
	#print(format(Counter(y_train))) # gives equal number of labels now

	# algorithms with parameters obtained from Grid Search tuning
	alg1 = ExtraTreesClassifier(max_depth=10, n_estimators=20, 
								max_features='auto',
								criterion='gini',
								min_samples_split=2,
								min_samples_leaf=1)
	alg2 = RandomForestClassifier(max_depth=10, n_estimators=20, 
								max_features='sqrt',
								criterion='gini',
								min_samples_split=2,
								min_samples_leaf=1)
	alg3 = GradientBoostingClassifier(loss='deviance', min_samples_split=2,
								min_samples_leaf=5,
								learning_rate=0.5,
								n_estimators=200)
	alg4 = SVC(C=1.2, probability=True)
	
	# Somehow, numpy could not figure out 'y_train' as a string
	# So, I changed the data type to "|S6", removing this throws error:
	# "Unknown Label type"
	y_train = np.asarray(y_train, dtype="|S6")
	y_test = np.asarray(y_test, dtype="|S6")

	# Call Grid Search for finding best hyper-parameters
	## tune_parameters(alg4, X_train, y_train)
	
	print("Ordering of scores: 1. Extra Trees  2. Random Forest  3. Gradient Boosting  4. SVM  5. Voting Classifier\n")
	# Find precision-recall score for each of the algorithms 
	algo_evaluation(alg1, X_train, X_test, y_train, y_test)
	algo_evaluation(alg2, X_train, X_test, y_train, y_test)
	algo_evaluation(alg3, X_train, X_test, y_train, y_test)
	algo_evaluation(alg4, X_train, X_test, y_train, y_test)
	
	
	######################## Voting Classifier #############################
	eclf1 = VotingClassifier(estimators=[('etc', alg1), ('rfc', alg2), ('gbc', alg3), ('svc', alg4)], voting = 'soft')
	algo_evaluation(eclf1, X_train, X_test, y_train, y_test)
	

# redirect the program to main() function on compilation
if __name__ == "__main__":
	main()