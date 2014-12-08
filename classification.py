import os, glob, argparse
import numpy as np

from scipy.io import wavfile

from sklearn.cross_validation import KFold, StratifiedKFold

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

from sklearn.preprocessing import Imputer

from scikits.talkbox.features import mfcc

from audio_load import *

"""
Com valores normalizados:
	Mean(scores)=0.54900	d-dev(scores)=0.03360

Com valores muito desconhecidos trocados pela media:
	Mean(scores)=0.47200	d-dev(scores)=0.04285


"""

def machine_learning():
	pass

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--normalized", help="normalized the dataset",
                    action="store_true")

parser.add_argument("-gl", "--genre_list", help="list of music genres",
                    choices=["classical", "jazz", "country", "pop", "rock", "metal", "blues", "hiphop", "disco", "reggae"], nargs='*',
					default=["classical","jazz","country","pop","rock","metal",
					"blues","hiphop","disco","reggae"])
				
parser.add_argument("-k", "---nfolds", help="number of folds used for KFold algorithm",
					type=int, default=5)
					
parser.add_argument("-c","--classifier", choices=['KNeighborsClassifier', 'SVC', 'GaussianNB',
					'DecisionTreeClassifier', 'KMeans'],
					default='KMeans', help="Represents the classifier that will be used (default: KMeans) .")

args = parser.parse_args()

scores = []

accuracies = []

genre_list = args.genre_list

base_dir = os.getcwd()

if args.normalized:
	X, y = read_ceps(genre_list, base_dir, 'norm')
else:
	X, y = read_ceps(genre_list, base_dir)
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	X = imp.fit_transform(X)
	
k = args.nfolds

for i in range(len(genre_list)):
	accuracies.append([])

cv = KFold(n=len(y), n_folds=k, shuffle=True, random_state=42)#, indices=True)

for train, test in cv:
	
	X_train, y_train = X[train], y[train]
	X_test, y_test = X[test], y[test]
	
	"""
	classifier = neighbors.KNeighborsClassifier(n_neighbors = 5)
	classifier.fit(X_train, y_train)
	scores.append(classifier.score(X_test, y_test))
	"""
	
	if args.classifier == 'KMeans':
		classifier = KMeans(n_clusters = len(genre_list), max_iter=10000)
	else:
		classifier = eval(args.classifier)()

	classifier.fit(X_train, y_train)
	
	y_ = classifier.predict(X_test)

	accu = 0
	for i in range(len(y_)):
		if y_[i] == y_test[i]:
			accu+=1
	acc=[]
	
	for i in range(len(genre_list)):
		total_correct = 0
		total_predict = 0
		for j in range(len(y_)):
			if y_test[j] == i:
				total_correct+=1
			if y_[j] == i:
				total_predict+=1
		accuracies[i].append(total_predict/total_correct*1.)
		
	
	scores.append(accu/float(len(y_)))

for i in range(len(genre_list)):
	print("Mean(%s)=%.5f\td-dev(scores)=%.5f" % (genre_list[i], np.mean(accuracies[i]), np.std(accuracies[i])))
print
print("Mean(%s)=%.5f\td-dev(scores)=%.5f" % (args.classifier, np.mean(scores), np.std(scores)))
