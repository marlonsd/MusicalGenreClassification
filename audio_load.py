import os, glob
import numpy as np

from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

from sklearn.preprocessing import StandardScaler


GENRE_DIR = os.getcwd()

def write_ceps(ceps, fn, op='plain'):
	base_fn, ext = os.path.splitext(fn)
	if op == 'norm':
		data_fn = base_fn + "_norm" + ".ceps"
	else:
		data_fn = base_fn + "_plain" + ".ceps"
	np.save(data_fn, ceps)
	print("Written %s" % data_fn)
	
def create_ceps(fn, op='plain'):
	sample_rate, X = wavfile.read(fn)
	
	if op == 'norm':
		norm = StandardScaler()
		X = norm.fit_transform(X)
	
	ceps, mspec, spec = mfcc(X)
	
	write_ceps(ceps, fn, op)
	
def read_ceps(genre_list, base_dir=GENRE_DIR, op='plain'):
	X, y = [], []
	
	for label, genre in enumerate(genre_list):
		for fn in glob.glob(os.path.join(base_dir, genre, "*"+op+".ceps.npy")):
			ceps = np.load(fn)
			
			#print ceps
			"""
			for i in range(len(ceps)):
				n = np.isnan(ceps[i])
				inf = np.isinf(ceps[i])
				for j in range(len(ceps[i])):
					if n[j] or inf[j]:
						ceps[i][j] = 0.
			"""
			num_ceps = len(ceps)
			# X.append(np.mean(ceps))
			#X.append(np.mean(ceps[int(num_ceps*1/10.):int(num_ceps*9/10.)], axis=0))
			X.append(np.mean(ceps, axis=0))
			
			y.append(label)
	return np.array(X), np.array(y)

# genre_list = ["classical", "jazz", "country", "pop", "rock", "metal"]
