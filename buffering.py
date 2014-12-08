import os, glob, argparse
import numpy as np

from scipy.io import wavfile
from sklearn.metrics import confusion_matrix
from scikits.talkbox.features import mfcc

from sklearn.preprocessing import StandardScaler

from audio_load import *

parser = argparse.ArgumentParser()

parser.add_argument("-n", "--normalized", help="normalized the dataset",
                    action="store_true")

args = parser.parse_args()

genre_list = ["classical", "jazz", "country", "pop", "rock", "metal", "blues",
				"hiphop", "disco", "reggae"]

base_dir = os.getcwd()

Xs = []

for label, genre in enumerate(genre_list):
	for fn in glob.glob(os.path.join(base_dir, genre, "*.wav")):
		if args.normalized: 
			create_ceps(fn, 'norm')
		else:
			create_ceps(fn, 'plain')
	
