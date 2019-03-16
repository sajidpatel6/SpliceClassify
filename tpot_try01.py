# import matplotlib.pyplot as plt
import numpy as np
# import pandas as pd
# import seaborn as sns
# import warnings

# from mpl_toolkits.mplot3d import Axes3D
# from scipy import stats
# from scipy.stats import norm
# from scipy.stats import linregress
# from sklearn.preprocessing import StandardScaler
# from numpy.polynomial import polynomial as npp

# warnings.filterwarnings('ignore')

# %matplotlib inline

import multiprocessing
import pandas as pd
from tpot import TPOTClassifier
from sklearn.externals.joblib import Memory

if __name__ == '__main__':
	# importing the dataset
	df = pd.read_csv("VMI_Data_BP1_V02_AK6_mod02.csv")
	_datasetX = df.drop(['SPLICE_OK'], axis=1)
	_datasetY = df['SPLICE_OK']

	_datasetY=_datasetY.values.reshape(-1, 1)
	train_size = int(len(df) * 0.75)
	test_size = len(df) - train_size
	
	# train & test data
	trainX = _datasetX[0:train_size]
	trainY = _datasetY[0:train_size]
	
	testX = _datasetX[train_size:len(df)]
	testY = _datasetY[train_size:len(df)]
	
	#trainY = trainY.ravel()
	# trainY = trainY.flatten()
	# trainY
	# type(trainY)
	# trainY.dtype	
	#trainY = np.array([trainY])
	
	#memory1 = Memory(cachedir='E://tmp_classifier', verbose=1)
	
	# create instance 
	# generations=100, population_size=100, offspring_size=None, mutation_rate=0.9, crossover_rate=0.1, scoring='accuracy', cv=5, subsample=1.0, n_jobs=1,
	# max_time_mins=None, max_eval_time_mins=5, random_state=None, config_dict=None, warm_start=False, memory=None, periodic_checkpoint_folder=None,
	# early_stop=None, verbosity=0, disable_update_check=False
	
	# , warm_start=True, memory=memory1, periodic_checkpoint_folder='E://classifier',
	tpot = TPOTClassifier(n_jobs=-1, verbosity=2)
	
	# fit instance
	tpot.fit(trainX, trainY)
	
	# evaluate performance on test data
	print(tpot.score(testX, testY.ravel()))
	
	# export the script used to create the best model
	tpot.export('tpot_classifier_pipeline.py')
	


