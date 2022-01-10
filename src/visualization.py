
import numpy as np
import glob

import os


import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

runsPath = "runs"


def load(path):

	allRuns = []
	allNames = []

	with os.scandir(path) as entries:
		for entry in entries:
			if entry.is_dir():

				print(f"{entry.name}:")

				allNames.append(entry.name)
				allRuns.append([])

				for file in glob.glob(f"{entry.path}/{entry.name}*.csv"):
					tab = np.genfromtxt(file, delimiter=',')
					allRuns[-1].append(tab[1:])

	return allNames, np.array(allRuns)


def visualize(methodsNames, methodsData, which=2):

	# Fixing random state for reproducibility
	np.random.seed(19680801)

	nbPlays = methodsData[0][0].shape[0]
	t = np.arange(nbPlays)

	# an (Nsteps x Nwalkers) array of random walk steps
	#S1 = 0.004 + 0.02*np.random.randn(Nsteps, Nwalkers)
	#S2 = 0.002 + 0.01*np.random.randn(Nsteps, Nwalkers)

	# an (Nsteps x Nwalkers) array of random walker positions
	#X1 = S1.cumsum(axis=0)
	#X2 = S2.cumsum(axis=0)


	# Nsteps length arrays empirical means and standard deviations of both
	# populations over time

	fig, ax = plt.subplots(1)

	for name, method in zip(methodsNames, methodsData):

		npMethod = np.array(method)[:, :, which]

		mu = npMethod.mean(axis=0)
		sigma = npMethod.std(axis=0)

		muW = savgol_filter(mu, 11, 2)

		ax.plot(t, muW, lw=2, label=name)
		
		ax.fill_between(t, mu+sigma, mu-sigma, alpha=0.4)


	ax.legend(loc='lower right')
	ax.set_xlabel('Plays')
	ax.set_ylabel('Food')
	ax.grid()

	plt.show()


allNames, allRuns = load(runsPath)


print(type(allRuns[0]))

#print( np.array(allRuns).shape )

#rez = allRuns[:, :, :, 2]

#print(rez.shape)

visualize(allNames, allRuns, which=2)