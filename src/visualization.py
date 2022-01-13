
import numpy as np
import glob

import os


import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def load(path):

	allRuns = []
	allNames = []

	with os.scandir(path) as entries:
		for entry in entries:
			if entry.is_dir():

				allNames.append(entry.name)
				allRuns.append([])

				for file in glob.glob(f"{entry.path}/*.csv"):
					tab = np.genfromtxt(file, delimiter=',')
					allRuns[-1].append(tab[1:])



	return allNames, np.array(allRuns)


def visualize(methodsNames, methodsData, which=2):

	nbPlays = methodsData[0][0].shape[0]
	t = np.arange(nbPlays)

	fig, ax = plt.subplots(1)

	for name, method in zip(methodsNames, methodsData):

		npMethod = np.array(method)[:, :, which]

		mu = npMethod.mean(axis=0)
		sigma = npMethod.std(axis=0)

		muW = savgol_filter(mu, 11, 2)
		#sigmaW = savgol_filter(sigma, 11, 2)

		ax.plot(t, muW, lw=2, label=name)
		
		ax.fill_between(t, mu+sigma, mu-sigma, alpha=0.4)


	ax.legend(loc='lower right')
	ax.set_xlabel('Plays')
	ax.set_ylabel('Food')
	ax.grid()

	plt.show()


###Start
runsPath = "runs_v3"

allNames, allRuns = load(runsPath)

seuil = 12.0

print("Number of Plays with mea")
for name, run in zip(allNames, allRuns):
	print(name, ":", (run[:, :, 2] > seuil).sum())
	# print(allNames[0], (allRuns[0, :, 200:300, 2].mean()))


visualize(allNames, allRuns, which=2)