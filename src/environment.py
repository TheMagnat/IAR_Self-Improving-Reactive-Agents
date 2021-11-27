
import numpy as np


class Environment:


	CHARACTER = {
		0: ' ', #Vide
		1: 'O', #Obstacle
		2: 'E',	#Enemie
		3: '$',	#Food
		4: 'I',	#Agent
	}


	"""
		mode:
			0: Random

	"""
	def __init__(self, width, height, foodNumber=13, mode=0):

		self.width = width
		self.height = height

		self.foodNumber = foodNumber


		self.mode = mode

		self.data = np.zeros((height, width), dtype=int)


	def reset(self):

		if self.mode == 0:
			
			params = np.array([0.8, 0.15, 0.05])
			params = params/params.sum()
			params = params.cumsum()


			draw = np.random.random_sample((self.height, self.width))

			last = 0.0
			for i, elem in enumerate(params):
				rez = np.argwhere((last < draw) & (draw <= elem))	

				self.data[rez[:, 0], rez[:, 1]] = i

				last = elem

			for i in range(self.foodNumber):

				rez = np.random.randint([self.height, self.width])
				self.data[rez[0], rez[1]] = 3

	def render(self):

		print(' '.ljust(self.width+1, '-'))
		for line in self.data:
			print('|', end='')
			for elem in line:
				print(Environment.CHARACTER[elem], end='')

			print('|')
		print(' '.ljust(self.width+1, '-'))



#########################################


if __name__ == '__main__':
	env = Environment(height=10, width=25, foodNumber=13)

	env.reset()
	env.render()

