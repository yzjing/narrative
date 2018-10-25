import os

data = [line for line in open('../../data/MovieSummaries/plot_summaries.txt', 'r')]

for line in data:
	line = line.split('\t')
	length = len(line[1].split())
	if length >=300:
		with open (line[0] + '.txt', 'w') as f:
			f.write(line[1])
