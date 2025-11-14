write a .py file that runs on modal and saves the results to wandb in tabular format.

all of the .py file should run on modal

load the micro-mass dataset from openml (id=1301)

(ref. https://pypi.org/project/openml/)

next, split the data 50-50 based on target class, such that about 50% is background and 50% is foreground. use a random seed, and then sort by size of descending order

run cpca in python, please refer to https://github.com/abidlabs/contrastive based on the forground and background data
reduce dimensions to 2, 10, 20, 50

then run tabpfn classification (https://github.com/PriorLabs/TabPFN) on the foreground data using 5-fold validation

log the results in wandb in tabular format
