import numpy as np 
import pandas as pd 
import cPickle as Pickle
from PIL import Image
import os, sys
from sklearn.utils import shuffle
import theano
import theano.tensor as T


# def batch_pad(datasets, batchsize):
# 	batch_num = int(len(datasets[0]) / batchsize)
# 	extra_num = len(datasets[0]) - batch_num * batchsize
# 	if extra_num > 0:
# 		pad_num = batchsize - extra_num
# 		for index, dataset in enumerate(datasets):
# 			datasets[index] += dataset[:pad_num]
# 		return batch_num + 1
# 	return batch_num

def batch_pad(dataset, batchsize):
	batch_num = int(len(dataset) / batchsize)
	extra_num = len(dataset) - batch_num * batchsize
	if extra_num > 0:
		pad_num = batchsize - extra_num
		dataset += dataset[:pad_num]
		
def batch_num(dataset, batchsize):
	batchs = int(len(dataset) / batchsize)
	extra_num = len(dataset) - batchs * batchsize
	if extra_num > 0:
		return batchs + 1
	else:
		return batchs


def load_bin_vec(fname):
	
	print fname
	word_vecs = {}
	with open(fname, "rb") as f:
		header = f.readline()
		vocab_size, layer1_size = map(int, header.split())
		binary_len = np.dtype('float32').itemsize * layer1_size
		print 'vocab_size, layer1_size', vocab_size, layer1_size
		for i, line in enumerate(xrange(vocab_size)):
			if i % 100000 == 0:
				print '.',
			word = []
			while True:
				ch = f.read(1)
				if ch == ' ':
					word = ''.join(word)
					break
				if ch != '\n':
					word.append(ch)
			word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
		print "done"
		return word_vecs



def mrr_metric(group):
	# print group.iloc[0]['question']
	group = shuffle(group, random_state = 121)
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	rr=candidates[candidates["flag"]==1].index.min()+1
	if rr != rr:
		return 0.
	return 1.0/rr
	
def map_metric(group):
	# print group.iloc[0]['question']
	group = shuffle(group, random_state = 121)
	ap=0
	candidates=group.sort_values(by='score',ascending=False).reset_index()
	correct_candidates=candidates[candidates["flag"]==1]
	correct_candidates_index = candidates[candidates["flag"]==1].index
	if len(correct_candidates)==0:
		return 0
	for i,index in enumerate(correct_candidates_index):
		# ap+=1.0* (i+1) /(index+1)
		ap += 1.0 * (i + 1)/(index + 1)
	return ap/len(correct_candidates)

def evaluation_plus(modelfile, groundtruth):
	answers=pd.read_csv(groundtruth,header=None,sep="\t",names=["question","answer","flag"],quoting =3)
	answers["score"]=pd.read_csv(modelfile,header=None,sep="\t",names=["score"],quoting =3)
	print answers.groupby("question").apply(mrr_metric).mean()
	print answers.groupby("question").apply(map_metric).mean()


def share_x(data_file, batchsize):
	data_x = Pickle.load(open(data_file, 'rb'))
	batch_pad(data_x, batchsize)
	return theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow = True)

def share_y(data_file, batchsize):
	data_x = Pickle.load(open(data_file, 'rb'))
	batch_pad(data_x, batchsize)
	return T.cast(theano.shared(np.asarray(data_x), borrow = True), 'int32')