#coding=utf8
from __future__ import division
import numpy as np
import pandas as pd
import cPickle as Pickle
import os, time, sys
import ConfigParser
from sklearn.utils import shuffle
from sklearn import linear_model, metrics


import theano
import theano.tensor as T 
import theano.tensor.shared_randomstreams
from theano.tensor.signal import downsample


from util import batch_num, mrr_metric, map_metric, share_x, share_y
from nnlayer import HiddenLayer, LogisticRegression, CNNLayer, Density, Density_Dot

cf = ConfigParser.ConfigParser()
cf.read('config.conf')

_, _, trainpara, testpara, gobal, _, _, args = cf.sections()




if __name__ == '__main__':
	trainfile = cf.get(trainpara, 'trainfile')
	testfile = cf.get(testpara, 'testfile')

	df_train= pd.read_csv(trainfile, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)
	df_test= pd.read_csv(testfile, header=None,sep="\t",names=['qid','aid',"question","answer","flag"],quoting =3)

	learning_rate = cf.getfloat(args, 'learning_rate')
	batch_size = cf.getint(args, 'batch_size')
	filter_row = cf.getint(args, 'filter_row')
	filter_num = cf.getint(args, 'filter_num')
	reg_rate = cf.getfloat(args, 'reg_rate')
	itera_number = cf.getint(args, 'itera_number')

	print 'learning_rate is: ', learning_rate 
	print 'batch_size is: ', batch_size
	print 'step is: ', filter_row
	print 'feature map is: ', filter_num
	print 'reg_rate is', reg_rate

	logfile = cf.get(gobal, 'log')
	qlen = cf.getint(gobal, 'qlen')
	alen = cf.getint(gobal, 'alen')
	vocab = cf.get(gobal, 'vocab')

	log = open(logfile, 'a')

	pen_rate = 0


	trainlabel = share_y(cf.get(trainpara, 'label'), batch_size)
	trainquestion = share_x(cf.get(trainpara, 'question_index'), batch_size)
	trainanswer = share_x(cf.get(trainpara, 'answer_index'), batch_size)
	trainqlens = share_x(cf.get(trainpara, 'qlens'), batch_size)
	trainalens = share_x(cf.get(trainpara, 'alens'), batch_size)

	testlabel = share_y(cf.get(testpara, 'label'), batch_size)
	testquestion = share_x(cf.get(testpara, 'question_index'), batch_size)
	testanswer = share_x(cf.get(testpara, 'answer_index'), batch_size)
	testqlens = share_x(cf.get(testpara, 'qlens'), batch_size)
	testalens = share_x(cf.get(testpara, 'alens'), batch_size)

	print 'data sets have been shared'

	embeddings = Pickle.load(open(vocab, 'rb'))
	embedding_ndim = len(embeddings[0])

	filter_col = filter_row
	train_sample_num = len(df_train)
	test_sample_num = len(df_test)
	train_batch_num = batch_num(df_train, batch_size)
	test_batch_num = batch_num(df_test, batch_size)


	channel_num = 1
	class_num = 2

	image_shape = (batch_size, channel_num, embedding_ndim, embedding_ndim)
	filter_shape = (filter_num, channel_num, filter_row, filter_col)

	rng = np.random.RandomState(23455)


	index = T.lscalar('index')
	x_q = T.matrix('x_q')
	x_a = T.matrix('x_a')   
	y = T.ivector('y')
	len_qs = T.vector('len_qs')
	len_as = T.vector('len_as')


	Embeddings = theano.shared(value = embeddings, name = 'Embeddings', borrow=True)

	layer0_q = Embeddings[T.cast(x_q.flatten(),dtype="int32")].\
	reshape((batch_size, qlen, embedding_ndim))
	layer0_a = Embeddings[T.cast(x_a.flatten(),dtype="int32")].\
	reshape((batch_size, alen, embedding_ndim))

	layer1_q = Density(rng, layer0_q, len_qs, qlen)
	layer1_a = Density(rng, layer0_a, len_as, alen)

	layer2 = Density_Dot(layer1_q.output, layer1_a.output)

	layer3 = CNNLayer(rng, input = layer2.output.reshape(image_shape), \
		filter_shape = filter_shape, image_shape = image_shape)

	layer4_q = layer3.output.max(axis = 3).flatten(2)
	layer4_a = layer3.output.max(axis = 2).flatten(2)

	Logistic_in_num = 2 * int(embedding_ndim - filter_row + 1) * filter_num
	Logistic_input = T.concatenate([layer4_q, layer4_a], axis = 1)

	layer5 = LogisticRegression(Logistic_input, Logistic_in_num, class_num)

	params = layer1_q.params + layer1_a.params + layer3.params + layer5.params

	reg_term = (layer1_q.params[0] ** 2).sum() + (layer1_a.params[0] ** 2).sum() \
	+ (layer3.params[0] ** 2).sum() + (layer5.params[0] ** 2).sum()
	# reg_term is the penalty for the complexity of the model

	pen_term = ((layer1_q.params[0] ** 2) ** 0.5).sum() + ((layer1_a.params[0] ** 2)**0.5).sum() - 2
	# pen_term is the penalty that the trace of the density matrix is not equal 1 

	cost = layer5.negative_log_likelihood(y) + reg_rate * reg_term + pen_rate * pen_term

	grads = T.grad(cost, params)
	updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]

	given_train = {x_q: trainquestion[index * batch_size : (index + 1) * batch_size], \
	x_a: trainanswer[index * batch_size : (index + 1) * batch_size], \
	len_qs: trainqlens[index * batch_size : (index + 1) * batch_size], \
	len_as: trainalens[index * batch_size : (index + 1) * batch_size], \
	y: trainlabel[index * batch_size : (index + 1) * batch_size]}

	given_test = {x_q: testquestion[index * batch_size : (index + 1) * batch_size], \
	x_a: testanswer[index * batch_size : (index + 1) * batch_size], \
	len_qs: testqlens[index * batch_size : (index + 1) * batch_size], \
	len_as: testalens[index * batch_size : (index + 1) * batch_size], \
	y: testlabel[index * batch_size : (index + 1) * batch_size]}

	train_model = theano.function([index], cost, updates = updates, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	train_score_model = theano.function([index], layer5.y_score, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	test_model = theano.function([index], layer5.y_score, givens = given_test, on_unused_input='ignore', allow_input_downcast=True)

	train_model_loss = theano.function([index], cost, givens = given_train, on_unused_input='ignore', allow_input_downcast=True)
	test_model_loss = theano.function([index], cost, givens = given_test, on_unused_input='ignore', allow_input_downcast=True)

	

	print "begin to train"
	
	test_mrr = 0.
	test_map = 0.

	for itera_i in range(itera_number):
		print 'The ', itera_i, 'epcho begin'
		count_i = 0
		train_score = []
		for i in shuffle(range(train_batch_num), random_state = 121):
		# for i in range(train_batch_num):
			train_model(i)
			if count_i % 10 == 0:
				print count_i, 'batch train data have been trained'
			count_i += 1


		count_i = 0
		for i in range(train_batch_num):
			score = train_score_model(i)
			for j in score:
				train_score.append(j[1])
				
			if count_i % 10 == 0:
				print count_i, 'batch train data have been tested, loss is ', train_model_loss(i)
			count_i += 1
		train_score = train_score[:train_sample_num]
		df_train['score'] = train_score
		train_flag = df_train['flag']
		roc = metrics.roc_auc_score(train_flag, train_score)
		print 'The roc of train is: ', roc*100


		test_score = []
		count_i = 0
		for i in range(test_batch_num):
			score = test_model(i)
			for j in score:
				test_score.append(j[1])
			if count_i % 10 == 0:
				print count_i, 'batch test data have been tested, loss is ', test_model_loss(i)
			count_i += 1
		test_score = test_score[:test_sample_num]
		df_test['score'] = test_score
		test_flag = df_test['flag']
		roc = metrics.roc_auc_score(test_flag, test_score)
		print 'The roc of test is: ', roc*100
		test_mrr = df_test.groupby('question').apply(mrr_metric).mean()
		test_map = df_test.groupby('question').apply(map_metric).mean()
		print test_mrr
		print test_map
		
	log.write('learning_rate ' + str(learning_rate) + ', ')
	log.write('batch_size ' + str(batch_size) + ', ')
	log.write('filter_num ' + str(filter_num) + ', ')
	log.write('filter_row ' + str(filter_row) + ', ')
	log.write('reg_rate ' + str(reg_rate) + ' :\n')
	print 'end train'
	print 'the mrr is : ', test_mrr
	print 'the map is : ', test_map

	log.write('the mrr is : ')
	log.write(str(test_mrr) + '\t')
	log.write('the map is : ')
	log.write(str(test_map) + '\t')
	
	log.write('\n\n')
	log.close()

