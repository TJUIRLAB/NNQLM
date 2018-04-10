import numpy as np 
import cPickle as Pickle 
import os
import pandas as pd 
from time import time
from alphabet import Alphabet
from nltk.corpus import stopwords
import ConfigParser


from util import load_bin_vec


cf = ConfigParser.ConfigParser()
cf.read('config.conf')

_, _, trainpara, testpara, gobal, _, _, _ = cf.sections()

inter_files = [[], []]
inter_vars = [[], []]



embeddings_file = cf.get(gobal, 'embeddings')
vocab_file = cf.get(gobal, 'vocab')
vocab_index_file = cf.get(gobal, 'vocab_index')
q_len = cf.getint(gobal, 'qlen')
a_len = cf.getint(gobal, 'alen')

trainfile = cf.get(trainpara, 'trainfile')
train_qaid_file = cf.get(trainpara, 'qaid')
inter_files[0].append(train_qaid_file)
train_qindex_file = cf.get(trainpara, 'question_index')
inter_files[0].append(train_qindex_file)
train_aindex_file = cf.get(trainpara, 'answer_index')
inter_files[0].append(train_aindex_file)
train_label_file = cf.get(trainpara, 'label')
inter_files[0].append(train_label_file)
train_qlens_file = cf.get(trainpara, 'qlens')
inter_files[0].append(train_qlens_file)
train_alens_file = cf.get(trainpara, 'alens')
inter_files[0].append(train_alens_file)


testfile = cf.get(testpara, 'testfile')
test_qaid_file = cf.get(testpara, 'qaid')
inter_files[1].append(test_qaid_file)
test_qindex_file = cf.get(testpara, 'question_index')
inter_files[1].append(test_qindex_file)
test_aindex_file = cf.get(testpara, 'answer_index')
inter_files[1].append(test_aindex_file)
test_label_file = cf.get(testpara, 'label')
inter_files[1].append(test_label_file)
test_qlens_file = cf.get(testpara, 'qlens')
inter_files[1].append(test_qlens_file)
test_alens_file = cf.get(testpara, 'alens')
inter_files[1].append(test_alens_file)

word_count = [0]
random_word_count = [0]
UNKNOWN_WORD_IDX_0 = 0

rng = np.random.RandomState(23455)




def add_to_vocab(data, alphabet):
	for sentence in data:
		for token in sentence.split():
			alphabet.add(token)


def sentence_index(sen, alphabet, input_lens):
	sen = sen.split()
	sen_index = []
	for word in sen:
		sen_index.append(alphabet[word])
	sen_index = sen_index[:input_lens]
	while len(sen_index) < input_lens:
		sen_index += sen_index[:(input_lens - len(sen_index))]

	return np.array(sen_index), len(sen)


def sentence_indece(crous, alphabet):
	qids = crous['qid']
	aids = crous['aid']
	questions = crous['question']
	answers = crous['answer']
	labels = crous['flag']
	question_indece = []
	answer_indece = []
	qlen_list = []
	alen_list = []
	for question in questions:
		question_index, question_len = sentence_index(question, alphabet, q_len)
		question_indece.append(question_index)
		qlen_list.append(question_len)
	for answer in answers:
		answer_index, answer_len = sentence_index(answer, alphabet, a_len)
		answer_indece.append(answer_index)
		alen_list.append(answer_len)
	labels_list = list(labels)
	qids_list = list(qids)
	aids_list = list(aids)
	qaids_list = []
	for i, j in zip(qids_list, aids_list):
		qaids_list.append([i, j])
	return qaids_list, question_indece, answer_indece, labels_list, qlen_list, alen_list

if __name__ == '__main__':
	know_dict = load_bin_vec(embeddings_file)
	ndim = len(know_dict[know_dict.keys()[0]])
	df_train= pd.read_csv(trainfile, header=None,sep="\t",names=["qid",'aid',"question","answer","flag"],quoting =3)
	df_test= pd.read_csv(testfile, header=None,sep="\t",names=["qid",'aid',"question","answer","flag"],quoting =3)

	df_train['question'] = df_train['question'].str.lower()
	df_train['answer'] = df_train['answer'].str.lower()
	df_test['question'] = df_test['question'].str.lower()
	df_test['answer'] = df_test['answer'].str.lower()

	stopwords = []

	alphabet = Alphabet(start_feature_id=0)
	alphabet.add('UNKNOWN_WORD_IDX_0')

	vocab_dict = {}

	for crous in [df_train, df_test]:
		add_to_vocab(crous['question'], alphabet)
		add_to_vocab(crous['answer'], alphabet)

	print alphabet.fid
	temp_vec = 0
	vocab_array = np.zeros((alphabet.fid, ndim), dtype = 'float32')
	for index in alphabet.keys():
		vec = know_dict.get(index, None)
		if vec is None:
			vec = rng.uniform(-0.25, 0.25, ndim)
			vec = list(vec)
			vec = np.array(vec, dtype = 'float32')
			random_word_count[0] += 1
		if alphabet[index] == 0:
			vec = np.zeros(ndim)
		temp_vec += vec
		vocab_array[alphabet[index]] = vec
	temp_vec /= len(vocab_array)
	for index, _ in enumerate(vocab_array):
		vocab_array[index] -= temp_vec

	Pickle.dump(alphabet, open(vocab_index_file, 'wb'))
	Pickle.dump(vocab_array, open(vocab_file, 'wb'))
	print alphabet.fid

	print 'train data begin to pro'
	train_inter_vars = [train_qaid, train_qindex, train_aindex, train_label, train_qlen, train_alen] \
	= sentence_indece(df_train, alphabet)
	inter_vars[0] += train_inter_vars
	print 'train data has been proed'
	print 'test data begin to pro'
	test_inter_vars = [test_qaid, test_qindex, test_aindex, test_label, test_qlen, test_alen] \
	= sentence_indece(df_test, alphabet)
	inter_vars[1] += test_inter_vars
	print 'test data has been proed'


	for ifiles, ivars in zip(inter_files, inter_vars):
		for ifile, ivar in zip(ifiles, ivars):
			Pickle.dump(ivar, open(ifile, 'wb'))

	print 'data has been proed'








