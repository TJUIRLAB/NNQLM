import ConfigParser
import os
import sys
from os.path import join as Join 



if __name__ == '__main__':
	
	cf = ConfigParser.ConfigParser()

	seccrous = 'crous'
	secpath = 'filepath'
	sectrain = 'trainfile'
	sectest = 'testfile'
	secgobal = 'gobal_var'

	sectrec = 'trec_arg'
	secwiki = 'wiki_arg'


	cf.add_section(seccrous)
	try:
		cf.set(seccrous, 'crous', sys.argv[1])
	except:
		cf.set(seccrous, 'crous', 'trec')#'trec'or'wiki'
	cf.set(seccrous, 'embeddingname', 'aquaint+wiki.txt.gz.ndim=50.bin')
	embeddingname = cf.get(seccrous, 'embeddingname')

	crous = cf.get(seccrous, 'crous')


	cf.add_section(secpath)
	cf.set(secpath, 'dataset', 'dataset')
	cf.set(secpath, 'interdata', 'interdata')
	cf.set(secpath, 'logdir', 'log')
	cf.set(secpath, 'embeddingdir', 'embeddings')


	dataset_dir = Join(cf.get(secpath, 'dataset'), crous)
	interdata_dir = cf.get(secpath, 'interdata')
	log_dir = Join(cf.get(secpath, 'logdir'), crous)
	embeddingdir = cf.get(secpath, 'embeddingdir')


	for sec in [sectrain, sectest]:
		cf.add_section(sec)
		name = sec[:-4]
		cf.set(sec, name+'file', Join(cf.get(secpath, 'dataset'), crous, name))
		path = Join(interdata_dir, name)
		if not os.path.exists(path):
			os.makedirs(path)
		cf.set(sec, 'qaid', Join(path, 'qaid'))
		cf.set(sec, 'label', Join(path, 'label'))
		cf.set(sec, 'question_index', Join(path, 'question_index'))
		cf.set(sec, 'answer_index', Join(path, 'answer_index'))
		cf.set(sec, 'qlens', Join(path, 'qlens'))
		cf.set(sec, 'alens', Join(path, 'alens'))

	cf.add_section(secgobal)
	cf.set(secgobal, 'qlen', '50')
	cf.set(secgobal, 'alen', '100')
	cf.set(secgobal, 'log', Join(log_dir, 'log.txt'))
	cf.set(secgobal, 'embeddings', Join(embeddingdir, embeddingname))
	cf.set(secgobal, 'vocab', Join(interdata_dir, 'vocab'))
	cf.set(secgobal, 'vocab_index', Join(interdata_dir, 'vocab_index'))


	cf.add_section(sectrec)
	cf.set(sectrec, 'learning_rate', '0.01')
	cf.set(sectrec, 'batch_size', '100')
	cf.set(sectrec, 'filter_row', '40')
	cf.set(sectrec, 'filter_num', '65')
	cf.set(sectrec, 'reg_rate', '0.001')
	cf.set(sectrec, 'itera_number', '9')

	cf.add_section(secwiki)
	cf.set(secwiki, 'learning_rate', '0.02')
	cf.set(secwiki, 'batch_size', '150')
	cf.set(secwiki, 'filter_row', '40')
	cf.set(secwiki, 'filter_num', '150')
	cf.set(secwiki, 'reg_rate', '0.01')
	cf.set(sectrec, 'itera_number', '25')




	secrunarg = crous
	cf.add_section(secrunarg)
	cf.set(secrunarg, 'learning_rate', cf.get(secrunarg + '_arg', 'learning_rate'))
	cf.set(secrunarg, 'batch_size', cf.get(secrunarg + '_arg', 'batch_size'))
	cf.set(secrunarg, 'filter_row', cf.get(secrunarg + '_arg', 'filter_row'))
	cf.set(secrunarg, 'filter_num', cf.get(secrunarg + '_arg', 'filter_num'))
	cf.set(secrunarg, 'reg_rate', cf.get(secrunarg + '_arg', 'reg_rate'))
	cf.set(secrunarg, 'itera_number', cf.get(secrunarg + '_arg', 'itera_number'))



	cf.write(open('config.conf', 'w'))
	print crous
