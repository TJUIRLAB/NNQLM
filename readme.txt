
This code implements NNQLM2 in the paper:
End-to-End Quantum-like Language Models with Application to Question Answering. AAAI2018

# DEPENDENCIES

- python 2.7+
- numpy
- theano
- scikit-learn (sklearn)
- ConfigParser
- cPickle
- pandas
- os
- sys
- time

Python packages can be easily installed using the standard tool: pip install <package>

# RUN

You can run this model by:
>$ python run.py
This model is for trecqa and wikiqa. 
The default is for trecqa, and if you want to run this model on wikiqa, you can:
>$ python run.py wiki
You can use other qa dataset. Please put your dataset on dir dataset, and preprocess your data according to trecqa.



# REFERENCES

Aliaksei Severyn and Alessandro Moschitti. 
Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks. 
SIGIR, 2015.

Sordoni A, Nie J Y, Bengio Y. 
Modeling term dependencies with quantum language models for IR. 
SIGIR, 2013.

Kim Y. 
Convolutional Neural Networks for Sentence Classification. 
Eprint Arxiv, 2014.
