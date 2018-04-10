import sys
import os
import subprocess


crous = 'trec' #'trec' or 'wiki'
subprocess.call('python config.py %s'%(crous), shell = True)
subprocess.call('python parse.py', shell = True)
subprocess.call('python nnqlm.py', shell = True)

