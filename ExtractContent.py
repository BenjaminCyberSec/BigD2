#!/usr/bin/python
# FileName: Subsampling.py 
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file 
# and store it in a new file with the same name in the dst dir.

import csv
import email.parser
import os, sys, stat
import re
import time
import sys

#Import libraries
import numpy as np
import pandas as pd
import nltk
import string
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import seaborn as sns

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import SnowballStemmer
nltk.download('stopwords')


from sklearn.model_selection import train_test_split
#importing libraries
import pandas as pd
import seaborn as sns
import string
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')



def ExtractSubPayload (filename):
	''' Extract the subject and payload from the .eml file.
	
	'''
	if not os.path.exists(filename): # dest path doesnot exist
		print ("ERROR: input file does not exist:", filename)
		os.exit(1)
	with open(filename, errors='ignore') as fp:
		msg = email.message_from_file(fp)
		payload = msg.get_payload()
		if type(payload) == type(list()) :
			payload = payload[0] # only use the first part of payload
		sub = msg.get('subject')
		sub = str(sub).replace(";", "")
		sub = str(sub).replace("\"", "")
		sub = str(sub).replace("'", "")
		if type(payload) != type('') :
			payload = str(payload).replace(";", "")
	
	return sub + str("; ") + payload

def ExtractBodyFromDir ( srcdir, dstdir ):
	'''Extract the body information from all .eml files in the srcdir and 
	
	save the file to the dstdir with the same name.'''
	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)  
	files = os.listdir(srcdir)
	for file in files:
		srcpath = os.path.join(srcdir, file)
		dstpath = os.path.join(dstdir, file)
		src_info = os.stat(srcpath)
		if stat.S_ISDIR(src_info.st_mode): # for subfolders, recurse
			ExtractBodyFromDir(srcpath, dstpath)
		else:  # copy the file
			body = ExtractSubPayload (srcpath)
			dstfile = open(dstpath, 'w')
			dstfile.write(body)
			dstfile.close()


def BuldingDataSet ( srcdir, dstdir, data, emailTargets ):

	i=0
	j = 0
	keyWord = ''

	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)

	if str(data).find("TrainningSet") != -1 :
		keyWord = 'TRAIN_'
		with open(data, 'w') as fichier:
			fichier.write("")
	else:
		keyWord = 'TEST_'
		with open(data, 'w') as fichier:
			fichier.write("")

	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)
	files = os.listdir(srcdir)

	numm=progressbarTime("Creation DataSet")
	for file in files:
		if ( j % (int(len(files)/numm)) )==0 :
			time.sleep(0.1)
			sys.stdout.write("-")
			sys.stdout.flush()

		NumEmail = 0
		if str(file).find("eml") != -1 & str(file).find(keyWord) != -1 :
			NumEmail1= str(file).replace(keyWord, "").replace(".eml", "")
			NumEmail = int(NumEmail1)

		srcpath = os.path.join(srcdir, file)
		dstpath = os.path.join(dstdir, file)
		src_info = os.stat(srcpath)
		if stat.S_ISDIR(src_info.st_mode): # for subfolders, recurse
			BuldingDataSet(srcpath, dstpath)
		else:  #
			temp = open(srcpath, 'r').read()
			fileobj = temp.splitlines()
			sentense = ''
			for row in fileobj:
				if (row):
					# try:
					# 	# sentense = sentense.encode('latin').strip() + row.encode('latin').strip()
					sentense = sentense + row
					# except ValueError:
					# 	print(str(file) +" That was no valid number "+ str(i))

			position = sentense.find(';')
			subject = sentense[0:position+1]
			body = sentense[position-1 : sentense.find('\n')].replace(";", "")

			if NumEmail !=0 :

				if str(data).find("TrainningSet") != -1:
					with open(data, 'a') as fichier:
						fichier.write(str(NumEmail) + ";"+ str(subject)  + str(body) + ";" + str(emailTargets[NumEmail])+"\n" )
				else:
					with open( data, 'a') as fichier:
						fichier.write(str(NumEmail)+ ";" + str(subject)  + str(body)+"\n" )
		j+=1

	sys.stdout.write("]\n")


def GetTrainingClassification (filename):
	if not os.path.exists(filename): # dest path doesnot exist
		print("ERROR: input file does not exist:", filename)
		exit()
	#pour que les id commence a 1 et pas 0
	emailTargets = [0]
	fp = open(filename, "r")
	fp.readline()
	#print(fp.readline())
	for line in fp:
		tmp = re.findall(r'\d+',line)
		emailTargets.insert(int(tmp[0]) ,int(tmp[1]))
	fp.close()
	return emailTargets

# cette fonction permet de nettoyer les messages et traiter les messages
# suprimer les ponctuations [! "# $% & '() * +, -. / :; <=>? @ [\] ^ _` {|} ~], et les 'stopwords'
# converti en munuscules,.

def process_msg (DataSet) :

	numm=progressbarTime("Traitement DataSet")
	sys.stdout.write("-")
	sys.stdout.flush()

	sm = SnowballStemmer("english")
	DataSet['Text_clain'] =''
	DataSet["wordNum"] = ''
	DataSet["messageLength"] = ''

	#Tokenisation et conversion en minuscules
	DataSet['Text_clain'] = DataSet['Content'].astype(str).map(lambda text: re.sub('[^a-zA-Z0-9]+', ' ',text)).apply(lambda x: (x.lower()).split())
	# compte le nombre de mot dans un message
	DataSet["wordNum"] = DataSet["Text_clain"].apply(len)
	DataSet['Text_clain']= DataSet['Text_clain'].apply(lambda text_list:' '.join(list(map(lambda word:sm.stem(word),(list(filter(lambda text:text not in set(stopwords.words('english')),text_list)))))))
	# compte le nombre de Caractere  du message
	DataSet["messageLength"] = DataSet["Text_clain"].apply(len)

	sys.stdout.write("]\n")


def save_results(names, resultat, filename='results_evaluator.txt'):
	li = 59
	sli = 59

	with open("DataSet" + os.path.sep +filename , 'w') as fichier:
		fichier.write('\n\n' * 3 + ' ' * 8 + 'Table: Performance comparison and cross validation: Training set   \n\n+' + "-" * li + "+\n")
		fichier.write("|ALGORITHME " + " " * 10 + "|" + " " * 5 + " evaluation  metrics " + " " * 10 + " | \n+" + "-" * li + "+")
		fichier.write("\n|" + " " * 21 + "|" + "%12s %12s %12s" % ('accuracy |', 'precision |', 'recall |') + "\n|" + "#" * li + "|")

		for name in names:
			fichier.write("\n| %20s" % (name))

			for value in resultat[name]:
				fichier.write("| {:.2f} %   ".format(100 * value))

			fichier.write("|\n"  + "+" + "-" * sli + "+ ")
			fichier.write("       %3s" % (' '))

def printfile(filename='results_evaluator.txt'):
	with open("DataSet" + os.path.sep + filename, "r") as filin:
		print(filin.read())

def progressbarTime(message, toolbar_width=50) :
	toolbar_width
	sys.stdout.write("%20s [%s]" % (message," " * 1))
	sys.stdout.flush()
	sys.stdout.write("\b" * (toolbar_width + 1))  # retoure a la ligne apres le '['
	return toolbar_width

