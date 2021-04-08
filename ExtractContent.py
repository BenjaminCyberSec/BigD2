#!/usr/bin/python
# FileName: Subsampling.py 
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file 
# and store it in a new file with the same name in the dst dir. 

import email.parser 
import os, sys, stat
import shutil
import re
from sklearn import datasets
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import metrics

def ExtractSubPayload (filename):
	''' Extract the subject and payload from the .eml file.
	
	'''
	if not os.path.exists(filename): # dest path doesnot exist
		print("ERROR: input file does not exist:", filename)
		exit()
	fp = open(filename)
	#{print(filename)
	msg = email.message_from_file(fp)
	payload = msg.get_payload()
	if type(payload) == type(list()) :
		payload = payload[0] # only use the first part of payload
	sub = msg.get('subject')
	sub = str(sub)
	if type(payload) != type('') :
		payload = str(payload)
	
	return sub + payload

def ExtractBodyFromDirTrain ( srcdir, dstdir, emailTargets ):
	'''Extract the body information from all .eml files in the srcdir and 
	
	save the file to the dstdir with the same name.'''
	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)  
	files = os.listdir(srcdir)
	for file in files:
		srcpath = os.path.join(srcdir, file)
		#dstpath = os.path.join(dstdir, file)

		#Getting the email ID from it's filename
		tmp = re.findall(r'\d+',file)

		if len(tmp) == 0 :
			continue
		id = int(tmp.pop())
		target = emailTargets[id]
		if target == 1:
			dstpath = os.path.join(dstdir,r"ham", file)
		else:
			dstpath = os.path.join(dstdir,r"spam", file)


		src_info = os.stat(srcpath)
		if stat.S_ISDIR(src_info.st_mode): # for subfolders, recurse
			ExtractBodyFromDirTrain(srcpath, dstpath, emailTargets)
		else:  # copy the file
			body = ExtractSubPayload (srcpath)
			dstfile = open(dstpath, 'w')
			dstfile.write(body)
			dstfile.close()

def ExtractBodyFromDirTest ( srcdir, dstdir ):
	'''Extract the body information from all .eml files in the srcdir and 
	
	save the file to the dstdir with the same name.'''
	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)  
	files = os.listdir(srcdir)
	for file in files:
		#print(file)
		srcpath = os.path.join(srcdir, file)
		dstpath = os.path.join(dstdir, file)
		src_info = os.stat(srcpath)
		if stat.S_ISDIR(src_info.st_mode): # for subfolders, recurse
			ExtractBodyFromDirTest(srcpath, dstpath)
		else:  # copy the file
			body = ExtractSubPayload (srcpath)
			dstfile = open(dstpath, 'w')
			dstfile.write(body)
			dstfile.close()

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
	#print(emailTargets[3])
	return emailTargets


###################################################################
# main function start here
if __name__ == "__main__":
	# srcdir is the directory where the .eml are stored
	'''
	print('Input source directory: ') #ask for source and dest dirs
	srcdir = input()
	if not os.path.exists(srcdir):
		print('The source directory %s does not exist, exit...' % (srcdir))
		sys.exit()
	# dstdir is the directory where the content .eml are stored
	print('Input destination directory: ') #ask for source and dest dirs
	dstdir = input()
	if not os.path.exists(dstdir):
		print('The destination directory is newly created.')
		os.makedirs(dstdir)
	'''

	###################################################################
	emailTargets = GetTrainingClassification(r"adcg-ss14-challenge-02-spam-mails-detection\spam-mail.tr.label")

	###################################################################
	ExtractBodyFromDirTrain ( r"adcg-ss14-challenge-02-spam-mails-detection\TR", r"adcg-ss14-challenge-02-spam-mails-detection\TRemailSet", emailTargets ) 

	###################################################################
	ExtractBodyFromDirTest ( r"adcg-ss14-challenge-02-spam-mails-detection\TT", r"adcg-ss14-challenge-02-spam-mails-detection\TTemailSet\test" ) 

	###################################################################
	# Now we have the exctracted content, let's train our model
	trainEmails = datasets.load_files(r"adcg-ss14-challenge-02-spam-mails-detection\TRemailSet")
	#print(list(trainEmails.target_names))

	
	testEmails = datasets.load_files(r"adcg-ss14-challenge-02-spam-mails-detection\TTemailSet")
	#print(testEmails.filenames.shape)

	vectorizer = TfidfVectorizer()

	#IL BUG ICI LE CONNARD TODO
	vectors = vectorizer.fit_transform(trainEmails.data)
	print(vectors.nnz / float(vectors.shape[0]))

	vectors_test = vectorizer.transform(testEmails.data)
	clf = MultinomialNB(alpha=.01)
	clf.fit(vectors, trainEmails.target)
	pred = clf.predict(vectors_test)
	print(metrics.f1_score(testEmails.target, pred, average='macro'))