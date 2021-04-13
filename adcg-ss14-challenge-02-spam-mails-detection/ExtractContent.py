#!/usr/bin/python
# FileName: Subsampling.py 
# Version 1.0 by Tao Ban, 2010.5.26
# This function extract all the contents, ie subject and first part from the .eml file 
# and store it in a new file with the same name in the dst dir.
import csv
import email.parser
import os, sys, stat
import shutil
import re
import pandas as pd
import seaborn as sns

from pip._vendor.distlib.compat import raw_input


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
	keyWord = ''

	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)

	if str(data).find("TestingSet") != -1 :
		keyWord = 'TEST_'
		with open('../'+ data, 'w') as fichier:
			fichier.write("Subject ; Content ; SPAM ")
	else:
		keyWord = 'TRAIN_'
		with open('../'+ data, 'w') as fichier:
			fichier.write("Subject ; Content ")

	if not os.path.exists(dstdir): # dest path doesnot exist
		os.makedirs(dstdir)
	files = os.listdir(srcdir)
	for file in files:
		NumEmail = 0
		if str(file).find("eml") != -1 & str(file).find(keyWord) != -1 :
			NumEmail1= str(file).replace(keyWord, "").replace(".eml", "")
			NumEmail = int(NumEmail1)

		print(NumEmail)

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
					try:
						# sentense = sentense.encode('latin').strip() + row.encode('latin').strip()
						sentense = sentense + row
					except ValueError:
						print(str(file) +" That was no valid number "+ str(i))

			position = sentense.find(';')
			subject = sentense[0:position+1]
			body = sentense[position-1 : sentense.find('\n')].replace(";", "")

			if str(data).find("TestingSet") != -1:
				with open('../'+data, 'a') as fichier:
					fichier.write("\n"+ str(subject)  + str(body) + ";" + str(emailTargets[NumEmail]))
			else:
				with open('../'+data, 'a') as fichier:
					fichier.write("\n"+ str(subject)  + str(body) )

		i +=1



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


# cette fonction nettoyer le texte et renvoyer les jetons. Le nettoyage du texte est effectué en supprimant d'abord
# les ponctuations et les mots vides.



###################################################################
# main function start here
# srcdir is the directory where the .eml are stored
# print ('Input source directory: ') #ask for source and dest dirs
# srcdir = raw_input()
# if not os.path.exists(srcdir):
# 	print ('The source directory %s does not exist, exit...' % (srcdir))
# 	sys.exit()
# # dstdir is the directory where the content .eml are stored
# print ('Input destination directory: ') #ask for source and dest dirs
# dstdir = raw_input()
# if not os.path.exists(dstdir):
# 	print ('The destination directory is newly created.')
# 	os.makedirs(dstdir)

###################################################################

emailTargets = GetTrainingClassification("spam-mail.tr.label")


# genere le dataset de Trainning Testing
ExtractBodyFromDir ( "TR", "TrDst" )
BuldingDataSet ( "TrDst", "DataSet", "TrainningSet.csv", emailTargets )


# genere le dataset de Testing
ExtractBodyFromDir ( "TT", 'TtDst' )
BuldingDataSet ( "TtDst", 'TtDst', "TestingSet.csv", emailTargets )

df = pd.read_csv('../TestingSet.csv' ,sep=';', names=('Subject', 'Content','SPAM'))



#Vérifier les doublons et les supprimer
df.drop_duplicates(inplace = True)

#Afficher le nombre de données manquantes (NAN, NaN, na) pour chaque colonne
df.isnull().sum()

sns.countplot(df.SPAM)