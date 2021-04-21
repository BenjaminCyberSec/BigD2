#Import libraries
import os
import json
import nltk
nltk.download('stopwords')
import pandas as pd
from ExtractContent import GetTrainingClassification, ExtractBodyFromDir, BuildDataSet, Sanitize_Data, remove_extra_fields
from print_tt import print_output
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

def generate_feature_sets():
    data_dir_path = "DataSet" + os.path.sep 
    tr_email_path = data_dir_path + "spam-mail.tr.label"
    tr_extract_dir_path = data_dir_path + "TR"
    tt_extract_dir_path = data_dir_path + "TT"
    tr_content_dir_path = data_dir_path + "TRemailSet"
    tt_content_dir_path = data_dir_path + "TTemailSet"
    tr_csv_path = data_dir_path + "TrainningSet.csv"
    tt_csv_path = "DataSet" + os.path.sep + "TestingSet.csv"

    #Loads the targets
    emailTargets = GetTrainingClassification(tr_email_path)
    
    print("\Loading testing and training emails:")
    # generate training dataset
    ExtractBodyFromDir(tr_extract_dir_path, tr_content_dir_path)
    BuildDataSet(tr_content_dir_path, tr_content_dir_path,tr_csv_path, emailTargets)

    # generate testing dataset
    ExtractBodyFromDir(tt_extract_dir_path, tt_content_dir_path)
    BuildDataSet(tt_content_dir_path, tt_content_dir_path,tt_csv_path, emailTargets)

    # Chargement des dataset ".csv"
    #sorting is important to keep the id matching the reality
    TrainDataSet = pd.read_csv(tr_csv_path, sep=';',names=('position', 'Subject', 'Content', 'SPAM')).sort_values('position')
    TestDataSet = pd.read_csv(tt_csv_path, sep=';',names=('position', 'Subject', 'Content')).sort_values('position')

    # Vérifie et supprime les doublons ----------------------- ON GARDE ?
    #TrainDataSet.drop_duplicates(inplace=True)

    print("\nTraitement DataSet:")
    Sanitize_Data(TestDataSet) #sanitize and collect the word number and the char number
    Sanitize_Data(TrainDataSet)
    
    return TestDataSet, TrainDataSet

if __name__ == "__main__":
    test_data_set_f = 'test_data_set.json'
    train_data_set_f = 'train_data_set.json'
    
    if os.path.isfile(test_data_set_f) and os.path.isfile(train_data_set_f):
        with open(train_data_set_f, 'r') as train_file, open(test_data_set_f, 'r') as test_file:
            TrainDataSet = pd.read_json(train_file)
            TestDataSet = pd.read_json(test_file)
    else:
        TestDataSet, TrainDataSet = generate_feature_sets()
        with open(train_data_set_f, 'w') as train_file, open(test_data_set_f, 'w') as test_file:
            TestDataSet.to_json(test_data_set_f)
            TrainDataSet.to_json(train_data_set_f)

    # separe le 'target' et les 'features' du DataSet de Trainning
    y = pd.DataFrame(TrainDataSet.SPAM) 
    x = TrainDataSet.drop(['SPAM'], axis=1)
    
    print("Moyenne Nomre de mots du DataSet d'entrainement :", int(TrainDataSet['wordNum'].mean()))
    print("Moyenne Nomre de mots du DataSet d'entrainement :", int(TestDataSet['wordNum'].mean()))
    max_feature = max(int(TrainDataSet['wordNum'].mean()), int(TestDataSet['wordNum'].mean()))
    
    traintarget = y
    trainset = x
    testset = TestDataSet


    # Vectorisation de comptage
    # Il s'agit de compter le nombre d'occurrences de chaque mot dans le texte donné.
    # --Training data
    vectorize_Train = CountVectorizer(max_features=max_feature)
    temp_train = vectorize_Train.fit_transform(trainset['Text_clain']).toarray()
    #temp_target = vectorize_Train.transform(traintarget['Text_clain']).toarray()
    # --Testing data
    vectorize_Test = CountVectorizer(max_features=max_feature)
    temp_test = vectorize_Test.fit_transform(testset['Text_clain']).toarray()

    ## tfidf : utiliser pour determiner à quel point un mot est important pour un texte dans un groupe de texte.
    # il est calculé en multipliant la fréquence d'un mot et la fréquence inverse du document
    # (la fréquence d'un mot, calculée par log (nombre de texte / nombre de texte contenant le mot)) du mot dans un groupe de texte.
    #-- pour le les donne d'entrainement
    tf_train = TfidfTransformer()
    temp_train = tf_train.fit_transform(temp_train)
    #temp_target = tf_train.transform(temp_target)

    #-- pour le les donne de Test
    tf_test = TfidfTransformer()
    temp_test = tf_test.fit_transform(temp_test)

    ## merging temp datafram avec le dataframe original
    temp_train = pd.DataFrame(temp_train.toarray(), index=trainset.index)
    trainset = pd.concat([trainset, temp_train], axis=1, sort=False)
    #temp_target = pd.DataFrame(temp_target.toarray(), index=traintarget.index)
    #traintarget = pd.concat([traintarget, temp_target], axis=1, sort=False)

    #-- pour le les donne de Test
    temp_test = pd.DataFrame(temp_test.toarray(), index=testset.index)
    testset = pd.concat([testset, temp_test], axis=1, sort=False)

    # supression de toutes les colonne des texte.
    remove_extra_fields(testset)
    remove_extra_fields(trainset)

    
    model = RandomForestClassifier()
    model.fit(trainset, traintarget)
    pred = model.predict(testset)
    print_output(pred)

    


