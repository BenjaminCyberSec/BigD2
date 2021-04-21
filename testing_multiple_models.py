#Import libraries
import sys
import os
import nltk
import json
nltk.download('stopwords')
import pandas as pd
from sklearn.neural_network import MLPClassifier
from ExtractContent import GetTrainingClassification, ExtractBodyFromDir, BuildDataSet, Sanitize_Data, save_results, printfile, progressbarTime, remove_extra_fields
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

#jupiter imports
#import seaborn as sns 

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



    # Chargement des dataset ".csv"
    TrainDataSet = pd.read_csv(tr_csv_path, sep=';',names=('position', 'Subject', 'Content', 'SPAM')).sort_values('position')
    TestDataSet = pd.read_csv(tt_csv_path, sep=';',names=('position', 'Subject', 'Content')).sort_values('position')

    # Vérifie et supprime les doublons ----------------------- ON GARDE ?
    #TrainDataSet.drop_duplicates(inplace=True)


    #print("TrainDataSet =", TrainDataSet.shape)
    #print("TestDataSet  =", TestDataSet.shape)

    # Afficher le graphe du nombre de mail Spam et Non Spam
    # sns.countplot(TrainDataSet.SPAM)

    # Statistique des SPAM et Non SPAM
    # print( TrainDataSet.groupby('SPAM').describe() )

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
    
    

    # creation des variable d'entrainement et de validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

    # Vectorisation de comptage
    # Il s'agit de compter le nombre d'occurrences de chaque mot dans le texte donné.
    # --Training data
    vectorize_Train = CountVectorizer()
    temp_train = vectorize_Train.fit_transform(x_train['Text_clain']).toarray()
    temp_val = vectorize_Train.transform(x_val['Text_clain']).toarray()


    # tfidf : utiliser pour determiner à quel point un mot est important pour un texte dans un groupe de texte.
    # il est calculé en multipliant la fréquence d'un mot et la fréquence inverse du document
    # (la fréquence d'un mot, calculée par log (nombre de texte / nombre de texte contenant le mot)) du mot dans un groupe de texte.
    #-- pour le les donne d'entrainement
    tf_train = TfidfTransformer()

    temp_train = tf_train.fit_transform(temp_train)
    temp_val = tf_train.transform(temp_val)

    # merging temp datafram avec le dataframe original

    #-- pour les donne d'entrainement
    temp_train = pd.DataFrame(temp_train.toarray(), index=x_train.index)
    temp_val = pd.DataFrame(temp_val.toarray(), index=x_val.index)
    x_train = pd.concat([x_train, temp_train], axis=1, sort=False)
    x_val = pd.concat([x_val, temp_val], axis=1, sort=False)


    # supression de toutes les colonne des texte.
    remove_extra_fields(x_train)
    remove_extra_fields(x_val)

    names = ["K_Nearest_Neighbors", "Decision_Tree", "Random_Forest", "Logistic_Regression", "SGD_Classifier", "Naive_Bayes", "SVM_Linear","MLPClassifier"]
    Y_preds = {}

    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier(max_iter=100),
        MultinomialNB(),
        SVC(kernel='linear'),
        MLPClassifier()
    ]

    models = zip(names, classifiers)
    score = {}
    j=0

    print("\nTrainning All Algoritheme:")
    numm = progressbarTime("\nTrainning All Models")
    for name, model in models:
        if (j % (len(classifiers) / numm)) == 0:
            sys.stdout.write("-"*len(classifiers))
            sys.stdout.flush()

        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        Y_preds[name] = y_preds
        score[name] = [accuracy_score(y_val, y_preds), 1 - model.score(x_train, y_train), 1 - model.score(x_val, y_val)]
    #     print("Precision: {:.2f}%".format(100 * precision_score(y_val, y_preds)))
    #     print("Recall: {:.2f}%".format(100 * recall_score(y_val, y_preds)))
    #     print("Confusion Matrix:\n")
    #     confusion_m = confusion_matrix(y_val, y_preds)
    #     print(confusion_m)

    sys.stdout.write("]\n")

    save_results(names, score)
    printfile()

