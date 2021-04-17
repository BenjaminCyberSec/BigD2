#Import libraries
from ExtractContent import *
import os
import nltk
nltk.download('stopwords')
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix


if __name__ == "__main__":

    #Sauvegarde des target
    emailTargets = GetTrainingClassification("DataSet" + os.path.sep + "spam-mail.tr.label")

    # genere le dataset de Trainning Testing
    ExtractBodyFromDir("DataSet" + os.path.sep + "TR", "DataSet" + os.path.sep + "TRemailSet")
    BuldingDataSet("DataSet" + os.path.sep + "TRemailSet", "DataSet" + os.path.sep + "TRemailSet",
                   "DataSet" + os.path.sep + "TrainningSet.csv", emailTargets)

    # genere le dataset de Testing
    ExtractBodyFromDir("DataSet" + os.path.sep + "TT", "DataSet" + os.path.sep + "TTemailSet")
    BuldingDataSet("DataSet" + os.path.sep + "TTemailSet", "DataSet" + os.path.sep + "TTemailSet",
                   "DataSet" + os.path.sep + "TestingSet.csv", emailTargets)

    # Chargement des dataset ".csv"
    TrainDataSet = pd.read_csv("DataSet"+os.path.sep+"TrainningSet.csv", sep=';', names=('Subject', 'Content', 'SPAM'))
    TestDataSet = pd.read_csv("DataSet"+os.path.sep+"TestingSet.csv", sep=';', names=('Subject', 'Content'))

    # Vérifie et supprime les doublons
    TrainDataSet.drop_duplicates(inplace=True)
    TestDataSet.drop_duplicates(inplace=True)

    print("TrainDataSet =", TrainDataSet.shape)
    print("TestDataSet  =", TestDataSet.shape)

    print("\n Afficher le nombre de données manquantes (NAN, NaN, na) pour chaque colonne du TrainDataSet")
    print( TrainDataSet.isnull().sum() )
    print("\n Afficher le nombre de données manquantes (NAN, NaN, na) pour chaque colonne du TestDataSet")
    print(TestDataSet.isnull().sum() )

    # Afficher le graphe du nombre de mail Spam et Non Spam
    # sns.countplot(TrainDataSet.SPAM)

    # Statistique des SPAM et Non SPAM
    # print( TrainDataSet.groupby('SPAM').describe() )

    process_msg(TestDataSet)
    process_msg(TrainDataSet)

    # separe le 'target' et les 'features'
    y = pd.DataFrame(TrainDataSet.SPAM)
    x = TrainDataSet.drop(['SPAM'], axis=1)

    # ceration des variable d'entrainement et de test
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

    # Vectorisation de comptage
    # Il s'agit de compter le nombre d'occurrences de chaque mot dans le texte donné.
    max_feature = max(int(TrainDataSet['wordNum'].mean()), int(TestDataSet['wordNum'].mean()))

    vectorize = CountVectorizer(max_features=max_feature)
    temp_train = vectorize.fit_transform(x_train['Text_clain']).toarray()
    temp_val = vectorize.transform(x_val['Text_clain']).toarray()

    # tfidf : utiliser pour determiner à quel point un mot est important pour un texte dans un groupe de texte.
    # il est calculé en multipliant la fréquence d'un mot et la fréquence inverse du document
    # (la fréquence d'un mot, calculée par log (nombre de texte / nombre de texte contenant le mot)) du mot dans un groupe de texte.
    tf = TfidfTransformer()
    temp_train = tf.fit_transform(temp_train)
    temp_val = tf.transform(temp_val)

    # merging temp datafram with original dataframe
    temp_train = pd.DataFrame(temp_train.toarray(), index=x_train.index)
    temp_val = pd.DataFrame(temp_val.toarray(), index=x_val.index)
    x_train = pd.concat([x_train, temp_train], axis=1, sort=False)
    x_val = pd.concat([x_val, temp_val], axis=1, sort=False)

    # supression de toutes les colonne des texte.

    x_train.drop(['Subject'], axis=1, inplace=True)
    x_train.drop(['Content'], axis=1, inplace=True)
    x_train.drop(['Text_clain'], axis=1, inplace=True)

    x_val.drop(['Subject'], axis=1, inplace=True)
    x_val.drop(['Content'], axis=1, inplace=True)
    x_val.drop(['Text_clain'], axis=1, inplace=True)

    names = ["K_Nearest_Neighbors", "Decision_Tree", "Random_Forest", "Logistic_Regression", "SGD_Classifier", "Naive_Bayes", "SVM_Linear"]
    Y_preds = {}

    classifiers = [
        KNeighborsClassifier(),
        DecisionTreeClassifier(random_state=0),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier(max_iter=100),
        MultinomialNB(),
        SVC(kernel='linear')
    ]

    models = zip(names, classifiers)
    score = {}

    for name, model in models:
        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        Y_preds[name] = y_preds
        score[name] = [accuracy_score(y_val, y_preds), 1, 4]
    #     print("Precision: {:.2f}%".format(100 * precision_score(y_val, y_preds)))
    #     print("Recall: {:.2f}%".format(100 * recall_score(y_val, y_preds)))
    #     print("Confusion Matrix:\n")
    #     confusion_m = confusion_matrix(y_val, y_preds)
    #     print(confusion_m)

    save_results(names, score)
    printfile()
