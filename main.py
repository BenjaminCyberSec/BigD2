#Import libraries
from sklearn.neural_network import MLPClassifier

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

    print("\nCreation DataSet:")
    # genere le dataset de Trainning Testing
    ExtractBodyFromDir("DataSet" + os.path.sep + "TR", "DataSet" + os.path.sep + "TRemailSet")
    BuldingDataSet("DataSet" + os.path.sep + "TRemailSet", "DataSet" + os.path.sep + "TRemailSet",
                   "DataSet" + os.path.sep + "TrainningSet.csv", emailTargets)

    # genere le dataset de Testing
    ExtractBodyFromDir("DataSet" + os.path.sep + "TT", "DataSet" + os.path.sep + "TTemailSet")
    BuldingDataSet("DataSet" + os.path.sep + "TTemailSet", "DataSet" + os.path.sep + "TTemailSet",
                   "DataSet" + os.path.sep + "TestingSet.csv", emailTargets)

    # Chargement des dataset ".csv"
    TrainDataSet = pd.read_csv("DataSet" + os.path.sep + "TrainningSet.csv", sep=';',names=('position', 'Subject', 'Content', 'SPAM'))
    TestDataSet = pd.read_csv("DataSet" + os.path.sep + "TestingSet.csv", sep=';',names=('position', 'Subject', 'Content'))

    # Vérifie et supprime les doublons
    #TrainDataSet.drop_duplicates(inplace=True)
    #TestDataSet.drop_duplicates(inplace=True)

    print("TrainDataSet =", TrainDataSet.shape)
    print("TestDataSet  =", TestDataSet.shape)

    # Afficher le graphe du nombre de mail Spam et Non Spam
    # sns.countplot(TrainDataSet.SPAM)

    # Statistique des SPAM et Non SPAM
    # print( TrainDataSet.groupby('SPAM').describe() )

    print("\nTraitement DataSet:")
    process_msg(TestDataSet)
    process_msg(TrainDataSet)

    # separe le 'target' et les 'features' du DataSet de Trainning
    y = pd.DataFrame(TrainDataSet.SPAM)
    x = TrainDataSet.drop(['SPAM'], axis=1)

    # ceration des variable d'entrainement et de validation
    x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, test_size=0.2, random_state=0)

    # ceration des variable de Testing
    x_test = TestDataSet

    print("Moyenne Nomre de mots du DataSet d'entrainement :", int(TrainDataSet['wordNum'].mean()))
    print("Moyenne Nomre de mots du DataSet d'entrainement :", int(TestDataSet['wordNum'].mean()))
    max_feature = max(int(TrainDataSet['wordNum'].mean()), int(TestDataSet['wordNum'].mean()))

    # Vectorisation de comptage
    # Il s'agit de compter le nombre d'occurrences de chaque mot dans le texte donné.

    # pour le les donne d'entrainement
    vectorize_Train = CountVectorizer(max_features=max_feature)
    temp_train = vectorize_Train.fit_transform(x_train['Text_clain']).toarray()
    temp_val = vectorize_Train.transform(x_val['Text_clain']).toarray()

    # pour le les donne de Teste
    vectorize_Test = CountVectorizer(max_features=max_feature)
    temp_test = vectorize_Test.fit_transform(x_test['Text_clain']).toarray()

    # tfidf : utiliser pour determiner à quel point un mot est important pour un texte dans un groupe de texte.
    # il est calculé en multipliant la fréquence d'un mot et la fréquence inverse du document
    # (la fréquence d'un mot, calculée par log (nombre de texte / nombre de texte contenant le mot)) du mot dans un groupe de texte.

    # pour le les donne d'entrainement
    tf_train = TfidfTransformer()

    temp_train = tf_train.fit_transform(temp_train)
    temp_val = tf_train.transform(temp_val)

    # pour le les donne de Test
    tf_test = TfidfTransformer()
    temp_test = tf_test.fit_transform(temp_test)

    # merging temp datafram avec le dataframe original

    # pour les donne d'entrainement
    temp_train = pd.DataFrame(temp_train.toarray(), index=x_train.index)
    temp_val = pd.DataFrame(temp_val.toarray(), index=x_val.index)
    x_train = pd.concat([x_train, temp_train], axis=1, sort=False)
    x_val = pd.concat([x_val, temp_val], axis=1, sort=False)

    # pour le les donne de Test
    temp_test = pd.DataFrame(temp_test.toarray(), index=x_test.index)
    x_test = pd.concat([x_test, temp_test], axis=1, sort=False)

    # supression de toutes les colonne des texte.

    x_train.drop(['position'], axis=1, inplace=True)
    x_train.drop(['Subject'], axis=1, inplace=True)
    x_train.drop(['Content'], axis=1, inplace=True)
    x_train.drop(['Text_clain'], axis=1, inplace=True)

    x_val.drop(['position'], axis=1, inplace=True)
    x_val.drop(['Subject'], axis=1, inplace=True)
    x_val.drop(['Content'], axis=1, inplace=True)
    x_val.drop(['Text_clain'], axis=1, inplace=True)

    x_test.drop(['position'], axis=1, inplace=True)
    x_test.drop(['Subject'], axis=1, inplace=True)
    x_test.drop(['Content'], axis=1, inplace=True)
    x_test.drop(['Text_clain'], axis=1, inplace=True)

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
    numm = progressbarTime("Trainning All Algoritheme")
    for name, model in models:
        if (j % (len(classifiers) / numm)) == 0:
            sys.stdout.write("-"*len(classifiers))
            sys.stdout.flush()

        model.fit(x_train, y_train)
        y_preds = model.predict(x_val)
        Y_preds[name] = y_preds
        score[name] = [accuracy_score(y_val, y_preds), 0, 0]
    #     print("Precision: {:.2f}%".format(100 * precision_score(y_val, y_preds)))
    #     print("Recall: {:.2f}%".format(100 * recall_score(y_val, y_preds)))
    #     print("Confusion Matrix:\n")
    #     confusion_m = confusion_matrix(y_val, y_preds)
    #     print(confusion_m)

    sys.stdout.write("]\n")

    save_results(names, score)
    printfile()

    # modelHigtperformence = DecisionTreeClassifier(random_state=0)
    # modelHigtperformence.fit(x_train, y_train)
    # y_test = modelHigtperformence.predict(x_test)
