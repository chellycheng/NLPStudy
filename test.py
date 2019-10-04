# import some library
# -*- coding: utf-8 -*
from nltk import pos_tag
import numpy as np
import scipy as scp
import random
import string
#model
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
#help_clean
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
#help_feature_process
from sklearn.feature_extraction.text import CountVectorizer
#help_analysis
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#analysis
from sklearn.utils.multiclass import unique_labels
from sklearn.model_selection import cross_val_score
import sklearn.naive_bayes
from sklearn import metrics
import matplotlib.pyplot as plt
#0 Initial
class Initial:
    def initiation():
        reviews_pos = Reader('rt-polarity.pos')
        reviews_neg = Reader('rt-polarity.neg')

        test_sentsp = reviews_pos.shuffle()
        test_sentsn = reviews_neg.shuffle()

        reviews_pos.write("pos_shuffle.pos")
        reviews_neg.write("neg_shuffle.neg")

# 1 Reading of the data
class Reader:
    def __init__(self, path):
        # open the file
        self.file = open(path, errors='ignore', encoding='utf-8')
        self.data = []
        # shuffle the documents at first

    def shuffle(self):
        data_p = [(random.random(), line) for line in self.file]
        data_p.sort()
        sample_sets = [str(w) for (a, w) in data_p]
        self.data = sample_sets
        return sample_sets


    def write(self,name):
        f = open(name, "w")
        for line in self.data:
            f.write(line)
        f.close()

    def reading(self):
        data_p = [ (line) for line in self.file]
        return data_p

    def close_file(self):
        self.file.close()

# 2 Cleaning of the data
class Cleaner:
    def __init__(self, sample_list,use_lemmer,use_stemmer, use_stopwords,add_tags):
        self.sents_list = sample_list
        self.words_list = [self.splitter(w) for w in sample_list]
        self.s =use_stemmer
        self.l =use_lemmer
        self.st = use_stopwords
        self.a =add_tags

    def splitter(self,sample_list):
        pos_words = sample_list.split()
        return pos_words

    def remove_punc(self):
        removed_punc = []
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            removed_punc.append( [w.translate(table) for w in s] )
        self.words_list = removed_punc

    def lowercase(self):
        lowered = []
        for s in self.words_list:
            lowered.append( [w.lower() for w in s])
        self.words_list = lowered

    def remove_noncharacter(self):
        remove_nonchar = []
        for s in self.words_list:
            remove_nonchar.append([w for w in s if w.isalnum()])
        self.words_list = remove_nonchar

    def remove_stopWord(self):
        removed_stop = []
        stop_words = stopwords.words('english')
        for s in self.words_list:
            removed_stop.append([w for w in s if not w in stop_words])
        self.words_list = removed_stop

    def lemmatizer(self):
        lemmatized = []
        lemmatizer = WordNetLemmatizer()
        for s in self.words_list:
            lemmatized.append([lemmatizer.lemmatize(w) for w in s])
        self.words_list = lemmatized

    def stemmer(self):
        stemmed = []
        porter = PorterStemmer()
        for s in self.words_list:
            stemmed.append( [porter.stem(word) for word in s])
        self.words_list = stemmed

    def clean_low_puc_nc_le_stop(self):
        cleaned = []
        #porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        stop_words = stopwords.words('english')
        table = str.maketrans('', '', string.punctuation)
        for s in self.words_list:
            cleaned.append([lemmatizer.lemmatize(word.translate(table).lower()) for word in s if word not in stop_words])
        self.words_list = cleaned

    def cleaned(self):
        self.lowercase()
        self.remove_punc()
        self.remove_noncharacter()
        if self.l:
            self.lemmatizer()
        if self.s:
            self.stemmer()
        if self.st:
            self.remove_stopWord()
        result = []
        if self.a:
            sents = []
            close_adjectives_list = []
            adj_noun_tags = ["JJ", "JJR", "JJS","NN","NNP","NNPS","VB","RB","RBR","RBS","VBG","VBN","VBP","VBZ"]
            for w in self.words_list:
                current =(pos_tag(w))
                sents.append(current)
            for s in sents:
                close_adjectives_list.append([a[0] for a in s if a[1] in adj_noun_tags ])

            #print("With tag\n",sents)
            #print("Flatten set\n",flattened_sents)
            print(close_adjectives_list)
            for s in close_adjectives_list:
                result.append(' '.join(s))
            print("Result",result)
        else:
            for s in self.words_list:
                result.append(' '.join(s))
        return result

    def joined(self):
        sents = []
        for s in self.words_list:
            sents.append(' '.join(s))
        return sents

# 3 Feature Extraction
class Feature_Processer:

    def count_vector_features_produce(self,data1,data2,threshold):
        cv = CountVectorizer(binary=True,min_df=threshold)
        cv.fit(data1)
        X = cv.transform(data1)
        X_test = cv.transform(data2)
        X_names = cv.get_feature_names()
        return X, X_test, X_names

    #More option？

#4 Combine pos and neg feature. Tag them, and seperate the train sets and test sets.

class Neg_Pos_Combinator:
    def __init__(self,fsp,fsn,r):
        sp = int(len(fsp) * r)
        sn = int(len(fsn) * r)
        self.train_sp = fsp[sp:]
        self.test_sp = fsp[:sp]
        self.train_sn = fsn[sn:]
        self.test_sn = fsn[:sn]

    def combine_train_return_x_y(self):
        y = [1] *len(self.train_sp) + [0]* len(self.train_sn)
        #should be append operation
        x = self.train_sp + self.train_sn
        return x, y

    def combine_test_return_x_y(self):
        y = [1] * len(self.test_sp) + [0] * len(self.test_sn)
        # should be append operation
        x = self.test_sp + self.test_sn
        return x, y

    #5 Able to create different classifier in this class

#5 Classifier
class classifier:
    def __init__(self,x_train, y_train, x_test, y_test,):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
    def logistic(self,c):
        model = LogisticRegression(C=c, dual=True,solver='liblinear')
        model.fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        #accuracy = (preds == self.y_test).mean()
        #print("Losistic Regression : accurancy_mean is", accuracy)
        scores1 = cross_val_score(model, self.x_train, self.y_train, cv=5,scoring='accuracy')
        print("Score of Logistic in Cross Validation", scores1.mean()*100)
        print("Losistic Regression : accurancy_matrix is", metrics.accuracy_score( self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n",cm)
        print("Report", classification_report(self.y_test,preds))

    def NaiveB(self,alpha):
        model = BernoulliNB(alpha=alpha).fit(self.x_train, self.y_train)
        preds = model.predict(self.x_test)
        scores2 = cross_val_score(model, self.x_train, self.y_train, cv=5,scoring='accuracy')
        print("Score of Naive Bayes", scores2.mean()*100)
        print("Naive Bayes : accurancy_matrix is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))
    def svm(self,c):
        model = LinearSVC(C=c)
        model.fit(self.x_train,self.y_train)
        preds = model.predict(self.x_test)

        scores3 = cross_val_score(model, self.x_train, self.y_train, cv=5,scoring='accuracy')
        print("Score of SVM in Cross Validation", scores3.mean()*100)
        print("SVM Regression : accurancy_is", metrics.accuracy_score(self.y_test, preds))
        cm = confusion_matrix(self.y_test, preds)
        print("Confusion Matrix\n", cm)
        print("Report", classification_report(self.y_test, preds))

    def dummy(self):
        clf = DummyClassifier(strategy='stratified', random_state=0)
        clf.fit(self.x_train, self.y_train)
        score = clf.score(self.x_test, self.y_test)
        print("Random Baseline's accurancy", score)
    # 6 Useful analysis method

#class Analysis:
class plot_matrix:
    def plot_confusion_matrix(y_true, y_pred, classes,normalize=False,cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred)
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print("\n",cm)
        """"
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(y_true, y_pred, classes=["positive","negative"],
                          title='Confusion matrix')
    plt.show()
    """
#7 Experiment
class experiment:
    def __init__(self,use_lemmer,use_stemmer, use_stopwords,threshold,tt_ratio,cost,add_tags):
        self.s = use_stemmer
        self.l = use_lemmer
        self.st = use_stopwords
        self. th = threshold
        self.r = tt_ratio
        self.c = cost
        self.add=add_tags


    def experiment_data(self):
        reviews_pos = Reader('pos_shuffle.pos')
        reviews_neg = Reader('neg_shuffle.neg')

        test_sentsp = reviews_pos.shuffle()
        test_sentsn = reviews_neg.shuffle()
        # print(test_sentsp,test_sentsn)
        # Combine Neg and Pos
        combinator = Neg_Pos_Combinator(test_sentsp, test_sentsn, self.r)
        train_x, train_y = combinator.combine_train_return_x_y()
        test_x, test_y = combinator.combine_test_return_x_y()
        # Clean
        train_cleaner = Cleaner(train_x,self.l,self.s,self.st,self.add)
        test_cleaner = Cleaner(test_x,self.l,self.s,self.st,self.add)
        train_cleaned_x = train_cleaner.cleaned()
        test_cleaned_x = test_cleaner.cleaned()
        # Feature extraction
        fp = Feature_Processer()
        train, test, feature_name = fp.count_vector_features_produce(train_cleaned_x, test_cleaned_x, self.th)

        for i in self.c:
            print("Cost at", i)
            csf = classifier(train, train_y, test, test_y)
            csf.logistic(i)
            csf.svm(i)
            csf.NaiveB(i)
            csf.dummy()

Initial.initiation()

def main():

    #use_lemmer,use_stemmer, use_stopwords,threshold,tt_ratio,cost,add_tages
    #cost =  [0.002,0.02,0.2,2]
    cost =  [0.2,2]
    rr_ratio =0.1
    print("Experiment1")
    #frequency = [0.000001,0.00001,0.0001,0.001,0.01,0.1]
    frequency = [1]
    for f in frequency:
        print("Frequency at: ", f)
        experiment1=experiment(True,False,False,f,rr_ratio,cost,False)
        experiment1.experiment_data()


if __name__ == "__main__":
    main()