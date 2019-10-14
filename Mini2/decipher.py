#import the useful library
# 1 command line operation
import os
import argparse
import nltk
import numpy as np

class reader:
    def parserArg(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-laplace", action="store_true", default=False,help="enable the laplace smoothing for hmms",dest='laplace',required=False)
        parser.add_argument("-lm", action="store_true", default=False,help="enable the improved plaintext modelling",dest='lm',required=False)
        parser.add_argument("folder",action="store",help="input the fold path")
        args = parser.parse_args()
        lm = args.lm
        laplace = args.laplace
        fold_path = args.folder
        all_files = os.listdir(fold_path+'/')
        print(all_files)
        files = [open(fold_path+'/'+f,"r") for f in all_files]
        test_cipher_file, train_ciper_file, test_plain_file,train_plain_file = files[0],files[1],files[2],files[3]
        test_cipher=  self.reading(test_cipher_file)
        train_ciper= self.reading(train_ciper_file)
        test_plain= self.reading(test_plain_file)
        train_plain= self.reading(train_plain_file)

        return test_cipher,train_ciper,test_plain,train_plain
    def reading(self, file):
        data_p = [ (line) for line in file]
        return data_p

class feature_lable:
    def create_labbled_list(self,sents_list,target_list):
        sents_modified = self.split_list(sents_list)
        #print(sents_modified)
        target_list = self.split_list(target_list)
        lable_sequence =[]
        for j in range(len(sents_modified)):
            s = sents_modified[j]
            t = target_list[j]
            lable_sequence.append([(s[i],t[i]) for i in range(len(s))])
        #print(lable_sequence)
        return lable_sequence


    def split_list(self,sents_list):
        sents_modified = [list(s) for s in sents_list[:1]]
        sents_modified_clean = [filter(lambda x: x not in ['\r','\n',],s) for s in sents_modified]
        #print(sents_modified_clean)
        return sents_modified_clean

class decoder:

    def decoder1(self,lable_sequence,test_cipher,test_plain):
        trainer = nltk.tag.hmm.HiddenMarkovModelTrainer()
        tagger = trainer.train_supervised(lable_sequence)
        test_tags = tagger.tag(test_cipher)
        #test_tags = tagger.tag_sents(test_cipher)
        print(test_tags)
        #tagger.evaluate(test_plain)
        

class main:
    test_cipher,train_cipher,test_plain,train_plain = reader().parserArg()
    #print("Tain_plain:", train_plain)
    fl = feature_lable()
    lable_sequence = fl.create_labbled_list(train_cipher,train_plain)
    test_cipher = fl.split_list(test_cipher)
    test_plain = fl.split_list(test_plain)
    dc = decoder()
    dc.decoder1(lable_sequence,test_cipher,test_plain)

if __name__ == '__main__':
    main()
