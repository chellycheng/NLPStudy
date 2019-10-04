import nltk
import sklearn
class ran:
    def distribution_features_produce(self):
        feature_sets = [ FreqDist(s) for s in self.datas]
        return feature_sets


    def combine_test(self):
        test_set = [(f, 1) for f in self.test_sp]
        test_set.append((f, 0) for f in self.test_sn)
        return test_set

    def combine_train(self):
        trains_set = [(f,1)for f in self.train_sp]
        trains_set.append((f,0) for f in self.train_sn)
        return trains_set

def main():
    print('The nltk version is {}.'.format(nltk.__version__))
    print('The scikit-learn version is {}.'.format(sklearn.__version__))

if __name__ == "__main__":
    main()

