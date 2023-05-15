import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import pandas as pd
from sklearn import metrics, datasets, utils, model_selection
import numpy as np
import matplotlib.pyplot as mp


wine = datasets.load_wine(as_frame=True)

# wine_df = pd.DataFrame({'data' : wine["data"], 'target': wine["target"]})

wine_df = wine.frame

class NaiveBayes:
    
    def __init__(self, X, y, priors = None):
        
        '''
        X and y denotes the features and the target labels respectively
        '''
        self.X, self.y = X, y 
        
        self.N = len(self.X) # Length of the training set

        self.dim = len(self.X.columns) # Dimension of the vector of features
        # print(self.X.columns)

        self.classes, self.priors = np.unique(y, return_counts=True)    

        self.n_classes = len(self.classes)

        self.class_dict = {range(self.n_classes)[i] : self.classes[i] for i in range(self.n_classes)}

        if(priors is None):
            self.priors = self.priors/self.N
        else:
            self.priors = np.array(priors)/100
        
        self.means = [self.X.loc[self.y == i, :].mean(axis = 0).to_list() for i in self.class_dict.values()]

        self.stds = [self.X.loc[self.y == i, :].std(axis = 0).to_list() for i in self.class_dict.values()]

        print("Priors - ", self.priors) #, self.class_dict, self.means, self.stds)
            
            

    def classify(self, entry):

        def to_exp(list, exp=1): 
            return np.array([number**exp for number in list])

        probability = []
        
        entry = np.array(entry)
        self.means = np.array([np.array(l) for l in self.means])
        self.stds = np.array([np.array(l) for l in self.stds])    
        self.priors = np.array(self.priors)
        probability = np.array([(np.prod(1/np.sqrt(2*np.pi*to_exp(self.stds[i],2))*np.exp(-0.5*to_exp(((entry-self.means[i])/self.stds[i]), 2)))*self.priors[i]) for i in self.class_dict.keys()])
        
        probability = probability / probability.sum()

        return self.class_dict[list(probability).index(max(probability))]

# Shuffle
wine_df = utils.shuffle(wine_df, random_state = 42).reset_index().drop('index', axis='columns')

# Split 70:30
sss = model_selection.StratifiedShuffleSplit(n_splits=1, train_size=0.7, random_state=42)

for train_index, test_index in sss.split(wine_df.iloc[:, 0:13], wine_df["target"]):
    pass # print(train_index, test_index )

X_train, X_test = wine_df.iloc[train_index, 0:13], wine_df.iloc[test_index, 0:13]
y_train, y_test = wine_df["target"][train_index], wine_df["target"][test_index]

#plot test and train class distributions
class_unique_train, class_count_train = np.unique(y_train, return_counts=True)    
class_unique_test, class_count_test = np.unique(y_test, return_counts=True)    

ax = mp.subplot(111)
train_bar = ax.bar(class_unique_train-0.2, class_count_train, width=0.4, ec='black', align='center')

ax.set_xlabel('Class')
ax.set_ylabel('Frequency')

test_bar = ax.bar(class_unique_test+0.2, class_count_test, width=0.4, ec='black', align='center')

ax.legend([train_bar, test_bar], ["Train Distribution", "Test Distribution"])
for rect in train_bar + test_bar:
    height = rect.get_height()
    mp.text(rect.get_x() + rect.get_width()/2.0, height, '%d' % int(height), ha='center', va='bottom')

mp.show()

#Train with NB
nbc = NaiveBayes(X_train, y_train)

# Train
y_train_predicted = pd.Series(index = y_train.index)

for ind, row in X_train.iterrows():
    y_train_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_train, y_train_predicted)

conf_mat = metrics.confusion_matrix(y_train, y_train_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.a) Train \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)

# Test
y_predicted = pd.Series(index = y_test.index)

for ind, row in X_test.iterrows():
    y_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_test, y_predicted)

conf_mat = metrics.confusion_matrix(y_test, y_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.a) Test \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)


# Part b, a
nbc = NaiveBayes(X_train, y_train, [40, 40, 20])

# Train
y_train_predicted = pd.Series(index = y_train.index)

for ind, row in X_train.iterrows():
    y_train_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_train, y_train_predicted)

conf_mat = metrics.confusion_matrix(y_train, y_train_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.b)a) Train \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)

# Test
y_predicted = pd.Series(index = y_test.index)

for ind, row in X_test.iterrows():
    y_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_test, y_predicted)

conf_mat = metrics.confusion_matrix(y_test, y_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.b)a) Test \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)

# Part b, b
nbc = NaiveBayes(X_train, y_train, [80, 10, 10])

# Train
y_train_predicted = pd.Series(index = y_train.index)

for ind, row in X_train.iterrows():
    y_train_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_train, y_train_predicted)

conf_mat = metrics.confusion_matrix(y_train, y_train_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.b)b) Train \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)

# Test
y_predicted = pd.Series(index = y_test.index)

for ind, row in X_test.iterrows():
    y_predicted[ind] = nbc.classify(list(row)[0:13])

acc_overall = metrics.accuracy_score(y_test, y_predicted)

conf_mat = metrics.confusion_matrix(y_test, y_predicted)

class_acc = conf_mat.diagonal()/conf_mat.sum(axis = 1)

print("Q2.b)b) Test \nOverall Acc. - ", acc_overall, "\nConfusion Matrix - \n", conf_mat, "\nClasswise Acc. - ", class_acc)






# ROC Code
# fig, ax = mp.subplots(figsize=(8,6))

# mp.plot([0, 1], [0, 1], 'k--')
# mp.xlim([0.0, 1.0])
# mp.ylim([0.0, 1.05])
# mp.xlabel('False Positive Rate')
# mp.ylabel('True Positive Rate')
# mp.title('Receiver operating characteristic example')

# for i in range(3):

#     y_test_target = np.int32(y_test == i)
#     y_test_predicted = np.int32(y_predicted == i)
    
#     fpr, tpr, thresholds = metrics.roc_curve(y_test_target, y_test_predicted)
#     roc_auc = metrics.auc(fpr, tpr)
#     # mp.subplot(2,2,i+1)
#     mp.plot(fpr, tpr, label='ROC curve class %d (area = %0.2f)' % (i, roc_auc))

# mp.legend(loc="lower right")
# mp.show()
