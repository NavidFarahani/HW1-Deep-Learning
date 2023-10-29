#In the name of God
import pandas as pd
import numpy as np
#For understand better the concept of Decision Tree and the application
#of Information_gain on training tree, I studied the following link of youtube:
#https://www.youtube.com/watch?v=ZVR2Way4nwQ
#I could wirte a programm to calculate informain_gains for all features, and I applied it on the data
# But I did not use its application for constructing the desicion tree.

def entropy(y):
    labels = np.unique(y)
    probs=[]
    for i in range(len(labels)):
        prob=np.shape(np.where(y==labels[i]))[1]/len(y)
        probs.append(prob)
    probs=np.array(probs)
    entropy=np.sum(-probs[:]*np.log2(probs[:]))
    return entropy

def information_gain(x: pd.Series, y: pd.Series):
    s=0
    for i in range(len(y)):
        s=s+(len(y[i])/len(x))    *  entropy(y[i])
        
    info_gain=entropy(x)-s
    return info_gain


def information_gains(X,y):
    X=np.array(X)
    a,b=y
    #print(a.shape)
    #print(b.shape)
    a=np.array(a)
    b=np.array(b)
    info_gains=np.zeros((np.shape(X)[1],1))
    #print(info_gains.shape)

    for i in range(len(info_gains)):
        info_gains[i]=information_gain(X[:,i],(a[:,i],b[:,i]))
    
    return info_gains


class Node():
    def __init__(self,best_features=None, threshold=None,left=None,right=None,info_gain=None,value=None,choice=None,children=[]):
        # Our tree is made up of nodes and each node has some attributes such as its position, choice for the next decision
        #, features, threshold for choice, its information_gain(for classification), its value(for the leaf nodes) and finally
        #,its children( the nodes that are down of the node)

        self.best_features=best_features
        self.children=[]
        self.threshold=threshold
        self.choice=choice
        self.left=left
        self.right=right
        self.info_gain=info_gain
        
        self.value = value
        
  
class Tree_Classifier():
    def __init__(self,depth):
        self.depth = depth

    def tree_loop(self,dataset,depth_now=0):
        
        X=dataset[:,:-1]
        Y=dataset[:,-1]

        num_samples=np.shape(X)[0]
        # We do the classification, it we are lower than the depth of tree.
        # and we have more than two samples because if we have 1 sample, it has been classified!
        if (depth_now<=self.depth and num_samples>=2):
            next_nodes=self.find_best_split_for_tree(dataset)
            
            #If information gain is 0, data is completely classified for that
            #feature, otherwise, we split them to two right and left subnodes.
            if next_nodes.info_gain>0:
                right_subtree=self.tree_loop(next_nodes.dataset_right,depth_now+1)
                left_subtree=self.tree_loop(next_nodes.dataset_left,depth_now+1)
                return Node(next_nodes.best_features,next_nodes.threshold,left_subtree,right_subtree,next_nodes.info_gain)

        # value_of_leaf_node is the mod of Y ( the most frequent element)
        Y=list(Y)
        value_of_leaf_node=max(Y,key=Y.count)
        return Node(value=value_of_leaf_node)
    
    def find_best_split_for_tree(self,dataset):
        next_nodes=Node()
        previous_nodes=[]
        
        #In the begining, we consider max_info_gain=-1 because at the first step, we have to classify the tree.
        #But then, we calculate information_gain by help of our function the we store that as max_info_gain to compare
        # the best possible classification.
        max_info_gain=-1
        
        num_features=dataset.shape[1]-1
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            
            # We may have many values, so for better calculation for information_gain, we consider the unique of them.
            # for example, information_gain([1,1,1,0,0,0],([1,1,1],[0,0,0]))=information_gain([1,0],[1],[0])
            possible_thresholds = np.unique(feature_values)

            for threshold in possible_thresholds:
        
                dataset_left=[]
                dataset_right=[]
                
                for i in range(dataset.shape[0]):
                    self.choice=dataset[i,feature_index]<=threshold
                    # if the considered feature of a data is lower than threshold, we selec the choice, 1
                    #,to update the left feature of the node otherwise, we update the right feature of it.
                    
                    if self.choice:
                        dataset_left.append(dataset[i])
                        next_nodes.choice=1
                    else:
                        dataset_right.append(dataset[i])
                        next_nodes.choice=0
                
                dataset_left=np.array(dataset_left)
                dataset_right=np.array(dataset_right)

                
                #if a node is not singular(or it has children), we have to split it into more nodes.
                if self._is_leaf(dataset_left,dataset_right)==False:
                    y=dataset[:,-1]

                    left_y=dataset_left[:,-1]
                    right_y=dataset_right[:,-1]
                    info_gain_now = information_gain(y,(left_y, right_y))
                    #info_gains_now=information_gains(dataset,(dataset_left, dataset_right))
                    #I calculate info_gains_now, but this is not needed for constructing the tree.
                    if info_gain_now>max_info_gain:
                        next_nodes=Node()
                        next_nodes.best_features=feature_index
                        next_nodes.threshold=threshold
                        next_nodes.dataset_left=dataset_left
                        next_nodes.dataset_right=dataset_right
                        next_nodes.info_gain=info_gain_now
                        
                        next_nodes.children=previous_nodes
                     
                        previous_nodes=next_nodes
                        
                        max_info_gain = info_gain_now

                
        return next_nodes
    

    def fit(self,X,Y):
        dataset=np.concatenate((X,Y),axis=1)
        self.root=self.tree_loop(dataset)
        
    def predict(self,X):
        preditions=[]
        #for all the input, we classify the new data.
        for i in range(X.shape[0]):
            preditions.append(self.prediction_for_feature(X[i,:],self.root))
        return preditions
    
    def prediction_for_feature(self,x_row,tree):
        # for every feature of the data, we predict the label independently.
        #, and we split every node, until we reach the leaf nodes. and we continue constructing the
        #, tree, by considering the feature of the node is located at the left side or the right side.
        if tree.value!=None:
            return tree.value
        
        feature_val=x_row[tree.best_features]
        if feature_val<=tree.threshold:
            return self.prediction_for_feature(x_row,tree.left)
        else:
            return self.prediction_for_feature(x_row,tree.right)


    def _is_leaf(self,dataset_left,dataset_right):
        return (len(dataset_left)==0 or len(dataset_right)==0)


######### New dataset


Data=pd.read_csv("./diabetes.csv")

from torch.utils.data import random_split
Data2=np.array(Data)
train_percent=0.8# 0.9
test_percent=1-train_percent
num_train=int(Data2.shape[0]*train_percent)
num_test=int(Data2.shape[0]*test_percent)
train_dataset,test_dataset=random_split(Data2,[num_train,1+num_test]) 
train_dataset=np.array(train_dataset)
test_dataset=np.array(test_dataset)
x_train=train_dataset[:,:-1]
x_test=test_dataset[:,:-1]
y_train=train_dataset[:,-1]
y_test=test_dataset[:,-1]


y_train=np.reshape(y_train,(y_train.shape[0],1))

classifier = Tree_Classifier(depth=5)
classifier.fit(x_train,y_train)

Y_pred = np.uint8(classifier.predict(x_test))
print("Accuracy Is: " +str(sum(Y_pred==y_test)/len(y_test)))




########## MNIST data
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from sklearn.decomposition import PCA
m=2000
x_train=x_train[:m,:,:]
y_train=y_train[:m]
x_test=x_test[:m//6,:,:]
y_test=y_test[:m//6]
def do_pca(n_components,x_train,x_test):
    pca = PCA(n_components=n_components)
    
    x_train_vector=np.reshape(x_train,(x_train.shape[0],x_train.shape[1]*x_train.shape[2]))/255
    x_test_vector=np.reshape(x_test,(x_test.shape[0],x_test.shape[1]*x_test.shape[2]))/255
    x_train_vector=x_train_vector.astype("float32")
    x_test_vector=x_test_vector.astype("float32")
    
    pca_components_train = pca.fit_transform(x_train_vector[:60000,:])
    pca_components_test = pca.transform(x_test_vector[:10000,:])
    return pca_components_train,pca_components_test

pc=10
pca_components_train,pca_components_test=do_pca(pc,x_train,x_test)
classifier = Tree_Classifier(depth=12)
y_train2=np.reshape(y_train,(len(y_train),1))
classifier.fit(pca_components_train,y_train2)


Y_pred = np.uint8(classifier.predict(pca_components_test))
print("Accuracy Is: " +str(sum(Y_pred==y_test)/len(y_test)))

#Accuracy Is: 0.6546546546546547 (for 10 components and 2000 images.)
#Accuracy Is: 0.7322929171668667 (for 10 components and 10000 images.)
#####################
