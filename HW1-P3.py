#In the name of God
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Directory : D:\Works\University\Arshad\Term 3\Deep Learning Sharif University\HW1
Data=pd.read_csv("./Heart_Disease_Dataset.csv")
magnitude_of_Data=Data.shape[0]*Data.shape[1]

print("Magnitude of data is: "+str(magnitude_of_Data)+" or "+str(Data.shape[0])+"*"+str(Data.shape[1]))

##########


num_missing_info=len(np.where(Data.isna()==True)[0])

print("Number of missing information is: "+str(num_missing_info))

##########


class_0=np.where(Data['target']==0)
num_0=np.shape(class_0)[1]

class_1=np.where(Data['target']==1)
num_1=np.shape(class_1)[1]

print("Number of elements in Class 0: "+str(num_0)+"\nNumber of elements in Class 1: "+str(num_1)+"\nMean is: "+str((num_0+num_1)/2)+"\nDifference is: "+str(num_1-num_0)+"\nSo the class is balance")
means_class_0=np.zeros(np.shape(Data)[1]-1)
stds_class_0=np.zeros(np.shape(Data)[1]-1)
means_class_1=np.zeros(np.shape(Data)[1]-1)
stds_class_1=np.zeros(np.shape(Data)[1]-1)
for i in range(Data.shape[1]-1):
    s_1=[]
    s_0=[]
    for j in range(Data.shape[0]):
        if (np.array(Data)[j,-1]==1):
            s_1.append(np.array(Data)[j,i])
        else:
            s_0.append(np.array(Data)[j,i])
            
    means_class_0[i]=np.mean(np.array(s_0))
    means_class_1[i]=np.mean(np.array(s_1))
    stds_class_0[i]=np.std(np.array(s_0))
    stds_class_1[i]=np.std(np.array(s_1))

print("Mean of Class 0 are:"+str(means_class_0))
print("Mean of Class 1 are:"+str(means_class_1))
print("Standard Deviation of Class 0 are:"+str(stds_class_0))
print("Standard Deviation of Class 1 are:"+str(stds_class_1))

print("Mean Absolute Error between means are:"+str(abs(means_class_0-means_class_1)/means_class_0))
print("Mean Absolute Error between stds are:"+str(abs(stds_class_0-stds_class_1)/stds_class_0))

print("\n\n---So according to the MAE, classes are balance and have a suitable distribution")
#Age
Data_numpy=np.array(Data)
elements_of_class_0=np.reshape(Data_numpy[class_0,:],(num_0,12))
elements_of_class_0[:,-1]

elements_of_class_1=np.reshape(Data_numpy[class_1,:],(num_1,12))
elements_of_class_1[:,-1]

ages_class_0=np.array(elements_of_class_0[:,0])


ages_class_1=np.array(elements_of_class_1[:,0])


sexes_class_0=np.array(elements_of_class_0[:,1])


sexes_class_1=np.array(elements_of_class_1[:,1])

plt.figure(figsize=(20,10))

plt.subplot(2,2,1)
plt.hist(ages_class_0)
plt.title("Ages of Class 0")

plt.subplot(2,2,2)
plt.hist(ages_class_1)
plt.title("Ages of Class 1")

plt.subplot(2,2,3)
plt.hist(sexes_class_0)
plt.title("Sexes of Class 0")

plt.subplot(2,2,4)
plt.hist(sexes_class_1)
plt.title("Sexes of Class 1")



############# Remove outlier data

#We consider np.nan if a datum is outlier, then we ignore np.nan values.


all_Data=np.zeros((Data.shape[0],12))
        
for index_data in range(Data_numpy.shape[1]):
    Data_Column=Data_numpy[:,index_data]
    mean=np.mean(Data_Column)
    std=np.std(Data_Column)
    Z_data_Column=(Data_Column-mean)/std
    Data_Column_reduced=np.zeros(Z_data_Column.shape[0])
    index_reduced=[]
    for i in range(Data.shape[0]):
        if(Z_data_Column[i]<=3 and Z_data_Column[i]>=-3):
            Data_Column_reduced[i]=Data_Column[i]
        else:
            Data_Column_reduced[i]=np.nan
    all_Data[:np.shape(Data_Column_reduced)[0],index_data]=Data_Column_reduced

indexes_outlier=np.where(np.isnan(all_Data)==True)
outlier_data=Data_numpy[indexes_outlier]

print("Outlier data are:"+str(outlier_data))

# We ignore rows that contain np.nan values by the follow code:
    
all_Data_reduced=[]
for i in range(Data.shape[0]):
    if i not in indexes_outlier[0]:
        all_Data_reduced.append(all_Data[i])
all_Data_reduced=np.array(all_Data_reduced)       
print("Final size of nonoutlier data is: "+str(all_Data_reduced.shape[0]*all_Data_reduced.shape[1])+" or "+str(all_Data_reduced.shape[0])+"*"+str(all_Data_reduced.shape[1]))

for index_data in ([0,3,4,7,9]):
    all_Data_reduced[:,index_data]=all_Data_reduced[:,index_data]/np.max(all_Data_reduced[:,index_data])


######################## 


from torch.utils.data import random_split
# We use torch library to determine train and test set by using random_split

train_percent=0.7
test_percent=1-train_percent
train_dataset,test_dataset=random_split(all_Data_reduced,[int(train_percent*all_Data_reduced.shape[0]),1+int(test_percent*all_Data_reduced.shape[0])]) 

x_train=np.array(train_dataset)[:,:11]
y_train=np.array(train_dataset)[:,-1]
x_test=np.array(test_dataset)[:,:11]
y_test=np.array(test_dataset)[:,-1]

###################### linear kernel
import numpy as np

# we use SVM function. In this function, if a data is missclassified, we update the w to get nearer to it
# Otherwise, we do nothing for updating weights and biases.
def svm(lr, num_ite, X_train, y_train, X_test):
    w=np.zeros(X_train.shape[1])
    b=0
    for iteration in range(num_ite):
        for index in range(X_train.shape[0]):
            missclassified_data_flag = y_train[index]*(np.dot(X_train[index,:],w)-b)
            if missclassified_data_flag<=1:
                w=w-lr*(-np.dot(X_train[index,:],y_train[index]))
                b=b-lr*y_train[index]
    flag=np.dot(X_test,w)-b
    y_pred=np.sign(flag)
    return y_pred


lr=1e-4
num_ite=2000

y_train2=y_train*2-1# labels are 0 and 1, we should map them to 1 and -1
y_pred=svm(lr=lr,num_ite=num_ite,X_train=x_train,y_train=y_train2,X_test=x_test)



y_test2=2*y_test-1# labels are 0 and 1, we should map them to 1 and -1
print("Accuracy Is: " +str(sum(y_pred==y_test2)/len(y_test2)))

TP=0
TN=0
FP=0
FN=0
for i in range(len(y_test2)):
    if y_test2[i]==1 and y_pred[i]==1:
        TP=TP+1
    if y_test2[i]==1 and y_pred[i]==-1:
        FN=FN+1
    if y_test2[i]==-1 and y_pred[i]==1:
        FP=FP+1
    if y_test2[i]==-1 and y_pred[i]==-1:
        TN=TN+1

Acc=(TP+TN)/(TN+TP+FP+FN)
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1_score=2*(Precision*Recall)/(Precision+Recall)
print("Parameters:\nAccuracy: " +str(Acc)+"\nPrecision: "+str(Precision)+"\nRecall: "+str(Recall)+"\nF1score: "+str(F1_score))

#Accuracy is 80.5157%

###################### RBF kernel
#We consider RBF_features. Then we use this function to produce more features from x_train and x_test

def RBF_features(data,gamma):
    expanded_data=data
    for_range=[0,3,4,7,9]
    for i in (for_range):
        for j in (for_range):
            the_feature=np.exp(-gamma*(data[:,i]-data[:,j])**2)
            the_feature=np.reshape(the_feature,(data.shape[0],1))
            expanded_data=np.concatenate((expanded_data,the_feature),1)

    return expanded_data
gamma=1e-5
x_train2=RBF_features(x_train,gamma=gamma)
x_test2=RBF_features(x_test,gamma=gamma)
print(x_train2.shape)

y_train2=y_train*2-1# labels are 0 and 1, we should map them to 1 and -1
y_pred=svm(lr=lr,num_ite=num_ite,X_train=x_train2,y_train=y_train2,X_test=x_test2)


y_test2=2*y_test-1# labels are 0 and 1, we should map them to 1 and -1
print("Accuracy Is: " +str(sum(y_pred==y_test2)/len(y_test2)))
print("Gamma is: "+str(gamma))

TP=0
TN=0
FP=0
FN=0
for i in range(len(y_test2)):
    if y_test2[i]==1 and y_pred[i]==1:
        TP=TP+1
    if y_test2[i]==1 and y_pred[i]==-1:
        FN=FN+1
    if y_test2[i]==-1 and y_pred[i]==1:
        FP=FP+1
    if y_test2[i]==-1 and y_pred[i]==-1:
        TN=TN+1

Acc=(TP+TN)/(TN+TP+FP+FN)
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1_score=2*(Precision*Recall)/(Precision+Recall)
print("Parameters:\nAccuracy: " +str(Acc)+"\nPrecision: "+str(Precision)+"\nRecall: "+str(Recall)+"\nF1score: "+str(F1_score))

#Accuracy is 84.5272206303725%(for gamma=1)
#Accuracy is 83.95415472779369(for gamma=0.01)
#Accuracy is 83.95415472779369(for gamma=0.01)
###################### polynomial kernel
#We consider polynomial_features. Then we use this function to produce more features from x_train and x_test
# Note that we multiplied all the numerial features together. We could determine every possible multiplication 
# among all data ( so we had 11^d features that are very huge)
def polynomial_features(data,d):
    expanded_data=data
    for_range=[0,3,4,7,9]
    if (d==4):
        for l in (for_range):
            for k in (for_range):
                for i in (for_range):
                    for j in (for_range):
                        the_feature=np.reshape(data[:,i]*data[:,j]*data[:,k]*data[:,l],(data.shape[0],1))
                        expanded_data=np.concatenate((expanded_data,the_feature),1)
            
    elif(d==3):
        for k in (for_range):
            for i in (for_range):
                for j in (for_range):
                    the_feature=np.reshape(data[:,i]*data[:,j]*data[:,k],(data.shape[0],1))
                    expanded_data=np.concatenate((expanded_data,the_feature),1)
        
    else:
        for i in (for_range):
            for j in (for_range):
                the_feature=np.reshape(data[:,i]*data[:,j],(data.shape[0],1))
                expanded_data=np.concatenate((expanded_data,the_feature),1)

    return expanded_data
d=4
lr=1e-4
num_ite=2000

x_train2=polynomial_features(x_train,d=d)
x_test2=polynomial_features(x_test,d=d)
print(x_train2.shape)

y_train2=y_train*2-1# labels are 0 and 1, we should map them to 1 and -1
y_pred=svm(lr=lr,num_ite=num_ite,X_train=x_train2,y_train=y_train2,X_test=x_test2)

y_test2=2*y_test-1# labels are 0 and 1, we should map them to 1 and -1
print("Accuracy Is: " +str(sum(y_pred==y_test2)/len(y_test2)))
print("Degree of polynomial kernel is: "+str(d))

TP=0
TN=0
FP=0
FN=0
for i in range(len(y_test2)):
    if y_test2[i]==1 and y_pred[i]==1:
        TP=TP+1
    if y_test2[i]==1 and y_pred[i]==-1:
        FN=FN+1
    if y_test2[i]==-1 and y_pred[i]==1:
        FP=FP+1
    if y_test2[i]==-1 and y_pred[i]==-1:
        TN=TN+1

Acc=(TP+TN)/(TN+TP+FP+FN)
Precision=TP/(TP+FP)
Recall=TP/(TP+FN)
F1_score=2*(Precision*Recall)/(Precision+Recall)
print("Parameters:\nAccuracy: " +str(Acc)+"\nPrecision: "+str(Precision)+"\nRecall: "+str(Recall)+"\nF1score: "+str(F1_score))

#Accuracy is 85.67335243553008% d=4
#Accuracy is 84.81375358166189% d=3
#Accuracy is 83.3810888252149% d=2

