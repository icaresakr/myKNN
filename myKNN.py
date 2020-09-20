from math import *
import numpy as np
from csv import reader
import random
import operator
#sfrom sklearn import datasets, neighbors
import matplotlib.pyplot as plt
#from mlxtend.plotting import plot_decision_regions


#To test with data for now

class myKNN:
    trainData = [[]] #array of sample dimensions
    trainClasses = [] #list of each sample's class
    n = 0 #number of variables
    N = 0 #dimension
    K=1 #

    def __init__(self,inputData,inputClasses):
        self.trainData = inputData
        self.trainClasses = inputClasses
        self.n = len(inputData) #number of samples
        self.N = len(inputData[0]) #dimension (number of features) , -1 because the class is not a dimension

    def train(self, inputData, inputClasses):
        self.trainData = inputData
        self.trainClasses = inputClasses
        self.n = len(inputData) #number of samples
        self.N = len(inputData[0]) #dimension (number of features) , -1 because the class is not a dimension

    def __euclidianDistance(self, dat1, dat2):
        dist = 0
        for j in range(self.N): #calculate euclidian distance
            dist = dist + pow((dat1[j] - dat2[j]),2)
        dist = sqrt(dist)
        return(dist)

    def __processDistances(self,sDat): #calculate distances and predict the test sample
        distVect = [0 for i in range(self.n)] #vector to store distances from each train sample

        for i in range(self.n): #for each train sample
            distVect[i] = self.__euclidianDistance(self.trainData[i],sDat)
        #WE NOW HAVE THE DISTANCE SEPARATING THE TEST VECTOR FROM ALL THE TRAIN VECTORS
        return(distVect)

    def __getKnearest(self, distVect, K): #deduce the K nearest from the distance vector
        res = []
        tempVect = distVect
        if(K<self.n):
            for i in range(K):
                idx = np.argmin(tempVect)
                res = res + [idx] # store the indexes of the nearest neighbors
                tempVect.pop(idx)
        return(res)

    def __majorityVoting(self, kNearest):
        votes = {} #dictionary of classes and corresponding votes
        for i in range(len(kNearest)):
            vote = self.trainClasses[kNearest[i]]#the class at index kNearest[i]
            if vote in votes:
                votes[vote]+=1
            else:
                votes[vote]=1
        val = max(votes.items(), key=operator.itemgetter(1))[0] #get class with maximum votes
        return(val)

    def predict(self, testData, K):
        predictedClasses = []
        for i in range(len(testData)): #for each test sample
            #calculate its euclidian distance from each of the training sample
            distVect = self.__processDistances(testData[i])
            kNearest = self.__getKnearest(distVect,K)
            predictedClasses += [self.__majorityVoting(kNearest)]
        return(predictedClasses)

    def getAccuracy(self, Ytest, Ypredicted):
        correct = 0
        for i in range(len(Ytest)):
            if Ytest[i] is Ypredicted[i]:
                correct += 1
        return (correct/float(len(Ytest))) * 100.0

    def getConfusionMatrix(self, Ytest, Ypredicted):
        cl1 = Ytest[0]
        confM = np.zeros([2, 2], dtype = int)
        for i in range(len(Ytest)):
            #Actual positive
            if (Ypredicted[i] is Ytest[i]) and (Ytest[i] is Ytest[0]) :
                confM[0][0]+=1
            if (Ypredicted[i] is not Ytest[i]) and (Ytest[i] is Ytest[0]) :
                confM[1][0]+=1

            #Actual negative
            if (Ypredicted[i] is Ytest[i]) and (Ytest[i] is not Ytest[0]) :
                confM[1][1]+=1
            if (Ypredicted[i] is not Ytest[i]) and (Ytest[i] is not Ytest[0]) :
                confM[0][1]+=1

        return confM

### Other Methods
def load_csv(filename):
	dataset = list()
	Ylabels = list()
	forbidden ='?'
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
		    if not row:
		        continue
		    if forbidden not in row: #remove data containing a '?'
		        dataset.append([int(row[i]) for i in range(1,len(row)-1)]) #without the labels
		        Ylabels.append(int(row[-1])) #the labels
	return (dataset,Ylabels)

def split_data(X,Y,probability):
    X1 = list()
    X2 = list()
    Y1 = list()
    Y2 = list()
    #X and Y should have same dimension
    for i in range(len(X)):
        j = random.random()
        if j<=probability:
            X1.append(X[i])
            Y1.append(Y[i])
        else:
            X2.append(X[i])
            Y2.append(Y[i])
    return(X1,Y1,X2,Y2)

def dataset_minmax(dataset):
	minmax = list()
	for i in range(len(dataset[0])):
		col_values = [row[i] for row in dataset]
		value_min = min(col_values)
		value_max = max(col_values)
		minmax.append([value_min, value_max])
	return minmax


def normalize_data(X):
    minmax = dataset_minmax(X)
    for row in X:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])

#plot for 2 classes
def plot_data(dim1, dim2, X, Y):
    for i in range(len(X)):
        if(Y[i]==Y[0]):
            plt.plot(X[i][dim1], X[i][dim2],'bs') #in blue square for class 1
        else:
            plt.plot(X[i][dim1], X[i][dim2],'g^') #in green triangle for class 2
    plt.show()

### First data set
Xall, Yall = load_csv("./breast-cancer-wisconsin.data")

#normalize_data(Xall)


Xtrain,Ytrain,Xtest,Ytest = split_data(Xall,Yall,0.8) # split data 80% train, 20% test

Predictor = myKNN(Xtrain,Ytrain)

##CHECK OPTIMAL VALUE FOR K
'''
#for plotting error as a function of K
error  = []
Ks = []

for k in range(1,20):
    res = Predictor.predict(Xtest,k) #predict test data

    #print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest,res))+"\n")
    error.append(100 - Predictor.getAccuracy(Ytest,res))
    Ks.append(k)


plt.plot(Ks, error)
plt.xlabel('K value')
plt.ylabel('Error (%)')

plt.title('Test prediciton Error as a function of K')

plt.show()
'''


#K = 2 IS OPTIMAL FOR THIS DATA SET (or Ks[argmin(error)])

##BEFORE NORMALIZING DATA
res = Predictor.predict(Xtest,2) #predict test data

print("DATA SET 1 : Breast Cancer Wisconsin (Diagnostic) Data Set")
#Accuracy
print("Accuracy:\n" + str(Predictor.getAccuracy(Ytest,res))+"\n")

#Confusion matrix
print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest,res))+"\n")

#NORMALIZE DATA
#normalize_data(Xtrain)
#normalize_data(Xtest)

##AFTER NORMALIZING DATA
'''
Predictor.train(Xtrain,Ytrain)

res2 = Predictor.predict(Xtest,2)

#Accuracy
print("Accuracy:\n" + str(Predictor.getAccuracy(Ytest,res2))+"\n")

#Confusion matrix
print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest,res2))+"\n")

'''
#no difference because all the variables have the same range of variation

##



#plot the data
#plot_data(6,7,Xall,Yall)



### Second data set
Xall2, Yall2 = load_csv("./haberman.data")

#normalize_data(Xall)




Xtrain2,Ytrain2,Xtest2,Ytest2 = split_data(Xall2,Yall2,0.8) # split data 80% train, 20% test

Predictor2 = myKNN(Xtrain2,Ytrain2)

# CHECK optimal K for this data set
error  = []
Ks = []

for k in range(1,20):
    res = Predictor2.predict(Xtest2,k) #predict test data

    #print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest,res))+"\n")
    error.append(100 - Predictor2.getAccuracy(Ytest2,res))
    Ks.append(k)


plt.plot(Ks, error)
plt.xlabel('K value')
plt.ylabel('Error (%)')

plt.title('Test prediciton Error as a function of K')

plt.show()


#Find optimal K for this data set
chosenK = Ks[np.argmin(error)]

res2 = Predictor2.predict(Xtest2,chosenK) #predict test data

print("DATA SET 2 : Haberman's Survival Data Set")
print("Before Normalization")
#Accuracy
print("Accuracy:\n" + str(Predictor.getAccuracy(Ytest2,res2))+"\n")

#Confusion matrix
print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest2,res2))+"\n")

##After normalizing
normalize_data(Xtrain2)
normalize_data(Xtest2)

Predictor.train(Xtrain2,Ytrain2)

res2n = Predictor.predict(Xtest2,chosenK)
print("After Normalization")
#Accuracy
print("Accuracy:\n" + str(Predictor.getAccuracy(Ytest2,res2n))+"\n")

#Confusion matrix
print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest2,res2n))+"\n")




