from math import *
import numpy as np
import csv
import random
import operator


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

###TESTING with random data
Xtrain = [[1,1,1],[2,2,2],[1,3,6]]
Ytrain = ['a','a','b']
Predictor = myKNN(Xtrain,Ytrain) #train the KNN

res = Predictor.predict([[3,3,3],[1,1,1],[3,2,5],[3,4,5]],1) #predict test data
Ytest = ['a','a','b','a'] #actual labels

print("Confusion Matrix:\n" + str(Predictor.getConfusionMatrix(Ytest,res))+"\n")
print("Accuracy:\n" + str(Predictor.getAccuracy(Ytest,res)) + "%"+"\n")


