

import pandas as pd
import math
import random as rand
import numpy as np
import statistics
""" Cierra Wallin
This method implements K-Medoids clustering. First k clusters is initialized to
the square root of the number of observations. This is a tunable parameter. Then,
random data points are assigned for medoids. Following this, each data point
is clustered around a medoid. The the median is calculated from those clusters.
Median values for each feature are computed and the new centroid is returned. Or
the row value that is most similar to the average. This is done for every cluster.
New medoids and clusters are calculated until the medoids no longer change.
However, since this may make our algorithm run for far too long, a cap of 10
repitions was implemented. """
class Kmedoids:
    def __init__(self, problem, VDMdict, file_norm, discrete, reduced):
        self.problem = problem
        self.VDMdict = VDMdict
        self.file_norm = file_norm
        self.discrete = discrete
        self.reduced = file_norm
        df = self.beginKMed(file_norm) #Kmeds returns the reduced data set
        self.reduced = df
        #allows for data frame to be returned and easily passed into KNN
    def getDataFrame(self):
        return (self.reduced)
    
    def beginKMed(self, file):
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)
        length = file.shape[0] #length of data set
        features = file.shape[1] - 1 # columnds of data set
        #creates an estimated k that can be tuned
        estk = math.floor(math.sqrt(length))
        #due to issues with cross validation and some really short reduced
        #data set, estk or clusters must be at least 10.
        if (estk < 10):
            estk = 10
        medoids = []
        centloc = []
        num = 0
        indexes = [0] * estk
        #initialies medoids
        for i in range(estk):
            index = rand.randint(0,length-1)#randomly selects a row to be a medoid
            medoids.append(file.loc[index])
            centloc.append(num)
            num += 1
        #list to give each medoid an indexed value or numerical representation for
        #ease of access
        roids = list(zip(centloc, moids))
        df = self.KMed(file,roids, features)
        return (df)
    #finds which medoid a row is closest to and chooses that medoid
    def closestDist(self,width, p, row,medoids):
        shortD = 10000000000
        chosen = 0
        for i in range (len(medoids)):
            d = self.getDist(row,width,medoids[i][1], p)
            if (d<shortD):
                shortD = d
                chosen = i
        return(chosen)
    #find the row equal/closest to the newly calculated medoid
    def closestVal(self,width , avg, file):
        chosen = 0
        dif = 1000000000
        p = 2
        for row in file.iterrows():
            d = self.getDist(row,width,avg, p)
            if (d<dif):
                dif = d
                chosen = row
        return(chosen)
    #gets the distance between two data points
    def getDist(self, row, width, i, p):
        tot = 0
        dif = 0
        for index in range(width):
            if(index in self.discrete):
                dif = 0
            elif(index != "class"):
                dif = abs(float(row[1][index]) - float(i[index]))
            tot += dif ** p
        distance = tot ** (1 / p)
        return(distance)

    #calculates new medoid based on medians
    #computes the median for each feature
    def getMed(self,width, file, pals, medoids):
        newcent = []
        for i in range(len(medoids)):
            count = 0
            featureindex = range(len(medoids))
            featuremed = []
          
            
            for j in range(len(pals) -1 ):
                if (pals [j][1] == i):
                    secret = file.loc[j]
                    featuremed.append(secret)
            total = []
            for l in range(width):
                temp = []
                if (l in self.discrete):
                    feature = 0
                    total.append(feature)
                elif(l != "class"):
                    for k in range(len(featuremed)):
                        temp.append(featuremed[k][l])
                    if (len(temp)>0):    
                        m = statistics.median(temp)
                    
                    total.append(m)
            #finds row most like the computed median and that row
            #becomes the new medoid
            newc = self.closestVal(width,total, file)
            newcent.append(newc)
            #returns new medoids
        return(newcent)

    def KMed (self,file,medoids, features):
        width = features
        p=2
        increment = 0
        #repeats kmed only 10 times to speed up algorithm
        while(increment <10):
            rows = []
            clusters = []
            num = 1
            #for every row, find which medoid the row should cluster to
            for row in file.iterrows():
                cent = self.closestDist(width, p, row, medoids)
                rows.append(num)
                clusters.append(cent)
                num += 1
            rows = np.array(rows)
            clusters = np.array(clusters)
            #medoids and their indexes or integer representations
            pals = list(zip(rows, clusters))
            medoids = self.getMed(width, file,pals,medoids)
            increment +=1
        df = pd.DataFrame(columns=file.columns)
        #puts chosen medoids into a data frame
        for j in range(len(medoids)):
            row = []
            for i in range (width + 1):
                row.append(medoids[j][1][i])
            df.loc[j] = row
        return(df)
    
