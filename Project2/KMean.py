##implement k-means clustering and
##use the cluster centroids as a reduced
##data set for k-NN.

import pandas as pd
import math
import random as rand
import numpy as np
from PrepKNN import PrepKNN
from KNN import KNN

class KMean:
    def __init__(self, problem, VDMdict, file_norm, discrete, reduced):
        self.problem = problem
        self.VDMdict = VDMdict
        self.file_norm = file_norm
        self.discrete = discrete
        self.reduced = file_norm
        df = self.beginKMean(file_norm)
        self.reduced = df
        #allows for data frame to be returned and easily passed into KNN
    def getDataFrame(self):
        return (self.reduced)
        
    def beginKMean(self, file):
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)
        length = file.shape[0]
        features = file.shape[1] - 1
        #creates an estimated k that can be tuned
        estk = math.floor(math.sqrt(length))
        print(estk)
        if (estk<10):
            estk = 10
        print(estk)
        centroids = []
        centloc = []
        num = 0
        indexes = [0] * estk
        #initializes centroids
        for i in range(estk):
            index = rand.randint(0,length-1)
            centroids.append(file.loc[index])
            centloc.append(num)
            num += 1
        #list to give each centroid an indexed value so they are easier to work with
        roids = list(zip(centloc, centroids))
        df = self.KMean(file,roids, features)
        return (df)
    
    #finds which centroid a row is closest to, and chooses that centroid
    def closestDist(self,width, p, row,centroids):
        shortD = 10000000000
        chosen = 0
        for i in range (len(centroids)):
            d = self.getDist(row,width,centroids[i][1], p)
            if (d<shortD):
                shortD = d
                chosen = i
        return(chosen)
    #finds the row equal/closest to equal the newly calculated centroid
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

     #calculates new centroids based on means                        
    def getMean(self,width, file, pals, centroids):
        newcent = []
        for i in range(len(centroids) ):
            count = 0
            featureavg = [0] * width
            for j in range(len(pals) ):
                if (pals [j][1] == i):
                    secret = file.loc[j]
                    for k in range(width):
                        if(k in self.discrete):
                            val = 0
                        elif(k != "class"):
                            featureavg[k]  += secret[k]
                    count += 1
            if (count == 0):
                count = 1
            for l in range(width):
                featureavg[l] = featureavg[l]/count
            
            newc = self.closestVal(width,featureavg, file)
            newcent.append(newc)
        return(newcent)

    
   
    def KMean (self,file,centroids, features):
        width = features
        p=2 #euclidean distance
        increment = 0
        #repeats kmeans 10 times in order to speed up algorithm
        while(increment <10):
            rows = []
            clusters = []
            num = 1
            for row in file.iterrows():
                cent = self.closestDist(width, p, row, centroids)
                rows.append(num)
                clusters.append(cent)
                num += 1
            rows = np.array(rows)
            clusters = np.array(clusters)
            pals = list(zip(rows, clusters))
            centroids = self.getMean(width, file,pals,centroids)
            increment +=1
        df = pd.DataFrame(columns=file.columns)
        
        for j in range(len(centroids)):
            row = []
            for i in range (width + 1):
                row.append(centroids[j][1][i])
            df.loc[j] = row
        #returns dataframe that can be passed into KNN
        return(df)
    
        
                   
            
            
    
        
            
