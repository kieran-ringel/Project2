import pandas as pd
from KNN import KNN
from KMean import KMean
from Kmedoids import Kmedoids
import math
class ProcessData:
    def __init__(self, file, problem, type, discrete):
        self.file = file
        self.problem = problem
        self.discrete = discrete
        self.type = type
        print('Calculating VDM')
        VDMdict = self.VDMdiscrete(file, discrete)
        print('Normalizing file')
        file_norm = self.normalize(file, discrete)
        #if type is reducedmed, Kmedoids is called and a reduced data set is returned.
        #this data set is then passed into KNN. Same for if type is reducedmean
        if (type == 'reducedmed'):
            reduced = Kmedoids(self.problem, VDMdict, file_norm, self.discrete, file_norm)
            file_norm = reduced.getDataFrame()
            type = 'none'
        if (type == 'reducedmean'):
            reduced = KMean(self.problem, VDMdict, file_norm, self.discrete, file_norm)
            file_norm = reduced.getDataFrame()
            type = 'none' 
        KNN(problem, type, VDMdict, file_norm, discrete)
        

    def normalize(self, file, discrete):
        """Kieran Ringel
        Normalizes all real valued data points using z score normalization"""
        for column in file.iloc[:,:-1]:
            mean= 0
            sd = 0
            if column not in discrete:
                for index,row in file.iterrows():
                    mean += float(file[column][index])
                mean /= file.shape[0]                   #calcualates the mean value for each attribute
                for index,row in file.iterrows():
                    sd += (float(file[column][index]) - mean) ** 2
                sd /= file.shape[0]
                sd = math.sqrt(sd)                      #calculated the standard deviation for each attribute
                for index, row in file.iterrows():
                    if sd == 0:
                        file[column][index] = mean      #gets rid of issue of sd = 0
                    else:
                        file[column][index] = (float(file[column][index]) - mean) / sd  #changed value in file to standardized value
        return(file)

    def VDMdiscrete(self, file, discrete):
        """Kieran Ringel
        Creates a VDM dictionary to hold all of the distance between attributes to be used for discrete attributes
        Does so by making a summation for each pairing of attributes over all the classes. The summation is of the
         count of attribute i corresponding to the class, over the count of attribute i, minus the count of
         attribute j corresponding to the class, over the count of attribute j, this difference is the taken to
         the power of the number of attributes
        """
        VDMdictionary = {}
        if self.discrete != [-1]:
            for features in range(len(discrete)):   #discrete[feature] will give index of feature that is discrete
                column = discrete[features]         #gets column locations dicrete features
                features = []
                classes = []
                for row in range(file.shape[0]):    #iterates through rows of data
                    if file[column][row] not in features:   #if not in features vector
                        features.append(file[column][row])  #add it
                    if file["class"][row] not in classes:   #if not in vector of classes
                        classes.append(file["class"][row])  #add it
                VDM = pd.DataFrame(index = features, columns = features)    #create empty dataframe
                for feat in range(len(features)):
                    for feat2 in range(len(features)):      #iterate over features twice
                        running_sum = 0
                        if self.problem == 'classification':    #if classification
                            for classify in range(len(classes)):    #iterate over classes
                                ci = self.ci(column, features[feat])    #call to calculate ci
                                cia = self.cia(column, classes[classify], features[feat])   #call to calculate cia
                                cj = self.ci(column, features[feat2])
                                cja = self.cia(column, classes[classify], features[feat2])
                                running_sum += abs((cia/ci) - (cja/cj))  #for the summation
                            VDM.at[features[feat], features[feat2]] = running_sum   #after going over all classes, adds to VDM
                        if self.problem == 'regression':            #if regression
                            ci = self.ci(column, features[feat])    #get ci
                            cj = self.ci(column, features[feat2])   #get cj
                            value = abs((ci/file.shape[0]) - (cj/file.shape[0]))    #divide over number of rows because there could be a new class value for every row
                            VDM.at[features[feat], features[feat2]] = value #add to VDM
                VDMdictionary[column] = []
                VDMdictionary[column] = VDM #makes a dictionary of all of the VDMs
        return(VDMdictionary)

    def ci(self, column, feature):
        count = self.file[column].value_counts()[feature]   #count of a feature in the column
        return count

    def cia(self, column, classify, feature):
        class_df = self.file[self.file['class'] == classify]    #count of times a feature is associated with a certain class
        count = class_df[column].value_counts()
        if feature not in count:
            return 0
        else:
            count = count[feature]
        return count



