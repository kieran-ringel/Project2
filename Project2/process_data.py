import pandas as pd
from KNN import KNN
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
        if (type == 'reducedmed'):
            reduced = Kmedoids(self.problem, VDMdict, file_norm, self.discrete, file_norm)
            file_norm = reduced.getDataFrame()
            print(file_norm)
            type = 'none'
        if (type == 'reducedmean'):
            reduced = KMean(self.problem, VDMdict, file_norm, self.discrete, file_norm)
            file_norm = reduced.getDataFrame()
            print(file_norm)
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
        VDMdictionary = {}
        if self.discrete != [-1]:
            for features in range(len(discrete)):   #discrete[feature] will give index of feature that is discrete
                column = discrete[features]
                features = []
                classes = []
                for row in range(file.shape[0]):
                    if file[column][row] not in features:
                        features.append(file[column][row])
                    if file["class"][row] not in classes:
                        classes.append(file["class"][row])
                VDM = pd.DataFrame(index = features, columns = features)
                for feat in range(len(features)):
                    for feat2 in range(len(features)):
                        running_sum = 0
                        if self.problem == 'classification':
                            for classify in range(len(classes)):
                                ci = self.ci(column, features[feat])
                                cia = self.cia(column, classes[classify], features[feat])
                                cj = self.ci(column, features[feat2])
                                cja = self.cia(column, classes[classify], features[feat2])
                                running_sum += abs((cia/ci) - (cja/cj))
                            VDM.at[features[feat], features[feat2]] = running_sum
                        if self.problem == 'regression':
                            ci = self.ci(column, features[feat])
                            cj = self.ci(column, features[feat2])
                            value = abs((ci/file.shape[0]) - (cj/file.shape[0]))
                            VDM.at[features[feat], features[feat2]] = value
                VDMdictionary[column] = []
                VDMdictionary[column] = VDM
        return(VDMdictionary)

    def ci(self, column, feature):
        count = self.file[column].value_counts()[feature]
        return count

    def cia(self, column, classify, feature):
        class_df = self.file[self.file['class'] == classify]
        count = class_df[column].value_counts()
        if feature not in count:
            return 0
        else:
            count = count[feature]
        return count



