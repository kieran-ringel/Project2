import pandas as pd
class PrepKNN:
    def __init__(self, file, discrete, VDMdict, problem):
        self.file = file
        self.discrete = discrete
        self.VDMdict = VDMdict
        self.problem = problem

    def stratification(self, file):
        """Kieran Ringel
        Makes sure that all fold are representative of the make up of the file
        Also makes sure each fold has at least one of each class
        """
        fold_size = file.shape[0] / 10
        count = file['class'].value_counts(normalize=True)
        fold = [None] * 10
        for cv in range(10):
            one_fold = pd.DataFrame()
            for classes, dataframe in file.groupby(by='class'):
                dataframe.reset_index(drop=True, inplace=True)
                proportion = count[classes]         #gets portions of the classes
                ex_per_fold = fold_size * proportion    #gets examples per fold of each class

                beginning = int(cv * ex_per_fold)   #gets begining of fold for a class
                end = int((cv + 1) * ex_per_fold)   #gets end of a fold for a class
                one_fold = one_fold.append(dataframe.loc[beginning:end, :]) #appends the data for the class
            fold[cv] = one_fold # once it contains all classes it is put in the array
        return(fold)

    def getDistanceM(self, test, train):
        """Kieran Ringel
        Creates distance matrix between all test and training points using euclidean distance"""
        p = 2 # TUNE currently euclidian distance
        distanceM = pd.DataFrame(index=test.index.values, columns=train.index.values)
        for testrow, testing in test.iterrows():
            for trainrow, training in train.iterrows():
                tot = 0
                for indexc, column in test.iteritems():
                    #print(indexc)
                    if indexc in self.discrete:  # need to reference VDM
                        datapoint = self.VDMdict.get(indexc)
                        dif = datapoint[testing[indexc]][training[indexc]]
                    elif indexc != "class": #get distance beween 2 points
                        dif = abs(float(testing[indexc]) - float(training[indexc]))

                    tot += dif ** p
                distance = tot ** (1 / p)   #distance is calculated
                distanceM.at[testrow, trainrow] = distance  #put in distance matrix
        return(distanceM)

    def getDistance(self, pt1, pt2):
        """Kieran Ringel
        Gets disatance between 2 points when a full matrix does not need to be made"""
        p = 2           #euclidean distance
        tot = 0
        for indexc, column in pt1.iteritems():
            if indexc in self.discrete:  # need to reference VDM
                datapoint = self.VDMdict.get(indexc)
                dif = datapoint[pt1[indexc]][pt2[indexc]]
            elif indexc != "class":     #gets distance beween 2 points
                dif = abs(float(pt1[indexc]) - float(pt2[indexc]))

            tot += dif ** p
        distance = tot ** (1 / p)
        return(distance)

