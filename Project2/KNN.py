import pandas as pd
class KNN:
    def __init__(self, problem, VDMdict, file_norm, discrete):
        self.problem = problem
        self.VDMdict = VDMdict
        self.file_norm = file_norm
        self.discrete = discrete
        self.startKNN()

    def startKNN(self):
        if self.problem == "classification":
            self.classification(self.file_norm)
        if self.problem == 'regression':
            self.regression()

    def classification(self, file):

        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        for cv in range(10): #get test and train datasets
            p = 2               #TUNE currently euclidian distance
            tot = 0
            test = file.iloc[cv::10]
            test.reset_index(drop=True, inplace=True)
            train = pd.concat([file, test]).drop_duplicates(keep=False)     #getting rid of too much
            train.reset_index(drop=True, inplace=True)

            distanceM = pd.DataFrame(index=test.index.values, columns=train.index.values)
            for testrow, test in test.iterrows():
                for trainrow, training in train.iterrows():
                    for indexc, column in train.iteritems():
                        if indexc in self.discrete:   #need to reference VDM
                            datapoint = self.VDMdict.get(indexc)
                            dif = datapoint[test[indexc]][training[indexc]]
                        elif indexc != "class":
                            dif = abs(test[indexc] - training[indexc])

                        tot += dif ** p
                    distance = tot ** (1/p)
                    distanceM.at[testrow, trainrow] = distance

            for index, val in distanceM.iterrows():
                print(type(val))
                min_val = val.idxmin()      #THIS SHOULD WORK, WHY NOT



    def regression(self, VDM, file):
        pass
