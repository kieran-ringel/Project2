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
            print("Run",cv)
            p = 2               #TUNE currently euclidian distance
            total = 0
            correct = 0
            test = file.iloc[cv::10]
            test.reset_index(drop=True, inplace=True)
            train = pd.concat([file, test]).drop_duplicates(keep=False)     #getting rid of too much
            test.reset_index(drop=True, inplace=True)

            distanceM = pd.DataFrame(index=test.index.values, columns=train.index.values)
            for testrow, testing in test.iterrows():
                for trainrow, training in train.iterrows():
                    tot = 0
                    for indexc, column in train.iteritems():
                        if indexc in self.discrete:   #need to reference VDM
                            datapoint = self.VDMdict.get(indexc)
                            dif = datapoint[testing[indexc]][training[indexc]]
                        elif indexc != "class":
                            dif = abs(float(testing[indexc]) - float(training[indexc]))

                        tot += dif ** p
                    distance = tot ** (1/p)
                    distanceM.at[testrow, trainrow] = distance

            for index, val in distanceM.iterrows():
                min_val = val.min()      #THIS SHOULD WORK, WHY NOT
                col = distanceM.columns[(distanceM == min_val).iloc[index]].astype(int)
                total += 1
                if(test['class'][index] == train['class'][col[0]]):
                    correct += 1
            print(correct/total * 100,"%")



    def regression(self, VDM, file):
        pass
