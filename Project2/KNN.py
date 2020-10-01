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
            self.regression(self.file_norm)

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
            train.reset_index(drop=True, inplace=True)

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
                total += 1          #increments count of test points
                min_val = val.min()      #gets value of the minimum values for each test point
                col = distanceM.columns[(distanceM == min_val).iloc[index]].astype(int) #returns training set index for minimum value for each test point
                if(test['class'][index] == train['class'][col[0]]): #sees if it has been classified correctly
                    correct += 1    #increments count of correct classifications
            print(correct/total * 100,"%")  #prints percentage of test points that have been classified correctly



    def regression(self, file):
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        print(file)

        for cv in range(10):
            test = file.iloc[cv::10]
            test.reset_index(drop=True, inplace=True)
            train = pd.concat([file, test]).drop_duplicates(keep=False)  # getting rid of too much, same as above
            train.reset_index(drop=True, inplace=True)

            for testrow, testing in test.iterrow():
                for trainrow, training in train.iterrows():
                    print(training)
