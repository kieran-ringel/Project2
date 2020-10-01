import pandas as pd
import math
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

        for cv in range(10):  # get test and train datasets
            print("Run", cv)
            p = 2  # TUNE currently euclidian distance
            total = 0
            correct = 0
            test = file.iloc[cv::10]
            test.reset_index(drop=True, inplace=True)
            train = pd.concat([file, test]).drop_duplicates(keep=False)  # getting rid of too much
            train.reset_index(drop=True, inplace=True)

            distanceM = pd.DataFrame(index=test.index.values, columns=train.index.values)
            for testrow, testing in test.iterrows():
                for trainrow, training in train.iterrows():
                    tot = 0
                    for indexc, column in train.iteritems():
                        if indexc in self.discrete:  # need to reference VDM
                            datapoint = self.VDMdict.get(indexc)
                            dif = datapoint[testing[indexc]][training[indexc]]
                        elif indexc != "class":
                            dif = abs(float(testing[indexc]) - float(training[indexc]))


                        tot += dif ** p
                    distance = tot ** (1/p)
                    distanceM.at[testrow, trainrow] = distance

            for row, val in distanceM.iterrows():
                total += 1          #increments count of test points
                min_val = val.min()      #gets value of the minimum values for each test point
                col = distanceM.columns[(distanceM == min_val).iloc[row]].astype(int) #returns training set index for minimum value for each test point
                if(test['class'][row] == train['class'][col[0]]): #sees if it has been classified correctly
                    correct += 1    #increments count of correct classifications
            print(correct/total * 100,"%")  #prints percentage of test points that have been classified correctly


    def regression(self, file):
        h = 1       #bandwidth
        n = 5      # number of neighbors
        p = 2       #makes it euclidean distance
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        for cv in range(10):  # get test and train datasets
            print("Run", cv)
            predicted = 0
            actual = 0
            test = file.iloc[cv::10]
            test.reset_index(drop=True, inplace=True)
            train = pd.concat([file, test]).drop_duplicates(keep=False)  # getting rid of too much
            train.reset_index(drop=True, inplace=True)

            distanceM = pd.DataFrame(index=test.index.values, columns=train.index.values)
            for testrow, testing in test.iterrows():
                for trainrow, training in train.iterrows():
                    tot = 0
                    for indexc, column in train.iteritems():
                        if indexc in self.discrete:  # need to reference VDM
                            datapoint = self.VDMdict.get(indexc)
                            dif = (datapoint[testing[indexc]][training[indexc]])/h
                        elif indexc != "class":
                            dif = (abs(float(testing[indexc]) - float(training[indexc])))/h

                        tot += dif ** p
                    distance = tot ** (1/p)
                    distanceM.at[testrow, trainrow] = distance

            for row, val in distanceM.iterrows():
                min_vals = val.sort_values(ascending=True).head(n)
                numerator = 0
                denominator = 0
                for small in min_vals:
                    first_term = (1/math.sqrt(2 * math.pi)) ** (file.shape[1] - 1)
                    exp = math.exp(-1/2*(small ** 2))
                    w = first_term * exp
                    true_index = min_vals.index[(min_vals == small)].astype(int)

                    numerator += float(w) * float(file['class'][true_index[0]])
                    denominator += w
                ghat = numerator/denominator
                predicted += ghat
                actual += float(test['class'][row])
            print(predicted/actual)
                #print('predicted', ghat)
                #print('actual', test['class'][row])


