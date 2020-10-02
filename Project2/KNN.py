from PrepKNN import PrepKNN
import pandas as pd
import math
class KNN:
    def __init__(self, problem, VDMdict, file_norm, discrete):
        self.problem = problem
        self.VDMdict = VDMdict
        self.file_norm = file_norm
        self.discrete = discrete
        prep = PrepKNN(file_norm, self.discrete, self.VDMdict, self.problem)
        self.startKNN(file_norm, prep)

    def startKNN(self, file, prep):
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        if self.problem == "classification":
            fold = prep.stratification(file)
        if self.problem == 'regression':
            fold = [None] * 10
            for cv in range(10):
                to_test = file.iloc[cv::10]
                to_test.reset_index(drop=True, inplace=True)
                fold[cv] = to_test

        self.tenfold(fold, prep)

    def tenfold(self, fold, prep):
        for cv in range(10):  # get test and train datasets
            print("Run", cv)
            p = 2  # TUNE currently euclidian distance
            predicted = 0
            actual = 0
            total = 0
            correct = 0
            test = fold[cv]
            train_list = fold[:cv] + fold[cv+1:]
            train = pd.concat(train_list)
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)

            distanceM = prep.getDistanceM(test, train, p)

            if self.problem == "classification":
                self.classification(distanceM, test, train, total, correct)
            if self.problem == 'regression':
                self.regression(self.file, distanceM, predicted, actual, test)

    def classification(self, distanceM, test, train, total, correct):
        k = 5       #number of neighbors
        for row, val in distanceM.iterrows():
            knn_classes = pd.Series()
            total += 1          #increments count of test points
            min_vals = val.sort_values(ascending=True).head(k)      #gets value of the minimum values for each test point
            for min_index, min in min_vals.items():
                col = distanceM.columns[(distanceM == min).iloc[row]].astype(int) #returns training set index for minimum value for each test point
                knn_classes.at[min_index] = train['class'][col[0]]

            voted_class = knn_classes.mode()[0]
            if(test['class'][row] == voted_class): #sees if it has been classified correctly
                correct += 1    #increments count of correct classifications
        print(correct/total * 100,"%")  #prints percentage of test points that have been classified correctly


    def regression(self, file, distanceM, predicted, actual, test):
        h = 1       #bandwidth
        n = 5      # number of neighbors
        p = 2       #makes it euclidean distance
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


