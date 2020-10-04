from PrepKNN import PrepKNN
import pandas as pd
import math
class KNN:
    def __init__(self, problem, type, VDMdict, file_norm, discrete):
        self.problem = problem
        self.type = type
        self.VDMdict = VDMdict
        self.file = file_norm
        self.discrete = discrete
        print(self.problem)
        print(self.type)
        prep = PrepKNN(file_norm, self.discrete, self.VDMdict, self.problem)
        self.startKNN(file_norm, prep)

    def startKNN(self, file, prep):
        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        #get every tenth as tuning
        tune = file.iloc[::10]
        tune.reset_index(drop=True, inplace=True)

        file.drop(file.iloc[::10].index)
        file.reset_index(drop=True, inplace=True)

        if self.problem == "classification" and self.type == 'none':
            fold = prep.stratification(file)
            self.tenfold(fold, prep)
        if self.problem == 'regression' and self.type == 'none':
            print('getting every 10th')
            fold = [None] * 10
            for cv in range(10):
                to_test = file.iloc[cv::10]
                to_test.reset_index(drop=True, inplace=True)
                fold[cv] = to_test
            self.tenfold(fold, prep)
        file = self.file.sample(frac=1).reset_index(drop=True)      #shuffles order of rows in file so it's not divided by class
        if self.problem == "classification" and self.type == 'edited':
            self.classEditedKNN(prep, file, tune)
        if self.problem == "regression" and self.type == 'edited':
            self.classEditedKNN(prep, file, tune)
        if self.problem == 'classification' and self.type == 'condensed':
            self.classCondensed(file, prep)
        if self.problem == 'regression' and self.type == 'condensed':
            self.classCondensed(file, prep)

    def classCondensed(self, file, prep):
        print('classes condensed')
        z = pd.DataFrame()
        z = z.append(file.sample(n=1))
        file.drop(z.index)
        print(z)
        for row, rowitem in file.iterrows():
            min = 1000
            for xprime, item1 in z.iterrows():
                #print('z again')
                #print(z)
                dist = prep.getDistance(item1, rowitem)
                if dist < min:
                    min = dist
                    closest = item1
            if closest['class'] != rowitem['class']:
                rowitem = rowitem.to_frame()
                rowitem = rowitem.transpose()
                #print('row item')
                #print('appending to z')
                z = z.append(rowitem)
        z.reset_index(drop=True, inplace=True)
        print(z)

    def tenfold(self, fold, prep):
        for cv in range(10):  # get test and train datasets
            print("Run", cv)
            p = 2  # TUNE currently euclidian distance
            test = fold[cv]
            train_list = fold[:cv] + fold[cv+1:]
            train = pd.concat(train_list)
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)

            if self.problem == "classification":
                distanceM = prep.getDistanceM(test, train)
                results = self.classification(distanceM, test, train)
                print('0/1 loss', results[0])
            if self.problem == 'regression':
                distanceM = prep.getDistanceM(test, train)
                print('regression')
                self.regression(distanceM, test, self.file)


    def classEditedKNN(self, prep, file, tune):
        newLoss = -1        #dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                test = data
                train1 = file[:datarow]
                train2 = file[datarow+1:]
                train = train1.append(train2)
                test = test.to_frame()
                test = test.transpose()
                train.reset_index(drop=True, inplace=True)
                test.reset_index(drop=True, inplace=True)
                distanceM = prep.getDistanceM(test, train)
                class_results = self.classification(distanceM, test, train)
                if class_results[1] == test['class'][0]:
                    file = file.drop([datarow])

                distanceM = prep.getDistanceM(tune, train)
                tune_result = self.classification(distanceM, tune, train)
                newLoss = tune_result[0]
                if datarow == 0:
                    loss = newLoss
            if newLoss != loss:
                print(file)
                break

    def regressEditedKNN(self, prep, file, tune):
        ET = .5        #errorthreshold, TUNE .1 = 10%
        newLoss = -1  # dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                print(datarow)
                test = data
                train1 = file[:datarow]
                train2 = file[datarow + 1:]
                train = train1.append(train2)
                test = test.to_frame()
                test = test.transpose()
                train.reset_index(drop=True, inplace=True)
                test.reset_index(drop=True, inplace=True)
                print(file)
                distanceM = prep.getDistanceM(test, train)
                class_results = self.regression(distanceM, test, file)
                if class_results[1] * (1 - ET) <= float(test['class'][0]) and float(test['class'][0]) <= class_results[1] * (1 + ET):
                    file = file.drop([datarow])
                train1 = file[:datarow]
                train2 = file[datarow + 1:]
                train = train1.append(train2)
                train.reset_index(drop=True, inplace=True)
                print(class_results[1])
                print(float(test['class'][0]))
                print(file)
                distanceM = prep.getDistanceM(tune, train)
                print('classification on tuning')
                tune_result = self.regression(distanceM, tune, train)
                newLoss = tune_result[0]
                if datarow == 0:
                    loss = newLoss
            if newLoss * (1 - ET) >= loss or loss >= newLoss * (1+ET):
                print(file)
                break

    def classification(self, distanceM, test, train):
        k = 5       #number of neighbors
        m = 0       #reset m summation to 0 for next test fold
        voted_class = 'none'       #creates voted_class outside of the for loop so it can be reference for edited
        for row, val in distanceM.iterrows():
            knn_classes = pd.Series()
            min_vals = val.sort_values(ascending=True).head(k)      #gets value of the minimum values for each test point
            print("min_vals", min_vals)
            for min_index, min in min_vals.items():

                col = distanceM.columns[(distanceM == min).iloc[row]].astype(int) #returns training set index for minimum value for each test point
                knn_classes.at[min_index] = train['class'][col[0]]

            voted_class = knn_classes.mode()[0]
            if(test['class'][row] == voted_class): #sees if it has been classified correctly
                m += 1    #increments count of correct classifications
            else:
                m += -1
        loss = m/test.shape[0]
        return ([loss, voted_class])

    def regression(self, distanceM, test, file):
        print('old dstanceM')
        print(distanceM)
        distanceM = distanceM.drop(distanceM.columns[(distanceM == 0).iloc[0]])
        print('new distanceM')
        print(distanceM)
        h = 1       #bandwidth
        n = 5      # number of neighbors
        p = 2       #makes it euclidean distance
        for row, val in distanceM.iterrows():
            val
            min_vals = val.sort_values(ascending=True).head(n)
            numerator = 0
            denominator = 0
            summation = 0
            print(min_vals)
            for small in min_vals:
                first_term = (1/math.sqrt(2 * math.pi)) ** (file.shape[1] - 1)
                exp = math.exp(-1/2*(small ** 2))
                w = first_term * exp
                true_index = min_vals.index[(min_vals == small)].astype(int)

                numerator += float(w) * float(file['class'][true_index[0]])
                denominator += w
            predicted = numerator/denominator
            actual = float(test['class'][row])
            summation += (predicted - actual) ** 2
        mean_sq_error = summation / test.shape[0]
        return([mean_sq_error, predicted])


