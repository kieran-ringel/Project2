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
        #get every tenth as tuning
        tune = file.iloc[::10]
        tune.reset_index(drop=True, inplace=True)
        #removes all every tenth from the file
        file.drop(file.iloc[::10].index)
        file.reset_index(drop=True, inplace=True)

        k = self.tuningk(tune, file, prep)
        print('k is', k)

        file.sort_values(by="class", inplace=True)
        file.reset_index(drop=True, inplace=True)

        if self.problem == "classification" and self.type == 'none':
            fold = prep.stratification(file)
            self.tenfold(fold, prep, k)
        if self.problem == 'regression' and self.type == 'none':
            print('getting every 10th')
            fold = [None] * 10
            for cv in range(10):
                to_test = file.iloc[cv::10]
                to_test.reset_index(drop=True, inplace=True)
                fold[cv] = to_test
            self.tenfold(fold, prep, k)
        file = self.file.sample(frac=1).reset_index(drop=True)      #shuffles order of rows in file so it's not divided by class
        if self.problem == "classification" and self.type == 'edited':
            reduced_file = self.classEditedKNN(prep, file, tune, k)
            fold = prep.stratification(reduced_file)
            self.tenfold(fold, prep, k)
        if self.problem == 'regression' and self.type == 'edited':
            new_file = self.regEditedKNN(prep, file, tune, k)
            fold = [None] * 10
            for cv in range(10):
                to_test = new_file.iloc[cv::10]
                to_test.reset_index(drop=True, inplace=True)
                fold[cv] = to_test
            self.tenfold(fold, prep, k)
        if self.type == 'condensed':
            new_file = self.condensed(file, prep)
            print('got the file')
            if self.problem == 'classification':
                fold = prep.stratification(new_file)
                self.tenfold(fold, prep, k)
            if self.problem == 'regression':
                fold = [None] * 10
                for cv in range(10):
                    to_test = new_file.iloc[cv::10]
                    to_test.reset_index(drop=True, inplace=True)
                    fold[cv] = to_test
                self.tenfold(fold, prep, k)

    def tuningk(self, tuning, file, prep):
        k = pd.Series([3, 4, 5, 6, 7])
        loss = pd.Series(5)
        for index in range(len(k)):
            distanceM = prep.getDistanceM(tuning, file)
            if self.problem == 'classification':
                results = self.classification(distanceM, tuning, file, k[index])
                loss[index] = results[0]
            if self.problem == 'regression':
                results = self.regression(distanceM, tuning, file, k[index])
                loss[index] = results[0]
        minindex = loss.idxmin()
        return(k[minindex])

    def tenfold(self, fold, prep, neighbors):
        tot_loss = 0
        for cv in range(10):  # get test and train datasets
            print("Run", cv+1)
            test = fold[cv]
            print(test)
            train_list = fold[:cv] + fold[cv+1:]
            train = pd.concat(train_list)
            train.reset_index(drop=True, inplace=True)
            test.reset_index(drop=True, inplace=True)

            if self.problem == "classification":
                distanceM = prep.getDistanceM(test, train)
                results = self.classification(distanceM, test, train, neighbors)
                print('0/1 loss', results[0])
                tot_loss += results[0]
            if self.problem == 'regression':
                distanceM = prep.getDistanceM(test, train)
                results = self.regression(distanceM, test, self.file, neighbors)
                print('Squared Loss', results[0])
                tot_loss += results[0]
        avg_loss = tot_loss/10
        print(avg_loss)

    def classification(self, distanceM, test, train, k):
        m = 0       #reset m summation to 0 for next test fold
        voted_class = 'none'       #creates voted_class outside of the for loop so it can be reference for edited
        for row, val in distanceM.iterrows():
            knn_classes = pd.Series()
            min_vals = val.sort_values(ascending=True).head(k)      #gets value of the minimum values for each test point
            for min_index, min in min_vals.items():
                col = distanceM.columns[(distanceM == min).iloc[row]].astype(int) #returns training set index for minimum value for each test point
                knn_classes.at[min_index] = train['class'][col[0]]

            voted_class = knn_classes.mode()[0]
            if(test['class'][row] == voted_class): #sees if it has been classified correctly
                m += 0
            else:
                m += 1
        loss = m/test.shape[0]
        return ([loss, voted_class])

    def regression(self, distanceM, test, file, n):
        #distanceM = distanceM.drop(distanceM.columns[(distanceM == 0).iloc[0]])
        h = 5       #bandwidth
        p = 2       #makes it euclidean distance
        summation = 0
        for row, val in distanceM.iterrows():
            min_vals = val.sort_values(ascending=True).head(n)
            numerator = 0
            denominator = 0
            for small in min_vals:
                first_term = (1/math.sqrt(2 * math.pi)) ** (file.shape[1] - 1)
                exp = math.exp((-1/2)*(small/h ** 2))
                w = first_term * exp
                true_index = min_vals.index[(min_vals == small)].astype(int)
                numerator += float(w) * float(file['class'][true_index[0]])
                denominator += w
            predicted = numerator/denominator
            actual = float(test['class'][row])
            summation += (predicted - actual) ** 2
        mean_sq_error = summation / test.shape[0]
        return([mean_sq_error, predicted])

    def condensed(self, file, prep):
        print('condensed')
        ET = .2
        z = pd.DataFrame()
        z = z.append(file.sample(n=1))
        file.drop(z.index)
        print(z)
        for row, rowitem in file.iterrows():
            min = 1000
            for xprime, item1 in z.iterrows():
                dist = prep.getDistance(item1, rowitem)
                if dist < min:
                    min = dist
                    closest = item1
            if self.problem == 'classification':
                if closest['class'] != rowitem['class']:

                    rowitem = rowitem.to_frame()
                    rowitem = rowitem.transpose()
                    z = z.append(rowitem)
            if self.problem == 'regression':
                print(closest['class'])
                print(rowitem['class'])
                if not (float(closest['class']) <= float(rowitem['class']) * (1 + ET) and float(closest['class']) >= float(rowitem['class']) * (1 - ET)):
                    rowitem = rowitem.to_frame()
                    rowitem = rowitem.transpose()
                    z = z.append(rowitem)
            print(z)
        z.reset_index(drop=True, inplace=True)
        print(z)
        return(z)

    def classEditedKNN(self, prep, file, tune, k):
        newLoss = -1  # dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                test = data
                train1 = file[:datarow]
                train2 = file[datarow + 1:]
                train = train1.append(train2)
                test = test.to_frame()
                test = test.transpose()
                train.reset_index(drop=True, inplace=True)
                test.reset_index(drop=True, inplace=True)
                distanceM = prep.getDistanceM(test, train)
                class_results = self.classification(distanceM, test, train, k)
                if class_results[1] == test['class'][0]:
                    file = file.drop([datarow])

                distanceM = prep.getDistanceM(tune, train)
                tune_result = self.classification(distanceM, tune, train, k)
                newLoss = tune_result[0]
                if datarow == 0:
                    loss = newLoss
            if newLoss != loss:
                print(file)
                return(file)
                break
    def regEditedKNN(self, prep, file, tune, n):
        print('regression edited')
        ET = .1
        newLoss = -1  # dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                test = data
                train1 = file[:datarow]
                train2 = file[datarow + 1:]
                train = train1.append(train2)
                test = test.to_frame()
                test = test.transpose()
                distanceM = prep.getDistanceM(test, train)
                class_results = self.regression(distanceM, test, file, n)
                if float(test['class'][datarow]) <= float(class_results[1]) * (1 + ET) and float(test['class'][datarow]) >= float(class_results[1]) * (1 - ET):
                    file = file.drop([datarow])
                distanceM = prep.getDistanceM(tune, file)
                tune_result = self.regression(distanceM, tune, file, n)
                newLoss = tune_result[0]
                if datarow == 0:
                    loss = newLoss
            if newLoss != loss:
                print(file)
                return(file)
                break



