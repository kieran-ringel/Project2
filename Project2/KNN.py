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
        """Kieran Ringel
        Removes a tuning set
        Tunes for k
        Figures out what type of classification/ regression is occuring"""
        #get every tenth as tuning
        tune = file.iloc[::10]
        tune.reset_index(drop=True, inplace=True)
        #removes all every tenth from the file
        file.drop(file.iloc[::10].index)
        file.reset_index(drop=True, inplace=True)

        k = self.tuningk(tune, file, prep)
        print('k is', k)

        file.sort_values(by="class", inplace=True)  #sort files by class to get every 10th later
        file.reset_index(drop=True, inplace=True)   #resets index

        if self.problem == "classification" and self.type == 'none':
            fold = prep.stratification(file)
            self.tenfold(fold, prep, k)
        if self.problem == 'regression' and self.type == 'none':
            #10 fold cross validation by getting every 10th data point of the sorted data
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
            # 10 fold cross validation by getting every 10th data point of the sorted data
            new_file = self.regEditedKNN(prep, file, tune, k)
            fold = [None] * 10
            for cv in range(10):
                to_test = new_file.iloc[cv::10]
                to_test.reset_index(drop=True, inplace=True)
                fold[cv] = to_test
            self.tenfold(fold, prep, k)
        if self.type == 'condensed':
            new_file = self.condensed(file, prep)
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
        """Kieran Ringel
        Tests for five values of k and returns the k that results in the smallest
        loss result."""
        k = pd.Series([3, 4, 5, 6, 7])  #centered around 5, since 5 is what was presented in the .name files
        loss = pd.Series(5)
        for index in range(len(k)):
            distanceM = prep.getDistanceM(tuning, file) #creates distance matrix
            if self.problem == 'classification':
                results = self.classification(distanceM, tuning, file, k[index])    #does classification on tuning set usnig k
                loss[index] = results[0]    #returns loss
            if self.problem == 'regression':
                results = self.regression(distanceM, tuning, file, k[index])    #does regression on tuning using k
                loss[index] = results[0]    #returns loss
        minindex = loss.idxmin()    #gets index of smallest loss
        return(k[minindex])         #returns k associated with that index

    def tenfold(self, fold, prep, neighbors):
        """
        Kieran Ringel
        Iterates through 10 folds of ten fold cross validation
        Determines to perform classification or validatoin
        THis method is called for both complete and reduced data sets
        """
        tot_loss = 0
        for cv in range(10):  # get test and train datasets
            print("Run", cv+1)  #prints run number
            test = fold[cv]     #gets test set
            train_list = fold[:cv] + fold[cv+1:]    #gets training set, everything besided the test set
            train = pd.concat(train_list)           #concatanates the 2 parts of the test set
            train.reset_index(drop=True, inplace=True)  #resets index on both
            test.reset_index(drop=True, inplace=True)

            if self.problem == "classification":
                distanceM = prep.getDistanceM(test, train)  #gets distance matrix
                results = self.classification(distanceM, test, train, neighbors)    #classified
                print('0/1 loss', results[0])   #prints loss
                tot_loss += results[0]      #add to total loss, used for average loss across all sets
            if self.problem == 'regression':
                distanceM = prep.getDistanceM(test, train) #gets distance matrix
                results = self.regression(distanceM, test, self.file, neighbors) #regressed
                print('Squared Loss', results[0]) #prints loss
                tot_loss += results[0]  #add to total loss, used for average loss across all sets
        avg_loss = tot_loss/10 #calculated average loss
        print(avg_loss)

    def classification(self, distanceM, test, train, k):
        """Kieran Ringel
        Classified every data point in a test set using a pluality vote
        Calculated 0/1 loss based on if it is classifing correctly"""
        m = 0       #reset m summation to 0 for next test fold
        voted_class = 'none'       #creates voted_class outside of the for loop so it can be reference for edited
        for row, val in distanceM.iterrows():
            knn_classes = pd.Series()
            min_vals = val.sort_values(ascending=True).head(k)      #gets value of the minimum values for each test point
            for min_index, min in min_vals.items():
                col = distanceM.columns[(distanceM == min).iloc[row]].astype(int) #returns training set index for minimum value for each test point
                knn_classes.at[min_index] = train['class'][col[0]]

            voted_class = knn_classes.mode()[0]     #get plurality vote
            if(test['class'][row] == voted_class): #sees if it has been classified correctly
                m += 0
            else:
                m += 1      #adds 1 to m if misclassified
        loss = m/test.shape[0]  #calculated 0/1 loss
        return ([loss, voted_class])    #returns loss and most recent voted_class(used for edited and condense)

    def regression(self, distanceM, test, file, n):
        """Kieran Ringel
        Regressed a data point using a running mean smoother and a kernel smoother
        Returns mean squared loss and the most recently regressed point
        """
        h = 5       #bandwidth tuned to 5
        summation = 0
        for row, val in distanceM.iterrows():
            min_vals = val.sort_values(ascending=True).head(n)  #get k nearest neighbors
            numerator = 0
            denominator = 0
            for small in min_vals:
                first_term = (1/math.sqrt(2 * math.pi)) ** (file.shape[1] - 1)  #first term in kernel smoother
                exp = math.exp((-1/2)*(small/h ** 2))   #exp term in kernel smoother
                w = first_term * exp    #kernel smoother
                true_index = min_vals.index[(min_vals == small)].astype(int)    #index to reference class of neighbors
                numerator += float(w) * float(file['class'][true_index[0]])     #summation of kernel times neighbors class
                denominator += w    #summation of kernel
            predicted = numerator/denominator   #predicted regression
            actual = float(test['class'][row])  #ground truth
            summation += (predicted - actual) ** 2  # used to calculated squared loss
        mean_sq_error = summation / test.shape[0]   #calculated mean squared loss
        return([mean_sq_error, predicted])  #return loss and most recent regression(used for edited and condensed)

    def condensed(self, file, prep):
        """Kieran Ringel
        Adds random data point to z and then iterates through the data
        If the clostest point in z, to the data point, has a different classification than
        the data point, add it to z
        Continue until z does not change"""
        print('condensed')
        ET = .2     #error threshold tuned to 5
        z = pd.DataFrame()
        z = z.append(file.sample(n=1))  #select random first data point
        file.drop(z.index)      #removes from file so it isnt looked at twice
        for row, rowitem in file.iterrows():    #iterate through all datapoints in file
            min = 100000    #dummy value
            for xprime, item1 in z.iterrows():  #iterates through all points in z
                dist = prep.getDistance(item1, rowitem) #gets disatance between value in file and value in z
                if dist < min:  #if this is the new closes value to the data in file
                    min = dist  #min distance is reset
                    closest = item1 #value is set to closest
            if self.problem == 'classification':
                if closest['class'] != rowitem['class']:    #if they do not have to same classification
                    rowitem = rowitem.to_frame()    #change from Series to df
                    rowitem = rowitem.transpose()
                    z = z.append(rowitem)       #append to z
            if self.problem == 'regression':
                #if they are not within the ET of eachother
                if not (float(closest['class']) <= float(rowitem['class']) * (1 + ET) and float(closest['class']) >= float(rowitem['class']) * (1 - ET)):
                    rowitem = rowitem.to_frame()    #change series to df
                    rowitem = rowitem.transpose()
                    z = z.append(rowitem)       #append to z
        z.reset_index(drop=True, inplace=True)  #reset index on z
        return(z)   #return reduced data set

    def classEditedKNN(self, prep, file, tune, k):
        """Kieran Ringel
        Checks each data point, if it can be correctly classified using the result of the data, it is removed
        If it's removal does not degrade the performance using tuning it stays removed"""
        newLoss = -1  # dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                test = data     #test is one data point
                train1 = file[:datarow]
                train2 = file[datarow + 1:]
                train = train1.append(train2)   #train is all data points around that
                test = test.to_frame()
                test = test.transpose()
                train.reset_index(drop=True, inplace=True)
                test.reset_index(drop=True, inplace=True)
                distanceM = prep.getDistanceM(test, train)  #get distance
                class_results = self.classification(distanceM, test, train, k)  #return class for the one data point
                if class_results[1] == test['class'][0]:    #if it is correctly classified
                    file = file.drop([datarow])         #remove it from the file

                distanceM = prep.getDistanceM(tune, train)  #get distance to test tune
                tune_result = self.classification(distanceM, tune, train, k)    #get loss from tune
                newLoss = tune_result[0]
                if datarow == 0:    #if first run though, this is the base loss
                    loss = newLoss
            if newLoss != loss:     #if the loss changed
                return(file)        #cannot remove more returns file
                break
    def regEditedKNN(self, prep, file, tune, n):
        """Kieran Ringel
        Similar to that of classification, but must use regression to test and check performance.
        Checks each data point, if it can be correctly classified using the result of the data, it is removed
        If it's removal does not degrade the performance using tuning it stays removed"""
        print('regression edited')
        ET = .1
        newLoss = -1  # dummy variable
        while True:
            loss = newLoss
            for datarow, data in file.iterrows():
                test = data     #test is one data point
                train1 = file[:datarow]     #train is all data around taht
                train2 = file[datarow + 1:]
                train = train1.append(train2)
                test = test.to_frame()
                test = test.transpose() #test is made horizontal dataframe
                distanceM = prep.getDistanceM(test, train)
                class_results = self.regression(distanceM, test, file, n) #gets results of regression on one point
                #if it is within the ET, the data point is removed
                if float(test['class'][datarow]) <= float(class_results[1]) * (1 + ET) and float(test['class'][datarow]) >= float(class_results[1]) * (1 - ET):
                    file = file.drop([datarow])
                distanceM = prep.getDistanceM(tune, file)   #distance is calculated for tuning
                tune_result = self.regression(distanceM, tune, file, n)
                newLoss = tune_result[0]
                if datarow == 0:
                    loss = newLoss
            if newLoss != loss: #if loss on tuning changed, return file
                return(file)
                break



