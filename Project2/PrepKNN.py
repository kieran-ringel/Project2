import pandas as pd
class PrepKNN:
    def __init__(self, file, discrete, VDMdict, problem):
        self.file = file
        self.discrete = discrete
        self.VDMdict = VDMdict
        self.problem = problem

    def stratification(self, file):
        print('strat')
        fold_size = file.shape[0] / 10
        count = file['class'].value_counts(normalize=True)
        fold = [None] * 10
        for cv in range(10):
            one_fold = pd.DataFrame()
            for classes, dataframe in file.groupby(by='class'):
                dataframe.reset_index(drop=True, inplace=True)
                proportion = count[classes]
                ex_per_fold = fold_size * proportion

                beginning = int(cv * ex_per_fold)
                end = int((cv + 1) * ex_per_fold)
                one_fold = one_fold.append(dataframe.loc[beginning:end, :])
            fold[cv] = one_fold
        return(fold)

    def getDistanceM(self, test, train, p):
        print('getting D')
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
                distance = tot ** (1 / p)
                distanceM.at[testrow, trainrow] = distance
        return(distanceM)
