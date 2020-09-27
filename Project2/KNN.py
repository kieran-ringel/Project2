import pandas as pd
class KNN:
    def __init__(self, file, problem, discrete):
        self.file = file
        self.problem = problem
        self.discrete = discrete
        self.check_discrete(discrete)

    def check_discrete(self, discrete):
        if discrete != [-1]:
            self.VDMdiscrete(self.file, self.discrete)

    def VDMdiscrete(self, file, discrete):
        VDMdictionary = {}
        for features in range(len(discrete)):   #discrete[feature] will give index of feature that is discrete
            column = discrete[features]
            features = []
            classes = []
            for row in range(file.shape[0]):
                if file[column][row] not in features:
                    features.append(file[column][row])
                if file["class"][row] not in classes:
                    classes.append(file["class"][row])
            VDM = pd.DataFrame(index = features, columns = features)
            for feat in range(len(features)):
                for feat2 in range(len(features)):
                    running_sum = 0
                    for clas in range(len(classes)):
                        ci = self.ci(column, features[feat])
                        cia = self.cia(column, classes[clas], features[feat])
                        cj = self.ci(column, features[feat2])
                        cja = self.cia(column, classes[clas], features[feat2])
                        running_sum += abs((cia/ci) - (cja/cj))
                    VDM.at[features[feat], features[feat2]] = running_sum
            VDMdictionary[column] = []
            VDMdictionary[column].append(VDM)

            print(VDMdictionary)

    def ci(self, column, feature):
        count = self.file[column].value_counts()[feature]
        return count

    def cia(self, column, clas, feature):
        class_df = self.file[self.file['class'] == clas]
        count = class_df[column].value_counts()
        if feature not in count:
            return 0
        else:
            count = count[feature]
        return count



