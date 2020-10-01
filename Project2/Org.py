import pandas as pd

class Org():
    def __init__(self, file_name, header, class_loc):
        self.file_name = file_name
        self.header = header
        self.class_loc = class_loc
        self.df = self.open()

    def open(self):                         #takes all data files and standardizes them so they all class in the same column
                                            #removes headers from any dataset that came with that
        file = open(self.file_name, 'r')
        df = pd.DataFrame([line.strip('\n').split(',') for line in file.readlines()])

        if self.header != [-1]:
            df = df.drop(self.header, axis=0)
            df = df.reset_index(drop=True)
        if self.class_loc != -1:
            end = df.shape[1] - 1
            col = df.pop(0)
            df.insert(end, end + 1, col)
            df.columns = df.columns - 1
        if self.file_name == "Data/glass.data":
            df = df.drop(0, axis=1)
            df = df.reset_index(drop=True)

        elif self.file_name == "Data/machine.data":
            df = df.drop(9, axis=1)
            df = df.reset_index(drop=True)

        df.columns = [*df.columns[:-1], "class"]
        return(df)
