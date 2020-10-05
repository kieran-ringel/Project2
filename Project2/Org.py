import pandas as pd

class Org():
    def __init__(self, file_name, header, class_loc):
        self.file_name = file_name
        self.header = header
        self.class_loc = class_loc

    """Kieran Ringel
    Takes all files and standardized them so they are formatted the same.
    This removes the header included on any file, and
    moves the class to the last column.
    Machines removes the ERP(estimated relative performance from the original article) column.
    Glass removes the index in the first column."""
    def open(self):
        file = open(self.file_name, 'r')        #opens file
        df = pd.DataFrame([line.strip('\n').split(',') for line in file.readlines()])   #splits file by lines and commas

        if self.header != [-1]:             #if user input that the data includes a header
            df = df.drop(self.header, axis=0)   #drop the header
            df = df.reset_index(drop=True)      #reset the axis
        if self.class_loc != -1:            #if the class is not in the last row
            end = df.shape[1] - 1           #moves class to last column
            col = df.pop(0)
            df.insert(end, end + 1, col)
            df.columns = df.columns - 1
        if self.file_name == "Data/glass.data": #if file is glass data
            df = df.drop(0, axis=1)             #remove column of index
            df = df.reset_index(drop=True)      #reset axis

        elif self.file_name == "Data/machine.data": #if file is machine data
            df = df.drop(9, axis=1)                 #remove column with ERP
            df = df.reset_index(drop=True)          #reset acis

        df.columns = [*df.columns[:-1], "class"]    #give column containing class label 'class'
        return(df)      #returns edited file
