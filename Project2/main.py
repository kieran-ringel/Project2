import pandas as pd
import random
class Reader():



    def __init__(self, file_name, problem, header, class_loc):
        file = open(file_name, 'r')
        df = pd.DataFrame([line.strip('\n').split(',') for line in
                          file.readlines()])
        if header != [-1]:
            df = df.drop(header, axis= 0)
            df = df.reset_index(drop=True)
        if class_loc != -1:
            end = df.shape[1] - 1
            print(end)
            col = df.pop(0)
            df.insert(end, end+1, col)
            df.columns = df.columns - 1

            print(df)
            print(df[end][0])

def main():
    #data file, type of knn being performed, any rows with extraneous information (headers) put as [-1] if
    #nothing needs to be removed, location of class column
    Reader('Data/glass.data', 'classification', [-1], -1)
    Reader('Data/segmentation.data', 'classification', [0, 1, 2, 3, 4], 0)
    Reader('Data/house-votes-84.data', 'classification', [-1], 0)
    Reader('Data/abalone.data', 'regression', [-1], -1)
    Reader('Data/machine.data', 'regression', [-1], -1)
    Reader('Data/forestfires.data', 'regression', [0], -1)

if __name__ == '__main__':
    main()
