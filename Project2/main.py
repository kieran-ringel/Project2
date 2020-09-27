import pandas as pd
from KNN import KNN
from Org import Org


def main():
    #data file, type of knn being performed, any rows with extraneous information (headers) put as [-1] if
    #nothing needs to be removed, location of class column, location of discrete data
    #glass = Reader('Data/glass.data', 'classification', [-1], -1, [-1])
    #glass.open()
    #img = Reader('Data/segmentation.data', 'classification', [0, 1, 2, 3, 4], 0, [-1])
    #img.open()
    #vote = Reader('Data/house-votes-84.data', 'classification', [-1], 0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    #vote.open()
    abalone = Org('Data/abalone.data', [-1], -1)
    df = abalone.open()
    KNN(df, 'regression', [0])
    #machine = Reader('Data/machine.data', 'regression', [-1], -1, [0, 1])
    #machine.open()
    #forest = Reader('Data/forestfires.data', 'regression', [0], -1, [0, 1])
    #forest.open()

if __name__ == '__main__':
    main()
