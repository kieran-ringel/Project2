from process_data import ProcessData
from Org import Org


def main():
    #data file, type of knn being performed, any rows with extraneous information (headers) put as [-1] if
    #nothing needs to be removed, location of class column, location of discrete data
    #glass = Org('Data/glass.data', [-1], -1)
    #df = glass.open()
    #ProcessData(df, 'classification', [-1])
    #img = Org('Data/segmentation.data', [0, 1, 2, 3, 4], 0)
    #df = img.open()
    #ProcessData(df, 'classification', [-1])
    #vote = Org('Data/house-votes-84.data', [-1], 0)
    #df = vote.open()
    #ProcessData(df, 'classification', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    abalone = Org('Data/abalone.data', [-1], -1)
    df = abalone.open()
    ProcessData(df, 'regression', [0])
    #machine = Org('Data/machine.data', [-1], -1) #REMOVE ERP!!!!
    #df = machine.open()
    #ProcessData(df, 'regression', [0, 1])
    #forest = Org('Data/forestfires.data', [0], -1)
    #df = forest.open()
    #ProcessData(df, 'regression', [2,3])

if __name__ == '__main__':
    main()
