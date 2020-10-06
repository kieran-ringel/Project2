from process_data import ProcessData
from Org import Org


def main():
    """ Kieran Ringel
    For each data set three lines are run in main.
    The first creates an instance of Org with the arguments being the data file name, an array of rows with header
    information to be removed, and the column location of the class so that all the classes can be put in the same column.
    The second line takes the instance of Org and calls the open method, returning the pandas dataframe of the file.
    The third line creates an instance of ProcessData, the arguments are the dataframe created in Org.open(), classification or
    regression, the type (none, edited, condensed, reducedmed, reducedmean), and an array of the columns with discrete values."""

    #print('glass')
    #glass = Org('Data/glass.data', [-1], -1)
    #df = glass.open()
    #ProcessData(df, 'classification', 'condensed', [-1])
    print('image')
    img = Org('Data/segmentation.data', [0, 1, 2, 3, 4], 0)
    df = img.open()
    ProcessData(df, 'classification', 'condensed', [-1])
    #print('vote')
    #vote = Org('Data/house-votes-84.data', [-1], 0)
    #df = vote.open()
    #ProcessData(df, 'classification', 'edited', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])
    #print('abalone')
    #abalone = Org('Data/abalone.data', [-1], -1)
    #df = abalone.open()
    #ProcessData(df, 'regression', 'edited', [0])
    #print('machine')
    #machine = Org('Data/machine.data', [-1], -1)
    #df = machine.open()
    #ProcessData(df, 'regression', 'condensed', [0, 1])
    #print('forest')
    #forest = Org('Data/forestfires.data', [0], -1)
    #df = forest.open()
    #ProcessData(df, 'regression', 'condensed', [2,3])

if __name__ == '__main__':
    main()
