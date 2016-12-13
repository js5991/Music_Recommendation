'''
Created on Dec 11, 2016

@author: Jingyi Su
'''
import pandas as pd
import sys

class featureSelection(object):
    '''
    classdocs
    '''
    def __init__(self, dataSource):
        '''
        Constructor
        '''
        self.df1=dataSource.ix[:,:"tags"]
        self.df2=dataSource.ix[:,"Bay Area":]
    
    
    def removeTagLessThanThreshold (self, threshold=5):
        self.df2=self.df2.applymap(lambda x: 1 if x>0 else 0)
        self.df2=self.df2[self.df2.columns[self.df2.sum()>threshold]]
    
    def returnDataSet(self):
        return pd.concat([self.df1,self.df2],axis=1)
        
    def printDataShape(self):
        '''
        this function prints the shape of data set
        '''
        print (self.returnDataSet().shape)

    def printDataHead(self):
        '''
        this function prints the head of data set
        '''
        print (self.returnDataSet().head())       
    
    def to_csvFile(self,directory):
        print ("Saving file at " +str(directory))
        self.returnDataSet().to_csv(directory)
    
if __name__=="__main__":
    try:
        data=pd.read_csv("../cleanedData.csv", index_col=0,sep=",", encoding="ISO-8859-1")
        selectedData=featureSelection(data)
        selectedData.removeTagLessThanThreshold()
        selectedData.to_csvFile("../featureSelectedData.csv")
        print ("Saved to File at ../featureSelectedData.csv")
    except EOFError:
        sys.exit()
    except IOError:
        print ("Cannot read/write file")
        sys.exit()
    except KeyboardInterrupt:
        print ("User directed exit")
        sys.exit()
    
        