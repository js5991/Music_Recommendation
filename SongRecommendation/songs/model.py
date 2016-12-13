'''
Created on Dec 4, 2016

@author: Jingyi Su
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

import sys

class song(object):
    '''
    classdocs
    '''
    def __init__(self, data, numSongs=5):
        '''
        Constructor
        '''
        self.Ytrain=[]
        self.data=data
        self.textCol=['artist_name','release','title_x','tags']
        self.data.sort('song_hotttnesss',ascending=False, inplace=True)
        self.Xtrain=self.data[:numSongs]
        self.data=self.data[numSongs:]
        self.model=None 
        
        i=0
        while (i<numSongs) or len(np.unique(self.Ytrain))<2:
            if i>=numSongs:
                self.Xtrain=self.Xtrain.append(self.data[:1])
                self.data=self.data[1:]
            self.targetValue(self.Xtrain.iloc[i])
            i=i+1

        #self.lrModel()
        self.decisionTreeModel()
        

    def songRecommendation(self, numSongs=5):
        self.data.sort('Ypredict',ascending=False, inplace=True)
        #print(self.data['Ypredict'])
        x=self.data[:numSongs].drop('Ypredict',axis=1)
        self.data=self.data[numSongs:]
        self.Xtrain=self.Xtrain.append(x)
                
        for i in range(numSongs):
            self.targetValue(x.iloc[i])   
        
        self.data=self.data.drop('Ypredict',axis=1)
        #self.lrModel()    
        self.decisionTreeModel()
    
    def lrModel(self):
        lrmodel=LogisticRegression()
        lrmodel.fit(self.Xtrain.drop(self.textCol,axis=1),self.Ytrain)
        self.data['Ypredict']=lrmodel.predict_proba(self.data.drop(self.textCol,axis=1))[:,1]
    
    def decisionTreeModel(self):
        self.model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 1, max_depth = 3)
        self.model = self.model.fit(self.Xtrain.drop(self.textCol,axis=1),self.Ytrain)
        self.data['Ypredict']=self.model.predict_proba(self.data.drop(self.textCol,axis=1))[:,1]
        
    def featureImportance(self):
        featureImportance=self.model.feature_importances_
        columns=self.Xtrain.drop(self.textCol,axis=1).columns.values
        stack=np.column_stack([columns, featureImportance])
        topRank=stack[stack[:,1].argsort()][-10:]
        topRankNonZero=topRank[topRank[:,1].nonzero()]
        
        if len(topRankNonZero)==0:
            print ("Sorry your song preference is too random!")
        else:
            fig, ax = plt.subplots()
            width = 0.35
            #ax.bar(train.drop(lab,1).columns.values, clf.feature_importances_, width, color='r')
            ax.bar(np.arange(len(topRankNonZero)), topRankNonZero[:,1], width, color = 'b')
            ax.set_xticks(np.arange(len(topRankNonZero[:,1])))
            ax.set_xticklabels(topRankNonZero[:,0], rotation = 25)
            plt.title('Features contribute most to your music preference')
            ax.set_ylabel('Normalized Feature Importance')
            plt.show()
        
    
    def targetValue(self, x):
        while True:
            try:
                
                inputstr=input("The song we recommended for you is Title: "+str(x["title_x"])+", Artist: "+str(x["artist_name"])+"\n Please enter 1 if you like it, otherwise enter 0.")
                if (inputstr=="quit"):
                    print("Thank you for using our song recommendation system. Here is the list of important features contributing to your sound preference.")
                    self.featureImportance()
                    sys.exit(1)
                y=int(inputstr)
                if (y!=0 and y!=1):
                    raise ValueError ("Need a value of 1 or 0") 
                else:
                    self.Ytrain.append(y)
                    print (" \n")
                break
            except ValueError:
                print("Error")
            except KeyboardInterrupt:
                sys.exit(1)
            except SystemExit:
                sys.exit()
            except:
                print ("There is an error. "+"The song we recommended for you is "+str(x["title_x"])+"\n Please enter 1 if you like it, otherwise enter 0")
       



        