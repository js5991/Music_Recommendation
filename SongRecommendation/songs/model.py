'''
Created on Dec 4, 2016

@author: Jingyi Su
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import sys
from . import myexception


class Song(object):
    """
    This class returns an object of the song class.
    """
    def __init__(self, data, num_songs=5):
        """
        :param data: the featureSelection data input
        :param num_songs: default number of recommended song is 5.
        """
        self.Ytrain = []
        self.data = data

        # TextCol are the columns that have text data.
        self.textCol = ['artist_name', 'release', 'title_x', 'tags']

        # We first sort the data with song_hotttness, so that the users of the recommendation program will
        # know the first several songs we recommend.
        self.data.sort('song_hotttnesss', ascending=False, inplace=True)

        # Xtrain data are the first several songs in the data in descending order
        self.Xtrain = self.data[:num_songs]

        # The data that have other songs
        self.data = self.data[num_songs:]
        self.model = None
        # Wrote a while loop that break when number of songs recommend is larger than num_songs,
        # or the users like all the songs or dislike all the songs.
        # (Since only with 0,1 would the decision tree be able to classify)
        
        i = 0
        while (i < num_songs) or len(np.unique(self.Ytrain)) < 2:
            # Xtrain append one more recommended song from data.
            if i >= num_songs:
                self.Xtrain = self.Xtrain.append(self.data[:1])
                self.data = self.data[1:]
            # The targetValue of the song would be from the user input: which is 0 or 1.
            self.targetValue(self.Xtrain.iloc[i])
            i=i+1

        # Using Xtrain and Ytrain (which is the targetValue from user) in decisionTreeModel
        self.decisionTreeModel()

    def songRecommendation(self, num_songs=5):
        """
        Function that recommend songs to user
        :param num_songs: default number of songs is 5.
        :return: targetValue
        """
        # Using the decision tree model, sort the predict probability in descending order
        self.data.sort('Ypredict',ascending=False, inplace=True)

        # Since we only need the Xtrian, drop Ypredict first.
        x = self.data[:num_songs].drop('Ypredict',axis=1)

        # data are the songs that have not been recommended before.
        self.data = self.data[num_songs:]

        # Append recommend songs x to Xtrain.
        self.Xtrain = self.Xtrain.append(x)

        # For each song in number of songs, find targetValue of the song from user.
        for i in range(num_songs):
            self.targetValue(x.iloc[i])

        # Drop Ypredict and run a decision Tree model.
        self.data=self.data.drop('Ypredict',axis=1)
        self.decisionTreeModel()
    
    def lrModel(self):
        """
        Logistic regression model that takes in Xtrain and return a column 'Ypredict' to be predicted value.
        """
        lrmodel = LogisticRegression()
        lrmodel.fit(self.Xtrain.drop(self.textCol,axis=1),self.Ytrain)
        self.data['Ypredict'] = lrmodel.predict_proba(self.data.drop(self.textCol, axis=1))[:,1]
    
    def decisionTreeModel(self):
        """
        Decision Tree model that takes in Xtrain and return a column 'Ypredict' to be predicted value.
        """
        self.model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf = 1, max_depth = 3)
        self.model = self.model.fit(self.Xtrain.drop(self.textCol,axis=1),self.Ytrain)
        self.data['Ypredict'] = self.model.predict_proba(self.data.drop(self.textCol,axis=1))[:,1]
        
    def featureImportance(self, path=None):
        """
        Function that return a plot using decision tree's feature importances:
        show the bar plot of the top ranked features in terms of importances.
        """
        featureImportance = self.model.feature_importances_
        columns = self.Xtrain.drop(self.textCol,axis=1).columns.values
        stack = np.column_stack([columns, featureImportance])
        topRank = stack[stack[:,1].argsort()][-10:]
        topRankNonZero = topRank[topRank[:,1].nonzero()]
        
        if len(topRankNonZero) == 0:
            print("Sorry your song preference is too random!")
        else:
            fig, ax = plt.subplots()
            width = 0.35
            ax.bar(np.arange(len(topRankNonZero)), topRankNonZero[:,1], width, color='b')
            ax.set_xticks(np.arange(len(topRankNonZero[:,1])))
            ax.set_xticklabels(topRankNonZero[:, 0], rotation = 25)
            plt.title('Features contribute most to your music preference')
            ax.set_ylabel('Normalized Feature Importance')
            
            if path!=None:
                plt.savefig(path)
            else:
                plt.show()
    
    def targetValue(self, x):
        """
        :param x: Xtrain that have all songs data.
        :return: User input of 0 or 1: dislike or like for certain song.
        """
        while True:
            try:
                
                inputstr=input("The song we recommended for you is Title: "+str(x["title_x"])+", Artist: "+str(x["artist_name"])+"\n Please enter 1 if you like it, otherwise enter 0.")
                if (inputstr == "quit"):
                    print("Thank you for using our song recommendation system. Here is the list of important features contributing to your sound preference.")
                    if len(self.Ytrain)>1 and self.model!=None:
                        self.featureImportance()
                        path=input("Where do you want to save your customized important feature list?")
                        self.featureImportance(path)
                        print("The customized important feature is saved.")
                    sys.exit(1)
                y=int(inputstr)
                if (y!=0 and y!=1):
                    raise myexception.InvalidInputError()
                else:
                    self.Ytrain.append(y)
                    print(" \n")
                break
            except myexception.InvalidInputError as e:
                print(e)
            except KeyboardInterrupt:
                sys.exit(1)
            except SystemExit:
                sys.exit()