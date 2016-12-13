'''
Created on Dec 4, 2016

@author: Jingyi Su
'''

from songs.model import song
import sys
import pandas as pd
import warnings


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    
    try:
        data=pd.read_csv("featureSelectedData.csv", index_col=0,sep=",", encoding="ISO-8859-1")
    except IOError:
        print ("Cannot read the file")
        sys.exit()
    
    print ("This is a concurrent song recommendation system from our music base. We are looking forward to hearing from you on whether you like the recommended song or not.\n"+"The system will be retrained after each song based on your preference. We strongly hope you stay in tune for the first few songs and your next favorite songs will be coming soon.")
    songRecommend=song(data,5)
    
    while True:
        try:
            print ("Based on your prior selections, we recommend you with this new song.")
            songRecommend.songRecommendation(1)
        except KeyboardInterrupt:
            sys.exit(1)
        except EOFError:
            print("EOFError")
            sys.exit(1)
    