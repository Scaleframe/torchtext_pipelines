"""
Our Goal here is to take raw text data and get it into a format we can use in torchtext. 

As we saw before, torchtext wants to read in files that are in json, csv, or tsv formats. So let's get there first. 
"""

import spacy # 
import pandas as pd # we'll use this for PRE pre processing
from torchtext.data import Field, BucketIterator, TabularDataset # use this for pre processing. 
from sklearn.model_selection import train_test_split # use for train test splitting our dataframe.  

# Data files can be found at: https://nlp.stanford.edu/projects/nmt/
    # WMT'14 English-German data [Medium], 
        # train.en and train.de. 


# First we need to get our data read in from our text files. 

# Read in file line by line (first 100 lines) to prevent memory overflow:
with open('mydata/WMT_train.en') as myfile:
    english_text = [next(myfile) for line in range(100)]
# returns list of strings, english sentences


with open('mydata/WMT_train.de') as myfile:
    german_text = [next(myfile) for x in range(100)]
# returns: list of strings, german sentences.

'''
We can use pandas dataframe to get a dictionary of text into a useable format for training.
'''

# First we need to get our text data into a dictionary. 
raw_data = {"English": [line for line in english_text[1:10000]],
            "German": [line for line in german_text[1:10000]]}
# returns: python dictionary


# From dictionary we can make DataFrame object:
df = pd.DataFrame(raw_data, columns=['English', 'German'])

# With dataframe we can use sklearn's train/test/split:
train, test = train_test_split(df, test_size=0.2)

# Now we can write our train / test split to json:
train.to_json('train.json', orient='records', lines=True)
train.to_json('train.json', orient='records', lines=True)
# returns: json file with structure: 
#               {"English": english sentence, 
#                "German": german_sentence}

# Or csv format: 
train.to_csv('train.csv')
train.to_csv('train.csv')
# returns: json file with structure: 
#               {"English": english sentence, 
#                "German": german_sentence}
