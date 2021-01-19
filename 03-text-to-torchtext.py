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

