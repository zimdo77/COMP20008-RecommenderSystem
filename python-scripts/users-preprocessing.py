'''
This program is used to preprocess data for the Users dataset.
Outputs the a csv of the preprocessed version of the dataset.
'''

import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from dataprep.clean import clean_country

MIN_AGE = 13
MAX_AGE = 100

###############################
# Functions used to test code #
###############################

def emptyRows(usersDf, column):
    '''
    Print the number of valid and NaN rows a given column
    '''
    print('No. of valid rows:', usersDf[column].notna().sum())
    print('No. of empty rows:', usersDf[column].isna().sum())

def countUniqueValues(usersDf, column):
    '''
    Count unique values in a given column
    '''
    print(usersDf[column].value_counts(dropna=False))

def plotColumnDistribution(usersDf, column, binEdges, title, fileName):
    '''
    Plot the a histogram to show the distribution of the column's data
    '''
    plt.hist(usersDf[column], bins=binEdges, edgecolor='black')
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.xticks(binEdges)
    plt.tight_layout()
    plt.savefig(fileName)

#####################################################
# Function to preprocess data for the users dataset #
#####################################################

def cleanUsers(csvInput, csvOutput):
    ## READ IN CSV
    users = pd.read_csv(csvInput)

    ## 'User-City' COLUMN PREPROCESSING ###############################

    # Standardise text
    users['User-City'] = users['User-City'].str.strip()
    users['User-City'] = users['User-City'].str.lower()
    users['User-City'] = users['User-City'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))

    # Remove instances of missing cities (since a very small percentage of values are missing ~0.13%)
    users = users[users['User-City'] != '']

    ## 'User-State' COLUMN PREPROCESSING ##############################
    # Standardise text

    users['User-State'] = users['User-State'].str.strip()
    users['User-State'] = users['User-State'].str.lower()
    users['User-State'] = users['User-State'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))

    # Remove instances of missing states (since a very small percentage of values are missing ~1.4%)
    users = users[users['User-State'] != '']

    ## 'User-Country' COLUMN PREPROCESSING ############################
    # Standardise text
    
    users['User-Country'] = users['User-Country'].str.strip()
    users['User-Country'] = users['User-Country'].str.lower()
    users['User-Country'] = users['User-Country'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))

    # Use the 'dataprep.clean' library to standardise different aliases for country names into a common one
    users = clean_country(users, 'User-Country', inplace=True)
    users.rename(columns={'User-Country_clean': 'User-Country'}, inplace=True)
    users['User-Country'] = users['User-Country'].str.lower()
    # Remove instances of missing countries (since a very small percentage of values are missing ~0.56%)
    users.dropna(subset=['User-Country'], inplace=True)

    ## 'User-Age' COLUMN PREPROCESSING ################################

    # Format error: remove characters which are not digits
    users['User-Age'] = users['User-Age'].apply(lambda x: re.sub(r'\D', '', str(x)))
    users['User-Age'] = users['User-Age'].apply(lambda x: int(x) if x else None)
    # Range error: remove values (make them NaN) if they are outside of the age range
    users.loc[(users['User-Age'] < MIN_AGE) | (users['User-Age'] > MAX_AGE), 'User-Age'] = np.nan

    # ~40% of data of user ages are missing - too high to just simply discard instaces
    # So sample n missing values (nMissing) from the given ages within the column, ...
    nMissing = users['User-Age'].isna().sum()
    np.random.seed(1375531)
    givenAges = users['User-Age'].dropna()
    sampledAges = givenAges.sample(nMissing, replace=True).tolist()
    # ... then impute them into the NaN entries
    users.loc[users['User-Age'].isna(), 'User-Age'] = sampledAges
    users['User-Age'] = users['User-Age'].astype(int)

    ## CONVERT DATAFRAME TO CSV #######################################
    users.to_csv(csvOutput, index=False)

#######################
# Preprocess datasets #
#######################

# ...for old books
cleanUsers('BX-Users.csv', 'BX-Users-CLEANED.csv')

# ...for new books
cleanUsers('BX-NewBooksUsers.csv', 'BX-NewBooksUsers-CLEANED.csv')
