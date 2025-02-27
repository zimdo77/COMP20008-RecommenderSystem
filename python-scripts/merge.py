'''
This program merges the users, books, and ratings datasets.
Outputs a csv of the merged dataset
'''

import pandas as pd

def mergeDatasets(booksCSV, usersCSV, ratingsCSV, outputCSV) :
    '''
    Merges users, books, and ratings datasets.
    Outputs a csv of the merged dataset.
    '''
    # Read in csv files
    books = pd.read_csv(booksCSV)
    users = pd.read_csv(usersCSV)
    ratings = pd.read_csv(ratingsCSV)

    # Merge ratings with books
    merged_data = pd.merge(ratings, books, on='ISBN', how='inner')

    # Merge merged_data with users
    merged_data = pd.merge(merged_data, users, on='User-ID', how='inner')

    # Drop unnecessary columns
    if 'Unnamed: 0.1' in merged_data:
        merged_data.drop(columns=['Unnamed: 0.1'], inplace=True)
    if 'index' in merged_data:
        merged_data.drop(columns=['index'], inplace=True)

    # Convert dataframe to csv
    merged_data.to_csv(outputCSV, index=False)

# Merge datasets for the old books
mergeDatasets('BX-Books-CLEANED.csv', 'BX-Users-CLEANED.csv', 'BX-Ratings-CLEANED.csv', 'BX-Merged.csv')

# Merge datasets for the new books
mergeDatasets('BX-NewBooks-CLEANED.csv', 'BX-NewBooksUsers-CLEANED.csv', 'BX-NewBooksRatings-CLEANED.csv', 'BX-NewBooksMerged.csv')
