# COMP20008 Assignment 2 - Recommender System (Grade: 30.7/34)

- Code for Assignment 2 for UniMelb subject: [Elements of Data Processing (COMP20008)](https://handbook.unimelb.edu.au/2024/subjects/comp20008) - Semester 2, 2024.
- Assignment specification, final report, and final slides attached.
- Relevant code inside `python-scripts` directory.
- Code written by myself (Quoc Khang Do), Alexander Broadley, and Miles Sandles.

## Main tasks
- Predicted user ratings of books using a content based recommender system - by using TF-IDF vectorisation on book titles and finding their cosine similarity.
- Conducted extensive data preprocessing - cleaning and standardizing datasets.

## README: `python-scripts`

### DATA PREPROCESSING

`users-preprocessing.py`
- Program that preprocesses data for the 'BX-Users.csv' and ‘BX-NewBooksUsers.csv’ datasets.
- Outputs the preprocessed version of the datasets: 'BX-Users-CLEANED.csv' and 'BX-NewBooksUsers-CLEANED.csv'
- Libraries used: pandas, numpy, regex, matplotlib.pyplot, clean_country (from dataprep.clean)

`books-preprocessing.py`
- Program that preprocesses data for the ‘BX-Books.csv’ and ‘BX-NewBooks.csv’
- Outputs the preprocessed version of the dataset: 'BX-Books-CLEANED.csv' and 'BX-NewBooks-CLEANED.csv'
- Libraries used: pandas, numpy, regex, requests, nltk

`ratings-preprocessing.py`
- Program that preprocesses data for the ‘BX-Ratings.csv’ and ‘BX-NewBooksRatings.csv’
- Outputs the preprocessed version of the dataset: 'BX-Ratings-CLEANED.csv' and 'BX-NewBooksRatings-CLEANED.csv'.
- Libraries used: pandas, regex.

`avg_rating_v_num_ratings.py`
- Program that takes data from ‘BX-Books.csv’, ‘BX-NewBooks.csv’, ‘BX-Ratings.csv’ and ‘BX-NewBooksRatings.csv’ datasets and generates graphs.
- Outputs graphs “New Ratings v No Ratings.png” and “Rating v No Ratings.png”.
- Libraries used: pandas, matplotlib.

NOTE: These programs must be run first as the output files are used in the other scripts

### RECOMMENDATION SYSTEM

`merge.py`
- Merges the cleaned users, books and ratings datasets (both for the Books dataset and NewBooks dataset).
- Outputs a merged version of the old books dataset ‘BX-Merged.csv’ and a merged version of the new books dataset ‘BX-NewBooksMerged.csv’.
- Libraries used: pandas.

`recommender.py`
- Takes a merged dataset (of books, users, ratings).
- Generates graphs for data exploration and analysis
- Implements a recommender system to predict book ratings.
- Evaluates the recommender system using graphs and metrics.
- Libraries used: pandas, numpy, matplotlib.pyplot, sklearn, seaborn.
