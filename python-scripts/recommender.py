'''
This program takes a merged dataset (of books, users, ratings):
- Generates graphs for data exploration and analysis
- Implements a recommender system to predict book ratings
- Evaluates the reccommender system using graphs and metrics
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.metrics import mean_absolute_error
import seaborn as sns

##################
# Get input data #
##################

# Read in csv files
oldBooks = pd.read_csv('BX-Merged.csv')
newBooks = pd.read_csv('BX-NewBooksMerged.csv')

# Use a subsample of the datasets
oldBooksSubsample = oldBooks.sample(n=5000, random_state=1375531)
newBooksSubsample = newBooks.sample(n=1000, random_state=1375531)

# Drop rows with empty book titles
oldBooksSubsample.dropna(subset=['Book-Title'], inplace=True)
newBooksSubsample.dropna(subset=['Book-Title'], inplace=True)

# Only keep relevant columns
oldBooksSubsample = oldBooksSubsample[['User-ID', 'ISBN', 'Book-Title', 'Book-Rating']]
newBooksSubsample = newBooksSubsample[['User-ID', 'ISBN', 'Book-Title', 'Book-Rating']]

# Merge old and new books dataset
data = pd.concat([oldBooksSubsample, newBooksSubsample])

###########################
# Generate graphs for EDA #
###########################

def makeRatedPlotsData(df):
    '''
    Function to generate data for plots
    '''
    rated_dictionary = {}

    for index, row in df.iterrows():
        if row['User-ID'] in rated_dictionary.keys():
            rated_dictionary[row['User-ID']].append(row['ISBN'])
        elif row['User-ID'] not in rated_dictionary.keys():
            rated_dictionary[row['User-ID']] = [row['ISBN']]

    for value in rated_dictionary:
            rated_dictionary[value] = len(rated_dictionary[value])

    books_rated = []
    for item in rated_dictionary:
        books_rated.append(rated_dictionary[item])

    xvals = [0]
    yvals = [0]

    values, counts = np.unique(books_rated, return_counts=True)
    for value, count in zip(values, counts):
        xvals.append(value)
        yvals.append(count)
    
    return(xvals, yvals)

# Generate data for plots
subsample_x, subsample_y = makeRatedPlotsData(oldBooksSubsample)
full_x, full_y = makeRatedPlotsData(oldBooks)

# Plot Figure. 2
sns.scatterplot(x = full_x, y = full_y, s = 10)
plt.xlabel('Number of books rated')
plt.ylabel('Number of users who have rated x # of books')
plt.title("Distribution of number of books users rated in the entire BX-Books dataset")
plt.yticks((np.arange(min(full_y), max(full_y)+100, 5000)))
plt.savefig("full_dataset_rated_number", bbox_inches='tight')
plt.close()

# Plot Figure. 3
sns.scatterplot(x = subsample_x, y = subsample_y)
plt.xticks((np.arange(min(subsample_x), max(subsample_x) + 1, 1)))
plt.yticks((np.arange(min(subsample_y), max(subsample_y)+100, 200)))
plt.xlabel('Number of books rated')
plt.ylabel('Number of users who have rated x # of books')
plt.title("Distribution of number of books users rated in the data subsample used for analysis")
plt.savefig("sub_dataset_rated_number", bbox_inches='tight')
plt.close()

######################
# Recommender system #
######################

# Create TF-IDF matrix
vectorizer = TfidfVectorizer()
tfidfMatrix = vectorizer.fit_transform(data['Book-Title'])

# Convert TF-IDF matrix to dataframe
tfidfMatrixDf = pd.DataFrame(tfidfMatrix.toarray(), columns=vectorizer.get_feature_names_out())

## CREATE A SIMILARITY MATRIX BETWEEN ALL THE BOOKS ##############################################

# Option 1: Find cosine similarity for each pair
def makeCosSimilarityMatrix(tfidfMatrix):
    '''
    Given a tfidf matrix, finds and returns a cosine similarity matrix.
    '''
    similarityMatrix = cosine_similarity(tfidfMatrix, tfidfMatrix)
    return similarityMatrix

# Option 2: Find the Euclidian distance d for each pair, then find their similarities s = 1/(1+d)
def makeEuclidSimilarityMatrix(tfidfMatrix):
    '''
    Given a tfidf matrix, finds a euclidian distance matrix, then finds 
    and returns the similarity matrix.
    '''
    distanceMatrix = euclidean_distances(tfidfMatrix, tfidfMatrix)
    similarityMatrix = 1 / (1 + distanceMatrix)
    return similarityMatrix

# Choose either option 1 or 2. Put a pound symbol (#) before option that is not used:
# Option 1:
similarityMatrix = makeCosSimilarityMatrix(tfidfMatrix)
# Option 2:
#similarityMatrix = makeEuclidSimilarityMatrix(tfidfMatrix)

# Convert similarity matrix to a dataframe for manipulation
similarityDf = pd.DataFrame(similarityMatrix, index=data['ISBN'], columns=data['ISBN'])

## PREDICT USER RATINGS OF NEW BOOKS #############################################################

def weightedAverage(values, weights):
    '''
    Calculates weighted average of a list of values and its weights
    '''
    weightedSum = sum(value * weight for value, weight in zip(values, weights))
    sumOfWeights = sum(weights)
    return weightedSum / sumOfWeights

def predictRating(ISBN, UserID):
    '''
    Function that predicts a user's rating of a given book (from the new dataset) based on the average
    of their ratings of its top 3 similar books (from the old dataset)
    '''
    # Extract a single column dataframe which stores the similarity score between a book and all other books
    similarityScores = similarityDf[ISBN]
    if isinstance(similarityScores, pd.Series):
        similarityScores = similarityScores.to_frame()
    if len(similarityScores.columns) > 1:
        similarityScores = similarityDf[ISBN].iloc[:, [0]]

    # Only keep rows which are in the dataset of old books since we are not meant to compare with new books
    similarityScores = similarityScores[similarityScores[ISBN].index.isin(oldBooksSubsample['ISBN'])]
    
    # Sort by highest similarity scores
    similarityScores.sort_values(by=ISBN, ascending=False, inplace=True)

    # Find books that the user has rated in the past, ...
    pastRatings = oldBooks['ISBN'][oldBooks['User-ID'] == UserID]
    # ...and make sure the similarity scores only include books that users have rated in the past
    similarityScores = similarityScores[similarityScores[ISBN].index.isin(pastRatings)]

    # Find if there are any similarity scores of 1 (same book title)
    similarityScoreOf1 = similarityScores[similarityScores[ISBN] == 1]
    # If there is, then return the user's rating of that book
    if not similarityScoreOf1.empty:
        output = oldBooks[(oldBooks['ISBN'] == similarityScoreOf1.index[0])]
        output = output[(oldBooks['User-ID'] == UserID)]
        return output['Book-Rating'].iloc[0]

    # If there are no similar books that the user have read/rated, return Null
    if similarityScores.empty or similarityScores[ISBN][0] == 0:
        return np.nan
    # If there are similar books that the user have read/rated, then...
    else:
        # Get the ISBNs of the top 3 similar books
        top3 = similarityScores.index[:3].tolist()
        # ... and their similarity scores
        top3Scores = similarityScores[ISBN].head(3).tolist()
        # Find the ratings of those top 3 books similar books
        ratings = []
        for ISBN in top3:
            rating = oldBooks['Book-Rating'][(oldBooks['ISBN'] == ISBN)].mean()
            ratings.append(rating)
        # Find and return its weighted average
        weightedAverageRating = weightedAverage(ratings, top3Scores)
        return weightedAverageRating

# Predict the rating of every new book
newBooksSubsample['Predicted-Rating'] = newBooksSubsample.apply(lambda x: predictRating(x['ISBN'], x['User-ID']), axis=1)

# Drop rows with no predicted ratings
predictedVsActual = newBooksSubsample.dropna(subset=['Predicted-Rating'])

#####################################
# Evaluation of reccommender system #
#####################################

# Plot a scaterplot of actual vs predicted ratings: Figure. 1
predictedRatings = predictedVsActual['Predicted-Rating']
actualRatings = predictedVsActual['Book-Rating']
meanPredictedRatings = predictedVsActual.groupby('Book-Rating')['Predicted-Rating'].mean()
meanPredictedRating = meanPredictedRatings.mean()

plt.figure(figsize=(8, 6))
plt.scatter(actualRatings, predictedRatings, color='b', label='actual vs predicted ratings')
plt.scatter(meanPredictedRatings.index, meanPredictedRatings.values, color='r', label='mean predicted ratings for each actual rating')
plt.axhline(y=meanPredictedRating, color='g', linestyle='--', label='mean predicted rating')
plt.title('Actual vs. Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(range(0, 11))
plt.yticks(range(0, 11))
plt.savefig('Actual-Vs-Predicted-Ratings')

# Find mean absolute error of predicted vs actual ratings
mae = mean_absolute_error(actualRatings, predictedRatings)
print("MAE:", mae)


###########################################################################
# Sorting for highest predicted rating to provide insight to store owners #
###########################################################################

# Exclude books with predicted ratings lower than 8 and sort highest to lowest
pva = predictedVsActual
pva.drop(columns=["Book-Title", "Book-Rating"]).to_csv("BX-PredictedRatings")
pva = pva[~(pva["Predicted-Rating"] < 8)].sort_values(by=["Predicted-Rating", "Book-Rating"], ascending=False)

# calculate books with highest number of predicted ratings 8 and above
pva = pva.groupby("ISBN").size().reset_index(name="Count").sort_values("Count", ascending=False)

