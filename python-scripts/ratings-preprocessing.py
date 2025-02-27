import pandas as pd
import regex as re

bRatings = pd.read_csv("BX-Ratings.csv")
nbRatings = pd.read_csv("BX-NewBooksRatings.csv")

#checks if the values in a dataframe column can be converted to a number
def checkInterger(df, column):
    for i in range(0, len(df)):
        try:
            number = int(df[column][i])
        except ValueError:
            print(df[column][i], "cannot be number")


#function to check for incorrectly formatted ISBN
def check_ISBN(potential_ISBN, index):
    #either 10 or 13 digits, with X only permitted as check digit
    pattern = r'^\d{10}$|^\d{13}$|^\d{9}X$|^\d{12}X$'
    if re.match(pattern, potential_ISBN):
        return True
    else:
        print("ISBN in row: ", index, " is incorrectly formatted: ", potential_ISBN)

#check for any missing values in either data frame
empty_cellsB = bRatings.isna().sum().sum()
empty_cellsNB = nbRatings.isna().sum().sum()

print(empty_cellsB)
print(empty_cellsNB)

#check ISBN validity in for both ratings datasets
for i in range(0, len(bRatings['ISBN'])):
    check_ISBN(bRatings['ISBN'][i], i)

for i in range(0, len(nbRatings['ISBN'])):
    check_ISBN(nbRatings['ISBN'][i], i)

#found inccorect capitalisation of last digit, corrected
bRatings.at[72566, 'ISBN'] = '039592720X'
bRatings.at[106228, 'ISBN'] = '039592720X'
bRatings.at[140240, 'ISBN'] = '039592720X'
bRatings.at[155055, 'ISBN'] = '039592720X'

#found ASIN number instead of ISBN, corrected
nbRatings.at[8213, 'ISBN'] = '038529929X'
nbRatings.at[12532, 'ISBN'] = '038529929X'
nbRatings.at[14375, 'ISBN'] = '038529929X'

#User IDS stored as strings, but can all be converted to integers
checkInterger(bRatings, 'User-ID')
checkInterger(nbRatings, 'User-ID')

#ratings are all integers, no floats or strings
checkInterger(bRatings, 'Book-Rating')
checkInterger(nbRatings, 'Book-Rating')

#check ratings in correct range in both datasets
for i in range(0, len(nbRatings)):
    if (nbRatings['Book-Rating'][i] <0) or (nbRatings['Book-Rating'][i] > 10):
        print('rating at row', i, 'out of range')

for i in range(0, len(bRatings)):
    if (bRatings['Book-Rating'][i] <0) or (bRatings['Book-Rating'][i] > 10):
        print('rating at row', i, 'out of range')

#output cleaned dataset
bRatings.to_csv("BX-Ratings-Cleaned.csv", index = False)
nbRatings.to_csv("BX-NewBooksRatings-Cleaned.csv", index = False)

