import pandas as pd
import numpy as np
import regex as re
import requests
import nltk

from nltk.corpus import stopwords
from num2words import num2words

from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

books = pd.read_csv('BX-Books.csv')
newBooks = pd.read_csv('BX-NewBooks.csv')

#check for na values
empty_cellsB = books['Book-Title'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Title'].isna().sum().sum()

print('Books, book title NAs',empty_cellsB)
print('newBooks, book title NAs',empty_cellsNB)

empty_cellsB = books['Book-Author'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Author'].isna().sum().sum()

print('Books, book author NAs',empty_cellsB)
print('newBooks, book author NAs', empty_cellsNB)

empty_cellsB = books['Book-Publisher'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Publisher'].isna().sum().sum()

print('Books, book publisher NAs',empty_cellsB)
print('newBooks, book publisher NAs', empty_cellsNB)


###Function definitions

def checkPubErrors(df):
  #set up array of error check digits
  hits = [0,0,0, 0, 0]
  #have a non-error counter
  no_hit = 0


  for item in df['Year-Of-Publication']:
      #flag any no int values
      if type(item) != int:
          hits[0] +=1
      elif item == 0 :
          #flag 0 values
          hits[1] +=1
      elif item < 1900:
          #flag values lower than 1900
          hits[2] +=1
      elif item > 2024:
          #flag any 'future' dates
          hits[3] += 1
      else:
          no_hit += 1
  return("Inccorectly fomormatted values", hits, no_hit, len(df), sum(hits)+ no_hit)

def checkPubDateCorrections(df):
  hits = [0,0,0, 0, 0]
  no_hit = 0

  #flags error codes as in checkPuberrors, and checks 
  for item in df['Year-Of-Publication']:
      if type(item) != int:
          hits[0] += 1
      elif item == 0 :
          hits[1] +=1
      elif item == 1:
          hits[2] +=1
      elif item == 2:
          hits[3] += 1
      else:
          no_hit += 1
  return("error code checks: ", hits, no_hit, len(df), sum(hits)+ no_hit)

#function to check for incorrectly formatted ISBN
def check_ISBN(potential_ISBN, index):
    #either 10 or 13 digits, with X only permitted as check digit
    pattern = r'^\d{10}$|^\d{13}$|^\d{9}X$|^\d{12}X$'
    if re.match(pattern, potential_ISBN):
        return True
    else:
        print("ISBN in row: ", index, " is incorrectly formatted: ", potential_ISBN)

#these two functions send requests to ISBN database API

def requestPubDate(ISBN):
    #set up request header, authorization key will expire on may 15th
    h = {'Authorization': '53302_6a2e1c42983e2e6f80cc4be4ad933d63'}

     #request info for given ISBN
    response = requests.get(f"https://api2.isbndb.com/book/{ISBN}", headers=h)

    if response.status_code == 200:
        book_info = response.json()
        #if book info retrieved, check if date published is a field
        if 'date_published' in book_info['book']:
            #if it is, take the first 4 numbers (the year)
            publication_date = str(book_info['book']['date_published'])[0:4]
            
            if len(publication_date) == 0:
                #catches cases where the publication date can be retrieved, but is an empty string
                print('Replaced with 2 for ISBN:', ISBN)
                return(2)
            else:
                #if publication date found return it
                print("Succesful replacement:", publication_date)
                return(int(publication_date))
        else:
            #if no pub date available return 0 which can easily be detected in df later
            print('Replaced with 0 for ISBN:', ISBN)
            return(0)
            
    else:
        #if request fails return one, again for easy detection in dataframe afterwards
        print('Replaced with 1 for ISBN:', ISBN)
        return(1)

def requestTitle(ISBN):
    h = {'Authorization': '53302_6a2e1c42983e2e6f80cc4be4ad933d63'}

    #request info for given ISBN
    response = requests.get(f"https://api2.isbndb.com/book/{ISBN}", headers=h)

    if response.status_code == 200:
        book_info = response.json()
        if 'title_long' in book_info['book']:
            return(book_info['book']['title_long'])
        elif 'title' in book_info['book']:
            return(book_info['book']['title'])
        else:
            return(np.nan)


def fillNaTitles(df):
    for i in range(0, len(df)):
        if type(df['Book-Title'][i]) != str:
            df.at[i, 'Book-Title'] = requestTitle(df['ISBN'][i])
    return(df)

def requestPublisher(ISBN):
    h = {'Authorization': '53302_6a2e1c42983e2e6f80cc4be4ad933d63'}

    #request info for given ISBN
    response = requests.get(f"https://api2.isbndb.com/book/{ISBN}", headers=h)


    if response.status_code == 200:
        book_info = response.json()
        if 'publisher' in book_info['book']:
            return(book_info['book']['publisher'])
        else:
            return("Unknown")

def fillNaPublishers(df):
    for i in range(0, len(df)):
        if type(df['Book-Publisher'][i]) != str:
            df.at[i, 'Book-Publisher'] = requestPublisher(df['ISBN'][i])
    return(df)

def requestAuthor(ISBN, original_author):
    #set up request header, authorization key will expire on may 15th
    h = {'Authorization': '53302_6a2e1c42983e2e6f80cc4be4ad933d63'}

    #request info for given ISBN
    response = requests.get(f"https://api2.isbndb.com/book/{ISBN}", headers=h)


    if response.status_code == 200:
        book_info = response.json()
        #if book info succesfully retrieved, check if author is a field
        if 'authors' in book_info['book']:
            #if it is check if it contains at least one name
            if len(book_info['book']['authors']) > 0:
                #if so replace 
                new_author_name = book_info['book']['authors'][0]
                if len(book_info['book']['authors'][0].split()) != 1:
                    #if name is longer than one word clean it and return it
                    new_author_name = book_info['book']['authors'][0]
                    return(cleanAuthorName(new_author_name))
                    
                else:
                    #if does not contain any author names return the original
                    return(original_author)
            else:
                #if authors not a filed return the original
                return(original_author)
            
    else:
        #if request fails return the original
        return(original_author)

def replacePubErrors(df):
    for i in range(0, len(df)):
        #if one of the errors previously identified is encountered, attempt to replace it with the results from the ISBN db
        if df['Year-Of-Publication'][i] == 0:
            df.at[i, 'Year-Of-Publication'] = requestPubDate(df['ISBN'][i])
        elif df['Year-Of-Publication'][i] > 2024:
            df.at[i, 'Year-Of-Publication'] = requestPubDate(df['ISBN'][i])
    return(df)

def cleanAuthorName(author):
    #this function for use in cleaning author names returned from API request
    #do not apply this directly to dataframes as massively increased run time compared to .apply methods
    author = author.lower()

    #remove any text in brackets (i.e. adapter)
    author = re.sub(r'\(.*\)', '', author)

    #remove lots of common punctuation
    punctuation = [':', ';', ',',"'", "&", '\\', '"','/','!', ')', '(','-']
    for mark in punctuation:
        author = author.replace(mark, "")
    
    #hanlde '.' differently as often separates abbreviations j.r.r, j. r. r
    author = author.replace('.', " ")

    #removing any et als
    author = re.sub(r'\bet al\b', '', author)

    #remove common titles/name linkers
    author = re.sub(r'\b(?:dr|mr|mrs|ms|jr|md|de|phd)\b', '', author)

    #issue where some authors had two spaces between their names rather than one
    author = re.sub(r'\s+', ' ', author)

    #remove any leading or trailing spaces
    author = author.strip()

    return(author)

def searchCompleteName(df):
    #if author name is only one word long, attempt to fetch longer version from ISBN db
    for index in range(0, len(df)):
        author = df['Book-Author'][index]
        author = author.split()
        if len(author) == 1:
            df.at[index, 'Book-Author'] = requestAuthor(books['ISBN'][index], author[0])
    return(df)

def finaliseAuthorFormat(author):

    author = author.split()

    if len(author) > 1:
        #sort alphabetically by first letter of each word, solves issues where jrr is sometimes j r r
        author = sorted(author, key=lambda x: x[0])

        #take first letter of first word and last word
        return(author[0][0] + " " + author[-1])
    else:
        #if only one word in author then just take that
        return(author[0])

def preProcessAuthorNames(df):
  #make lower case
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.lower())

  #remove any text between brackets before remoing punctuation - sometimes (adapter) in author name
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\(.*\)', '', x))
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\[.*\]', '', x))

  #replace all punctuation with a space in case separating two words with space i.e. (hello:world), then remove and double space
  df = removePunctuation(df, 'Book-Author')

  #issue where some authors had two spaces between their names rather than one
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\s+', ' ', x))

  #removed a bunch of common titles/name linkers
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\b(?:dr|mr|mrs|ms|jr|md|de|phd)\b', '', x))

  #remove any stop words that may be in the name
  df['Book-Author'] = df['Book-Author'].apply(lambda x: removeStopwords(x))

  #now remove ', as didn't wouldn't have been recongised as a stop word otherwise
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.replace("'", ""))

  #removing any et als
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\bet al\b', '', x))

  #remove any leading and ending spaces
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.strip())

  return(df)

def lemmatizeString(string):
    lemmatizer = WordNetLemmatizer()

    newString = ""
    
    #for each word in the string replace it with its lemmatized version
    for word in string.split():
        newString = newString + " " + lemmatizer.lemmatize(word)
        
    newString = newString.strip()
    return(newString)

def stemString(string):
    ps = PorterStemmer()

    newString = ""
    
    #for each word in string take its stemmed version
    for word in string.split():
        newString = newString + " " + ps.stem(word)
        
    newString = newString.strip()
    return(newString)

def removeStopwords(string):
    newString = ""
    
    #search through words in string, exclude them if they are a stop word
    for word in string.split():
        if word not in stopwords.words('english'):
            newString = newString + " " + word
        
    newString = newString.strip()
    return(newString)

def convertNum2Word(string):
    newString = ""
    

    for word in string.split():
        try:
            #try and make the word an integer, if you can convert it to word format
            word = int(word)
            newString = newString + " " + num2words(word)
        except ValueError:
            #othweriwse leave word as is
            newString = newString + " " + word
        
    newString = newString.strip()
    return(newString)

def removePunctuation(df, col):
    #replace any symbol with a space (execpt ' which are handles differently)
    #done like this as these may or may not be separators, so add a space for now and remove double spaces afterwrds
    punctuation = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n,"
    for symbol in punctuation:
        df[col] = df[col].apply(lambda x: x.replace(symbol, " "))

    return(df)

def preProcessBookTitle(df):
    #lower case
    df['Book-Title'] = df['Book-Title'].apply(lambda x: x.lower())

    #remove any text between brackets before remoing punctuation

    #books['Book-Publisher'] = books['Book-Publisher'].apply(lambda x: re.sub(r'\(.*\)', '', x))

    #handle the - connector separately
    df['Book-Title'] = df['Book-Title'].apply(lambda x: x.replace('-', ' '))

    #remove the th/st after some numbers, like 21st
    df['Book-Title'] = df['Book-Title'].apply(lambda x: re.sub(r'\b(\d+)[a-zA-Z]+\b',r'\1', x))

    df = removePunctuation(df, 'Book-Title')
    
    #downstream will perform tf idf, so want to remove any stopwords
    df['Book-Title'] = df['Book-Title'].apply(lambda x: removeStopwords(x))

    #now remove ', as didn't wouldn't have been recongised as a stop word otherwise
    df['Book-Title'] = df['Book-Title'].apply(lambda x: x.replace("'", ""))

    #convert all numbers to their word equivalent
    df['Book-Title'] = df['Book-Title'].apply(lambda x: convertNum2Word(x))

    #this adds some punctuation back in so remove
    df = removePunctuation(df, 'Book-Title')

    #also adds stop words in
    df['Book-Title'] = df['Book-Title'].apply(lambda x: removeStopwords(x))

    #lemmatize words
    df['Book-Title'] = df['Book-Title'].apply(lambda x: lemmatizeString(x))

    #stem words
    df['Book-Title'] = df['Book-Title'].apply(lambda x: stemString(x))

    return(df)

def preProcessAuthorNames(df):
  #make lower case
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.lower())

  #remove any text between brackets before remoing punctuation - sometimes (adapter) in author name
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\(.*\)', '', x))
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\[.*\]', '', x))

  #replace all punctuation with a space in case separating two words with space i.e. (hello:world), then remove and double space
  df = removePunctuation(df, 'Book-Author')

  #issue where some authors had two spaces between their names rather than one
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\s+', ' ', x))

  #removed a bunch of common titles/name linkers
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\b(?:dr|mr|mrs|ms|jr|md|de|phd)\b', '', x))

  #remove any stop words that may be in the name
  df['Book-Author'] = df['Book-Author'].apply(lambda x: removeStopwords(x))

  #now remove ', as didn't wouldn't have been recongised as a stop word otherwise
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.replace("'", ""))

  #removing any et als
  df['Book-Author'] = df['Book-Author'].apply(lambda x: re.sub(r'\bet al\b', '', x))

  #remove any leading and ending spaces
  df['Book-Author'] = df['Book-Author'].apply(lambda x: x.strip())

  return(df)

def preProcessPublisher(df):
  #make lower case
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: x.lower())

    #remove text in brackets
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\(.*\)', '', x))
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\[.*\]', '', x))

  df = removePunctuation(df, 'Book-Publisher')

  #issue where some authors had two spaces between their names rather than one
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\s+', ' ', x))

  #remove any stop words that may be in the name
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: removeStopwords(x))

  #now remove ', as didn't wouldn't have been recongised as a stop word otherwise
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: x.replace("'", ""))
  #remove any plurals
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\b(\w+)s\b', r'\1', x))

  #remove any numbers
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\b(\d+)\b', '', x))

  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\b(for|usa|house|square|le|publishing|publication|book|collection|llc|gmbh|classic|ag|br|com|pres|pr|co|intl|corp|ltd|pub|plc|tb|pty|sud|inc|assn|st|group|")\b', '', x))

  #remove single letters
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: re.sub(r'\b[a-zA-Z]\b', '', x))

  #remove any leading and ending spaces
  df['Book-Publisher'] = df['Book-Publisher'].apply(lambda x: x.strip())  

  return(df)

def finalisePublisherFormat(df):
    publisher_freq_dict = {}

    for i in range(0, len(df)):
        #make a dictionary with words as keys and all documents they appear in as values
        if type(df['Book-Publisher'][i]) == str:
            for word in df['Book-Publisher'][i].split():
                if word in publisher_freq_dict:
                    publisher_freq_dict[word].append(i)
                else:
                    publisher_freq_dict[word] = [i]

    #turn the list of documents they appear in to a count 
    for value in publisher_freq_dict:
        publisher_freq_dict[value] = len(publisher_freq_dict[value])

    #sort words in reverse frequency order
    publisher_freq_dict = sorted(publisher_freq_dict.items(), key=lambda x:x[1], reverse=True)

    #iterate through frequency dict from most to least populatior 
    #when the word in the frequency dict is found in the publisher, substitute it for the full publisher name
    for i in range(0, len(df)):
        if type(df['Book-Publisher'][i]) == str:
            for value in publisher_freq_dict:
                if value[0] in df['Book-Publisher'][i].split():
                    df.at[i, 'Book-Publisher'] = value[0]
                    break
    return(df)

#######ISBN check section

#check that ISBNs are correctly formatted in books df
for i in range(0, len(books['ISBN'])):
    check_ISBN(books['ISBN'][i], i)

#found only one error where last X was not capitalised, corrected it manually
books.loc[books['ISBN'] == '039592720x', 'ISBN'] = "039592720X"

#check for New books
for i in range(0, len(newBooks['ISBN'])):
    check_ISBN(newBooks['ISBN'][i], i)

#found one ASIN number (rather than a ISBN number), corrected manually

newBooks.at[5139, 'ISBN'] = '038529929X'

#reset both indexes
books.reset_index(inplace=True)
newBooks.reset_index(inplace=True)

##### Publication date corrections

print(checkPubErrors(books))
print(checkPubErrors(newBooks))

#use requestPubDate function to try and find replacement publication dates in both datsets for those identified as incorrect (found only 0 and > 2024 errors)
books = replacePubErrors(books)
newBooks = replacePubErrors(newBooks)

#check for error codes from replace pub error function

print(checkPubDateCorrections(books))
print(checkPubDateCorrections(newBooks))

#given so few errors in books correct manually, left the record with errors in newBooks as unlikely to be central part of our analysis

books.at[12925, 'Year-Of-Publication'] = 1999
books.at[15719, 'Year-Of-Publication'] = 2000
books.at[16111, 'Year-Of-Publication'] = 1982
books.at[16754, 'Year-Of-Publication'] = 2000
books.at[16846, 'Year-Of-Publication'] = 1984
books.at[17766, 'Year-Of-Publication'] = 1997


######Author name correctons

print("cleaning author names")

books = preProcessAuthorNames(books)
print("pre process - books - done")
#books = searchCompleteName(books)
print("name search complete - books")
books['Book-Author'] = books['Book-Author'].apply(lambda x: finaliseAuthorFormat(x))
print("author finalising done - books")

newBooks = preProcessAuthorNames(newBooks)
print("pre process  - new books - done")
#newBooks = searchCompleteName(newBooks)
print("name search complete - newbooks")
newBooks['Book-Author'] = newBooks['Book-Author'].apply(lambda x: finaliseAuthorFormat(x))
print("author finalising done - newBooks")

######Book title cleaning
print("cleaning book titles")
books = fillNaTitles(books)
newBooks = fillNaTitles(newBooks)

books = preProcessBookTitle(books)
newBooks = preProcessBookTitle(newBooks)

print("title cleaning done")

######Book publisher cleaning


print("cleaning book publisher")

books = fillNaPublishers(books)
newBooks = fillNaPublishers(newBooks)

books = preProcessPublisher(books)
newBooks = preProcessPublisher(newBooks)

books = finalisePublisherFormat(books)
newBooks = finalisePublisherFormat(newBooks)

print("publisher cleaning done")

books.to_csv("BX-Books-CLEANED.csv", index = False)
newBooks.to_csv("BX-newBooks-CLEANED.csv", index = False)


#get final NA value counts after attempting to find corrections
#check for na values
print('Checking for final NA values')
empty_cellsB = books['Book-Title'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Title'].isna().sum().sum()

print('Books, book title NAs',empty_cellsB)
print('newBooks, book title NAs',empty_cellsNB)

empty_cellsB = books['Book-Author'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Author'].isna().sum().sum()

print('Books, book author NAs',empty_cellsB)
print('newBooks, book author NAs', empty_cellsNB)

empty_cellsB = books['Book-Publisher'].isna().sum().sum()
empty_cellsNB = newBooks['Book-Publisher'].isna().sum().sum()

print('Books, book publisher NAs',empty_cellsB)
print('newBooks, book publisher NAs', empty_cellsNB)