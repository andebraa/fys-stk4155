# fys-stk4155
Project 3, we wished to look at sentiment analysis using decision trees. This is not something often u
sed for sentiment analysis, so it was something we wished to look more into.

Code is essentially divided into a pre processing script, which trims a dataset and strips it for anal
ysis, and a decision tree and different ensemble method scipts.

The layout is in large part very specific for my code setup, as pre processing reads and writes to the
 archive folder.

## XGBoost
uses the XG Boost package to produce a prediction from the test data. 
##### args:
&nbsp;&nbsp;&nbsp;&nbsp; filename(string): path to csv data file
&nbsp;&nbsp;&nbsp;&nbsp; depth (int) optional: the size of each tree in the method.

## randomforest
Applies the sklearn random forest pacage. prints accuracy for test and train data.
##### args:
&nbsp;&nbsp;&nbsp;&nbsp; filename(string): path to csv data file which is predicted.
##### returns:
&nbsp;&nbsp;&nbsp;&nbsp; test_acc (float): accuracy of test dataset
&nbsp;&nbsp;&nbsp;&nbsp; train_acc (float): accuracy of train dataset

## adaboost
Applies the sklearn random forest pacage. prints accuracy for test and train data.
##### args:
&nbsp;&nbsp;&nbsp;&nbsp; filename(string): path to csv data file which is predicted.
&nbsp;&nbsp;&nbsp;&nbsp; lambda (int): learning rate of the algorithm. 
##### returns:
&nbsp;&nbsp;&nbsp;&nbsp; test_acc (float): accuracy of test dataset
&nbsp;&nbsp;&nbsp;&nbsp; train_acc (float): accuracy of train dataset

## decisionTree
Applies the sklearn decision tree pacage. prints accuracy for test and train data.
##### args:
&nbsp;&nbsp;&nbsp;&nbsp; filename(string): path to csv data file which is predicted.
&nbsp;&nbsp;&nbsp;&nbsp; depth(int): depth of decision tree pased to sklearn decision tree.
##### returns:
&nbsp;&nbsp;&nbsp;&nbsp; test_acc (float): accuracy of test dataset
&nbsp;&nbsp;&nbsp;&nbsp; train_acc (float): accuracy of train dataset


## preprocessing
Program for removing uneccesary characters from strings in a pandas database.
NOTE the format of the database is very specific, and made especially for our dataset.
stopwords are from the nlkt database.

removes elements in the following order (Note, the order is important)
- retweets
- duplicates
- url and usernames
- tweets containing both positive and negative tweets
- certain neutral smilies due to twitter API error 
- replaces smiles with 'positive', and negative emoticons with 'negative'
- replaces everything with lower case 
- replces repititions of xoxo and haha 
- sequential letters apearing more than thrice
- negations replaced with 'NOT'
- removing stopwords (altered version of nltk.stopwords)
- stemming data

##### args: 
&nbsp;&nbsp;&nbsp;&nbsp; filename_in (string): The name of the file. path is assumed to be archive/
&nbsp;&nbsp;&nbsp;&nbsp; filename_out (string): Name of the file as it will be written after processing. path is again archive/


## trim
reads csv file, extracts collumn 0 and 5 (label and tweet) then uses pandas df.sample() to extract random collumns in data. This is because the original dataset was 800 000 negative and then positive tweets.
writes resulting dataset to file


## dependencies:
Dependencies for the whole project

&nbsp;&nbsp;&nbsp;&nbsp;nltk- natural language tool kit
    &nbsp;&nbsp;&nbsp;&nbsp;might have to run

      '''
      >> import nltk
      >> nltk.download('stopwords')
      '''
&nbsp;&nbsp;&nbsp;&nbsp;Pandas
&nbsp;&nbsp;&nbsp;&nbsp;Numpy 
&nbsp;&nbsp;&nbsp;&nbsp;nltk.corpus 
&nbsp;&nbsp;&nbsp;&nbsp;nltk.stem.snowball 
&nbsp;&nbsp;&nbsp;&nbsp; sklearn

      
      

