from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

def trim(filename_inn, filename_out):
    """
    simply reads csv file, filters out
    """
    data = pd.read_csv('archive/'+filename_inn,
        usecols = [0,5],
        names=['label', 'tweet'],
        encoding='latin1'
    )
    data_trim = data.sample(200)
    data_trim.to_csv('tiny'+filename_out)

def pp(filename):
    """
    takes the filename of a csv file and edits it in preperation for sentiment analysis.
    writes it to file

    args:
        filename(string): csv file with data

    """
    #TODO:
    #remove all two (and one?) character words
    #remove ! ?



    data = pd.read_csv(filename)


    """
    regex definitions:
    """
    users_reg = r"(@[\w]{1,15}\s)"

    url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
    misc_reg = r"([^a-zA-Z:\(\)\[\] -'])"
    amps_reg = r"&amp;", # ampresand &
    lt_reg = r"&lt;", # less than <
    gt_reg = r"&gt;", # greater than >
    quote_reg = "&quot;"
    smile_reg = (r":-\)|:\)|:-\]|:]|:-3|:3|:-&gt;|:&gt;|8-\)|8\)|:-\}|:\}|:o\)|"
                 + r":c\)|:\^\)|=\]|=\)|:-D|:D|8-D|8D|x-D|xD|X-D|XD|=D|=3|B\^D"
                 + r":'-\)|:'\)|:-\*|:\*|;-\)|;\)|\*-\)|\*\)|;-\]|;\]|;\^\)|:-,|"
                 + r";D|&lt;3+")
    sad_reg = (r":-\(|:\(|:-c|:c|:-&lt;|:&lt;|:-\[|:\[|:-\|\||&gt;:\[|:\{|:@|"
               + r";\(|:'-\(|:'\(|D-':|D:&lt;|D:|D8|D;|D=|:-\/|:\/|:-\.|"
               + r"&gt;:\\|&gt;:\/|:\\|=\/|=\\|:L|=L|:S|&gt;:-\)|&gt;:\)|"
               + r"\}:-\)|\}:\)|3:-\)|3:\)|&gt;;\)|&gt;:3|;3|&gt;:\]|"
               + r":-###\.\.|:###\.\.|&lt;\/3+|&lt;\\3+")
    neutral_reg = (r":-O|:O|:-o|:o|:-0|8-0|&gt;|:-\||:\||:\$|:\/\/\)|:\/\/3|:-X|:X|"
                   + r":-#|:#|:-&|:&|&lt;:-\|',:-\||',:-l|%-\)|%\)|:E|O_O|o-o|O_o|"
                   + r"o_O|o_o|O-O|O\.o|O\.O|o\.o|o\.O")

    delete_reg = r":-P|:P|X-P|XP|x-p|xp|:-p|:p|:-b|:b|d:|=p|&gt;:P"

    # replace negations with not
    negation_list = ['ain', 'aint', 'aren', 'arent', "aren't", 'couldn',
                     'couldnt', "couldn't", 'didn', 'didnt', "didn't", 'doesn',
                     'doesnt', "doesn't", 'hadn', 'hadnt', "hadn't", 'hasn',
                     'hasnt', "hasn't", 'haven', 'havent', "haven't", 'isn',
                     'isnt', "isn't", 'mightn', "mightn't", 'mightnt', 'mustn',
                     'mustnt', "mustn't", 'needn', 'neednt', "needn't", 'shan',
                     'shant', "shan't", 'shouldn', 'shouldnt', "shouldn't",
                     'wasn', 'wasnt', "wasn't", 'weren', 'werent', "weren't",
                     'won', 'wont', "won't", 'wouldn', 'wouldnt', "wouldn't",
                     'don', 'dont', "don't", 'cant', "can't", 'cannot',
                     'darent', "daren't"]


    # remove stopwords
    stopword_list = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours',
                     'ourselves', 'you', "you're", "you've", "you'll", "you'd",
                     'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
                     'his', 'himself', 'she', "she's", 'her', 'hers', 'herself',
                     'it', "it's", 'its', 'itself', 'they', 'them', 'their',
                     'theirs', 'themselves', 'what', 'which', 'who', 'whom',
                     'this', 'that', "that'll", 'these', 'those', 'am', 'is',
                     'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
                     'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
                     'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
                     'while', 'of', 'at', 'by', 'for', 'with', 'about',
                     'against', 'between', 'into', 'through', 'during',
                     'before', 'after', 'above', 'below', 'to', 'from', 'up',
                     'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                     'further', 'then', 'once', 'here', 'there', 'when',
                     'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
                     'more', 'most', 'other', 'some', 'such', 'only', 'own',
                     'same', 'so', 'than', 'too', 'very', 's', 't', 'can',
                     'will', 'just', 'should', "should've", 'now', 'd', 'll',
                     'm', 'o', 're', 've', 'y', 'ma', 'could', 'need']



    #initially removing all lines with retweets and duplicates
    RT = data['tweet'].str.contains(r"\bRT @|\brt @|\bRt @", regex=True)
    data = data.loc[~(RT)]
    data = data.drop_duplicates()

    #removing url and usersnames before emoticons due to this interfering with emoticons
    data['tweet'] = data['tweet'].replace(to_replace=[url_reg, users_reg], value=[' URL ', ' USER '], regex=True)

    #remove lines with both smiles and frowns, as this fuckes with the algorithm
    #also remove all liens with smilies in the delete reges for the same reason
    smiles = data['tweet'].str.contains(smile_reg, regex=True)
    frowns = data['tweet'].str.contains(sad_reg, regex=True)
    delete = data['tweet'].str.contains(delete_reg, regex=True)

    data = data.loc[~(smiles&frowns)]
    data = data.loc[~(delete)]

    # replace postive and negative emoticons
    data = data.replace(to_replace=smile_reg, value=' positive ', regex=True)
    data = data.replace(to_replace=sad_reg, value=' negative ', regex=True)

    # remove neutral emoticons
    data = data.replace(to_replace=neutral_reg, value=' ', regex=True)

    #removing all upper case
    data['tweet'] = data['tweet'].str.lower()


    # replace n't ending of words with not
    data = data.replace(to_replace=r"can't|cannot", value=r"can not", regex=True)
    data = data.replace(to_replace=r"won't", value=r"will not", regex=True)
    data = data.replace(to_replace=r"([a-zA-Z]+)(n't)",
                    value=r"\1 not",
                    regex=True)

    #removing misc

    data['tweet'] = data['tweet'].replace(to_replace=[quote_reg, gt_reg, lt_reg, amps_reg, misc_reg],
                                          value=['', '', '', '', ''],
                                          regex=True)


    # replace repetitions of xoxo and haha
    data = data.replace(to_replace=r"(xo)(?:\1)+x{0,1}",
                    value=r"xoxo",
                    regex=True)
    data = data.replace(to_replace=r"(ha)(?:\1)+h{0,1}",
                    value=r"haha",
                    regex=True)

    #replacin sequential letters
    data['tweet'] = data['tweet'].replace(to_replace= r"([a-zA-z])\1{2,}",
                                          value=r"\1\1",
                                          regex=True)

    # data['tweet'] = data['tweet'].replace(stopwords_reg, '', regex=True)
    data['tweet'] = data['tweet'].replace(r"'", '', regex=True)
    data['tweet'] = data['tweet'].replace(r"\s+", ' ', regex=True)

    #negations
    negation_regex = r"\b{}\b".format(r'\b|\b'.join(negation_list))
    data = data.replace(to_replace=negation_regex, value=r"not", regex=True)

    #removing stopwords
    stopword_regex = r"\b{}\b".format(r'\b|\b'.join(stopword_list))
    data = data.replace(to_replace=stopword_regex, value=r"", regex=True)

    # remove all apostrophe
    data = data.replace(to_replace=r"'", value=r"", regex=True)

    # splitting data
    tokens = data['tweet'].str.split(' ')

    stemmer = SnowballStemmer('english');
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
    data['tweet'] = stemmed_tokens.str.join(' ')



    # 
    # vectorizer = TfidfVectorizer(min_df = 0.1, max_df = 0.8)
    # vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    # print(np.shape(vectorized))

    data.to_csv('data_trim_processed.csv')




if __name__ == '__main__':
    # pp('data_trim.csv')
    # trim('training.1600000.processed.noemoticon.csv', 'training.1600000.processed.noemoticon_trimmed.csv')
    pp('archive/tinytraining.1600000.processed.noemoticon_trimmed.csv')
