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
    # data_trim = data.sample(200000)
    data.to_csv(filename_out)

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
    #retweets

    #test data generator:
    # dic = {'label': [0,4,0], 'tweet': ["me no, such. here! mightn't :)", 'suck a dick! you; twat :) :(', "frick :] that's uncool"]}
    # data = pd.DataFrame(data=dic)


    data = pd.read_csv(filename)



    users_reg = r"(@[\w]{1,15}\s)"
    seqlet_reg = r"([a-zA-z])\1{3,}"
    url_reg = r"(http(?:s){0,1}://[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*))"
    misc_reg = r"([^a-zA-Z:\(\)\[\] -'])"
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

    stopwords_reg = r"\bi\b|\bme\b|\bmy\b|\bmyself\b|\bwe\b|\bour\b|\bours\b|\bourselves\b|\byou\b|\byou're\b|\byou've\b|\byou'll\b"+\
    r"|\byou'd\b|\byour\b|\byours\b|\byourself\b|\byourselves\b|\bhe\b|\bhim\b|\bhis\b|\bhimself\b|\bshe\b|\bshe's\b|\bher\b|\bhers\b"+\
    r"|\bherself\b|\bit\b|\bit's\b|\bits\b|\bitself\b|\bthey\b|\bthem\b|\btheir\b|\btheirs\b|\bthemselves\b|\bwhat\b|\bwhich\b|\bwho\b"+\
    r"|\bwhom\b|\bthis\b|\bthat\b|\bthat'll\b|\bthese\b|\bthose\b|\bam\b|\bis\b|\bare\b|\bwas\b|\bwere\b|\bbe\b|\bbeen\b|\bbeing\b|\bhave\b"+\
    r"|\bhas\b|\bhad\b|\bhaving\b|\bdo\b|\bdoes\b|\bdid\b|\bdoing\b|\ba\b|\ban\b|\bthe\b|\band\b|\bbut\b|\bif\b|\bor\b|\bbecause\b|\bas\b"+\
    r"|\buntil\b|\bwhile\b|\bof\b|\bat\b|\bby\b|\bfor\b|\bwith\b|\babout\b|\bagainst\b|\bbetween\b|\binto\b|\bthrough\b|\bduring\b|\bbefore\b"+\
    r"|\bafter\b|\babove\b|\bbelow\b|\bto\b|\bfrom\b|\bup\b|\bdown\b|\bin\b|\bout\b|\bon\b|\boff\b|\bover\b|\bunder\b|\bagain\b|\bfurther\b"+\
    r"|\bthen\b|\bonce\b|\bhere\b|\bthere\b|\bwhen\b|\bwhere\b|\bwhy\b|\bhow\b|\ball\b|\bany\b|\bboth\b|\beach\b|\bfew\b|\bmore\b|\bmost\b"+\
    r"|\bother\b|\bsome\b|\bsuch\b|\bno\b|\bnor\b|\bonly\b|\bown\b|\bsame\b|\bso\b|\bthan\b|\btoo\b|\bvery\b|\bs\b|\bt\b|\bcan\b"+\
    r"|\bwill\b|\bjust\b|\bdon\b|\bdon't\b|\bshould\b|\bshould've\b|\bnow\b|\bd\b|\bll\b|\bm\b|\bo\b|\bre\b|\bve\b|\by\b|\bain\b|\baren\b"+\
    r"|\baren't\b|\bcouldn\b|\bcouldn't\b|\bdidn\b|\bdidn't\b|\bdoesn\b|\bdoesn't\b|\bhadn\b|\bhadn't\b|\bhasn\b|\bhasn't\b|\bhaven\b"+\
    r"|\bhaven't\b|\bisn\b|\bisn't\b|\bma\b|\bmightn\b|\bmightn't\b|\bmustn\b|\bmustn't\b|\bneedn\b|\bneedn't\b|\bshan\b|\bshan't\b"+\
    r"|\bshouldn\b|\bshouldn't\b|\bwasn\b|\bwasn't\b|\bweren\b|\bweren't\b|\bwon\b|\bwon't\b|\bwouldn\b|\bwouldn't\b"

    #removing url and usersnames before emoticons due to this interfering with emoticons
    data['tweet'] = data['tweet'].replace(to_replace=[url_reg, users_reg], value=['', ''], regex=True)
    #
    data = data.drop_duplicates()
    smiles = data['tweet'].str.contains(smile_reg, regex=True)
    frowns = data['tweet'].str.contains(sad_reg, regex=True)
    delete = data['tweet'].str.contains(delete_reg, regex=True)
    RT = data['tweet'].str.contains(r"\bRT @|\brt @|\bRt @", regex=True)

    data = data.loc[~(smiles&frowns)]
    data = data.loc[~(delete)]
    data = data.loc[~(RT)]


    # removing misc

    data['tweet'] = data['tweet'].replace(to_replace=[misc_reg, quote_reg, neutral_reg], value=['', '', ''], regex=True)
    data['tweet'] = data['tweet'].replace(to_replace=seqlet_reg, value=r"\1\1\1", regex=True)

    # data['tweet'] = data['tweet'].replace(stopwords_reg, '', regex=True)
    data['tweet'] = data['tweet'].replace(r"'", '', regex=True)
    data['tweet'] = data['tweet'].replace(r"\s+", ' ', regex=True)

    #removing all upper case
    data['tweet'] = data['tweet'].str.lower()

    # splitting data
    tokens = data['tweet'].str.split(' ')

    stemmer = SnowballStemmer('english');
    stemmed_tokens = tokens.apply(lambda x: [stemmer.stem(i) for i in x])
    data['tweet'] = stemmed_tokens.str.join(' ')




    vectorizer = TfidfVectorizer(min_df = 0.1, max_df = 0.8)
    vectorized = vectorizer.fit_transform(data['tweet'].to_numpy())
    print(np.shape(vectorized))

    # data.to_csv('data_trim_processed.csv')




if __name__ == '__main__':
    # pp('data_trim.csv')
    #trim('training.1600000.processed.noemoticon.csv', 'archive/training.1600000.processed.noemoticon_trimmed.csv')
    pp('archive/training.1600000.processed.noemoticon_trimmed.csv')
