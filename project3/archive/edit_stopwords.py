import numpy as np

def readfile(filename):
    stopwords=[]
    infile = open(filename, 'r')
    for line in infile:
        stopwords.append(line.strip())
    infile.close()
    return stopwords

def writefile(list):
    outfile = open('stopwords_alt_edited.txt', 'w')
    outfile.write("'[")
    for i in range(len(list)):
        outfile.write(f"\\b{list[i]}\\b|")
    outfile.close()

if __name__ == '__main__':
    import nltk
    from nltk.corpus import stopwords
    print(stopwords.words('english'))
    list = []
    for i in stopwords.words('english'):
        list.append(i)

    writefile(list)
