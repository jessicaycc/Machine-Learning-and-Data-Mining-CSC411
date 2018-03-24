import csv
import nltk
from nltk.tokenize import word_tokenize
import re, string; pattern = re.compile('[^a-zA-Z0-9_]+')

def clean_str(string):
    string = string.replace('U.S.', 'USA')
    string = string.replace('U.S.A', 'USA')
    string = string.replace('US ', 'USA ')
    string = string.replace('\'t', 't')
    string = string.replace('\'s', 's')
    string = string.replace('\'m', 'm')
    string = string.replace('\'ve', 've')
    string = string.replace('\'re', 're')
    string = string.replace('\'d', 'd')
    string = string.replace('\'ll', 'll')
    string = string.replace('(video)', 'video')
    string = string.replace('[video]', 'video')
    string = string.replace('re:', '')
    string = string.replace('100percentfedup.com','')
<<<<<<< HEAD
    string = re.sub(r"[^A-Za-z0-9\,.\`]", " ", string)   
    string = string.replace("n t ", "nt ") 
    string = string.replace(" s ", "s ")
    string = string.replace(" ll ", "ll ") 
    string = string.replace(" d ", "d ")
    string = string.replace(" ve ", "ve ")  
    string = string.replace(" re ", "re ") 
    string = string.replace(" m ", "m ") 
    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)   
    string = re.sub(r"\'s", "", string)   
=======
    string = string.replace("n t ", "nt ")

    # string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", "", string)
    string = re.sub(r"[^A-Za-z0-9\,.\`]", " ", string)
>>>>>>> 16c17dd4b2ed65b93165f4ab568e5bc1a088a4dc
    string = re.sub(r'(?<!\d)\.(?!\d)', " ", string)
    string = re.sub(r'(?<!\d)\,(?!\d)', " ", string)
    string = string.replace(r',', '')
    string = re.sub(r"\s{2,}", " ", string)

    if string[0] == ' ':
        string = string[1:]
    i = string.find('USA')
    
    if i != -1:
        string = string.strip().lower()
        s = list(string)
        s[i:i+3] = list('USA')
        string = ''.join(s)
        return string
    
    return string.lower()


def clean_fake():
    titles = set()
    try:
        for line in csv.DictReader(open("data/fake.csv")):
            if line['language']!= 'english' or line['country']!='US' or line['type']=='conspiracy' or line['type']=='satire':
                continue
            if line['thread_title']:
                otitle = line['thread_title'].lower()
                if "links" in otitle or "link" in otitle:
                    continue
                # print(otitle)
                otitle = clean_str(otitle)
                # "don t" -> "dont"; "wasn t" -> "wasnt"; etc
                otitle = otitle.replace("n t ", "nt ") 
                # otitle = otitle.replace(" s ", "s ")
                # otitle = otitle.replace(" ll ", "ll ") 
                # print(otitle)
                titles.add(otitle)
                # print (titles)
    except:
        pass

    outfile = open("data/clean_fake.txt", "w")
    for ntitle in titles:
        outfile.write(ntitle + "\n")


if __name__ == "__main__":
    clean_fake()
    