import json
import nytimes
import time
import csv
import re
from cleaning import *


search_obj = nytimes.get_article_search_obj ('8feaa600919f492092252a8a1231f47d')
data = []

for x in range(200):
    try:
        print ("Page %d" %(x))
<<<<<<< HEAD
        f = search_obj.article_search(q='trump', fl=['headline'], begin_date='20150101', page=str(x), sort='oldest')   
    except:break
=======
        f = search_obj.article_search(q='trump', fl=['headline'], begin_date='20170601', page=str(x), sort='newest')   
    except:
        break
>>>>>>> 16c17dd4b2ed65b93165f4ab568e5bc1a088a4dc

    try:
        for k in f['response']['docs']:
            title = k['headline']['print_headline'].encode('utf-8')
            if title:
                data.append([title])
        time.sleep(1)
    except:
        pass

print ("Writing to file...")
with open('data/ny_times.csv', 'w') as f:
    wr = csv.writer(f)
    wr.writerows(data)

with open('data/ny_times.csv', 'r') as f:
    for line in f:
        print (line)
        new_line = clean_str(line)
<<<<<<< HEAD
        print (new_line)
=======

>>>>>>> 16c17dd4b2ed65b93165f4ab568e5bc1a088a4dc
        with open("data/clean_real.txt", "a") as myfile:
            myfile.write(new_line+'\n')


