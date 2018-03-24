import json
import re
import requests
from os import makedirs
from os.path import join, exists
from datetime import date, timedelta
from cleaning import *



ARTICLES_DIR = join('tempdata', 'articles')
makedirs(ARTICLES_DIR, exist_ok=True)
# Sample URL
#
# http://content.guardianapis.com/search?from-date=2016-01-02&
# to-date=2016-01-02&order-by=newest&show-fields=all&page-size=200
# &api-key=your-api-key-goes-here

MY_API_KEY = "9e539038-379f-40eb-b119-11faeb340999"
API_ENDPOINT = 'http://content.guardianapis.com/search'
my_params = {
    'from-date': "",
    'to-date': "",
    'order-by': "newest",
    'show-fields': 'all',
    'page-size': 200,
    'api-key': MY_API_KEY
}

# day iteration from here:
# http://stackoverflow.com/questions/7274267/print-all-day-dates-between-two-dates
start_date = date(2017, 6, 1)
end_date = date(2018,3, 23)
dayrange = range((end_date - start_date).days + 1)

for daycount in dayrange:
    dt = start_date + timedelta(days=daycount)
    datestr = dt.strftime('%Y-%m-%d')

    # if not exists('guardian.txt'):
        # then let's download it
    print("Downloading", datestr)
    all_results = []
    my_params['from-date'] = datestr
    my_params['to-date'] = datestr
    current_page = 1
    total_pages = 1

    while current_page <= total_pages:
        print("...page", current_page)
        my_params['page'] = current_page
        resp = requests.get(API_ENDPOINT, my_params)
        data = resp.json()
        headline = data['response']['results'][0]['fields']['headline']
        headline = ''.join(headline)
        #all_results.extend(headline)
        headline = clean_str(headline)

        with open('data/guardian.txt', 'a') as f:
            f.write(headline+"\n")
        print(headline)
        # if there is more than one page
        current_page += 1
        total_pages = data['response']['pages']

with open('data/clean_real.txt', 'a') as f:
    with open('data/guardian.txt', 'r') as myfile:
        f.write(myfile.read())
    