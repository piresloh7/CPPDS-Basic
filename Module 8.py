#JSON
data = { "president": 
{ "name": "Zaphod Beeblebrox",
 "species": "Betelgeusian", 
"male": 1 
} }
import json 
type(data)#Data Type is Dictionary
jsonserial= json.dumps(data, indent=2) # serialize 
print(jsonserial)
type(jsonserial)#Data Type is json string
jsondeserial = json.loads(jsonserial) # deserialize 
print(jsondeserial)
type(jsondeserial)# Data type is Dictionary




#WEB SCRAPING
import requests
response=requests.get('http://www.hubertiming.com/results/2017gptr10k')
print(response.status_code)
print(response.text)
import bs4
soup = bs4.BeautifulSoup(response.text,'lxml')
title = soup.title
print(title)
# Print out the text
text = soup.get_text()
#print(soup.text)
soup.find_all('a')
all_links = soup.find_all("a")
for link in all_links:
    print(link.get("href"))
# Print the first 10 rows for sanity check
rows = soup.find_all('tr')
print(rows[:10])
for row in rows:
    row_td = row.find_all('td')
print(row_td)
type(row_td)
str_cells = str(row_td)
cleantext = bs4.BeautifulSoup(str_cells, "lxml").get_text()
print(cleantext)
import re
import pandas as pd
list_rows = []
for row in rows:
    cells = row.find_all('td')
    str_cells = str(cells)
    clean = re.compile('<.*?>')
    clean2 = (re.sub(clean, '',str_cells))
    list_rows.append(clean2)
print(list_rows)
print(clean2)
type(clean2)
# The next step is to convert the list into a dataframe and get a quick view of the first #10 rows using Pandas.
df = pd.DataFrame(list_rows)
df.head(10)

# Data Manipulation and Cleaning
df1 = df[0].str.split(',', expand=True)
df1.head(10)
df1[0] = df1[0].str.strip('[')
df1.head(10)
#Scraping the header fields
col_labels = soup.find_all('th')
all_header = []
col_str = str(col_labels)
cleantext2 = bs4.BeautifulSoup(col_str, "lxml").get_text()
all_header.append(cleantext2)
print(all_header)
df2 = pd.DataFrame(all_header)
df2.head()
df3 = df2[0].str.split(',', expand=True)
df3.head()
frames = [df3, df1]
df4 = pd.concat(frames)
df4.head(10)
df5 = df4.rename(columns=df4.iloc[0])
df5.head()
df5.info()
df5.shape
df6 = df5.dropna(axis=0, how='any')
df6.head()
df7 = df6.drop(df6.index[0:2])
df7.head()
df7.rename(columns={'[Place': 'Place'},inplace=True)
df7.rename(columns={' Team]': 'Team
'},inplace=True)
df7.head()
df7['Team'] = df7['Team'].str.strip(']')
df7.head()




#To learn about Twitter Web Scraping follow this link
#https://www.promptcloud.com/blog/scrape-twitter-data-using-python-r/





















