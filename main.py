from bs4 import BeautifulSoup
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
import matplotlib.pyplot as plt

finviz_url = "https://finviz.com/quote.ashx?t="
tickers = ['AAPL', 'MSFT', 'SIX']

news_tables = {}
for ticker in tickers:
    url = finviz_url + ticker

    req = Request(url=url, headers={'user-agent': "my-app"})
    response = urlopen(req)

    html = BeautifulSoup(response, 'html.parser')
    news_table = html.find(id = "news-table")
    news_tables[ticker]= news_table


#print(news_tables)

parsed_data = []
for ticker, news_table in news_tables.items():
    for rows in news_table.findAll('tr'):
        title = rows.a.get_text()
        date_data = rows.td.text.split(" ")

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]
        parsed_data.append([ticker, date, time, title])

df = pd.DataFrame(parsed_data, columns = ["ticker", "date", "time", "title"])
vader = SentimentIntensityAnalyzer()

f = lambda title: vader.polarity_scores(title)['compound']
df['compound'] = df['title'].apply(f)
#convert from string to date time format
df['date'] = pd.to_datetime(df.date).dt.date

plt.figure(figsize=(10,8))
mean_df = df.groupby(['ticker', 'date']).mean()
mean_df = mean_df.unstack()
#taking cross section and transposing
mean_df = mean_df.xs("compound", axis = 'columns').transpose()
mean_df.plot(kind = 'bar')
plt.show()
