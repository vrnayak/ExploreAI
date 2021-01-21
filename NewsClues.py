# NewsClues.py
# OVERVIEW: Contains implementation of a web scraper that gathers headlines from news outlets

# News Outlet Abbreviations
# WSJ = Wall Street Journal
# FNT = Financial Times

import requests
import pandas as pd
import datetime as dt
from bs4 import BeautifulSoup


class NewsClues:

	def __init__(self, start = '01/01/2020', end = str(dt.date.today())):

		self.sources = ['WSJ']
		self.start = pd.to_datetime(start)
		self.end = pd.to_datetime(end)

		if self.start > self.end:
			temp = self.start
			self.start = self.end
			self.end = temp

		self.userAgent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_5) AppleWebKit/537.36 ' \
						 '(KHTML, like Gecko) Chrome/84.0.4147.89 Safari/537.36'
		self.headlineClass = {'WSJ': 'WSJTheme--headline--unZqjb45',
							  'FNT': 'js-teaser-heading-link'}

	# Returns a list of headlines from the specified news sources over the
	# specified time interval. Default arguments are in __init__()
	def getHeadlines(self, sources = None, start = None, end = None):

		sources = self.sources if sources is None \
								else [source.upper() for source in sources
									  	if source.upper() in self.sources]

		start = self.start if start is None else start
		end = self.end if end is None else end

		headlines = list()
		date = pd.to_datetime(start)
		enddate = pd.to_datetime(end)

		while date < enddate:
			
			for source in sources:
				parse = getattr(self, 'getDaily' + source + 'Headlines')	# Gets function of given name
				headlines.extend(parse(date))	# Add new headlines to existing ones
			date += dt.timedelta(days = 1)

		return headlines

	# Returns BeautifulSoup object created from a webpage given a URL
	def soupify(self, url):

		session = requests.Session()
		webpage = session.get(url, headers={'User-Agent': self.userAgent})
		return None if webpage.status_code == 404 else BeautifulSoup(webpage.content, 'html.parser')

	# Returns WSJ news headlines from the archive for any given day
	def getDailyWSJHeadlines(self, date):

		# Format URL for specific date
		url = 'http://www.wsj.com/news/archive/'
		identifier = '{}{}{}'.format(date.year, str(date.month).zfill(2), str(date.day).zfill(2))

		soup = self.soupify(url + identifier)
		if soup is None: return None

		headlines = list()
		tags = soup.find_all('h2', {'class': self.headlineClass['WSJ']})
		for tag in tags:
			link = tag.find('a', href=True)
			headlines.append((tag.string, date, link['href']))
		return headlines

	# Returns FNT news headlines from the archive for any given day
	# DO NOT RUN this, it is illegal without getting permission first
	def getDailyFNTHeadlines(self, date):

		pageNum = 1
		url = 'http://www.ft.com/companies?page={}'.format(pageNum)

		headlines = list()
		soup = self.soupify(url)
		while not soup is None:
			
			tags = soup.find_all('a', {'class': self.headlineClass['FNT']})
			dates = soup.find_all('time', {'class': 'o-teaser__timestamp'})

			headlines += [(tag.string, pd.to_datetime(day.string))
						  	for (tag, day) in zip(tags, dates)
						  	if pd.to_datetime(day.string) == date]

			pageNum += 1

		return headlines

parser = NewsClues(start = '07/15/2020', end = '07/18/2020')
hdlns = parser.getHeadlines(sources = ['WSJ'])
print(len(hdlns))

for hdln in hdlns:
	print(hdln)