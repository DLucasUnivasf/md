from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta
from calendar import monthrange
import pandas as pd
import numpy as np
import statistics as st
import copy

last_date_train = datetime(2017, 12, 31)
last_date_test = datetime(2018, 1, 31)

csv_data_folder = '.\\csv_data\\'
items_csv_data = '.\\itemsResolved.csv'
prices_csv_data = '.\\prices.csv'
train_csv_data = '.\\train.csv'

def main():

	items_data = pd.read_csv(items_csv_data, sep=',')
	prices_data = pd.read_csv(prices_csv_data, sep='|')	
	train_data = pd.read_csv(train_csv_data, sep='|')

	release_dates = items_data['releaseDate'].tolist()
	for i in range(0, len(release_dates)):
		release_dates[i] = datetime.strptime(release_dates[i], "%Y-%m-%d")

	items_data['releaseDate'] = release_dates
	train_results = get_results(train_data, [last_date_train.month, last_date_train.month-1, last_date_train.month-2])

	treat_subCategory(items_data)

	generate_base(release_dates, items_data, train_data, last_date_train, train_results, prices_data)

def treat_subCategory(items_data):
	# most_common = [3, ]

def generate_base(release_dates, items_data, train_data, last_date, results, prices_data):

	d = dict()	
	data = pd.DataFrame(columns = ['pid', 'size', 'color', 'brand', 'rrp', 'mainCategory', 'category', 'subCategory', 'stock', 'age', 'weekday', 'weeknumber', 'price', 'price_rrp', 'max_rrp', 'target'])

	for i in range(0, len(release_dates)):
		print(str(i))
		items = items_data.iloc[i]
		prices = prices_data.iloc[i]
		max_value = prices.tolist()
		max_value = max(max_value[2:])
		max_rrp = max_value/items.loc['rrp']
					
		d['pid'] = items.loc['pid']
		d['size'] = items.loc['size']
		d['color'] = items.loc['color']
		d['brand'] = items.loc['brand']
		d['rrp'] = items.loc['rrp']
		d['mainCategory'] = items.loc['mainCategory']
		d['category'] = items.loc['category']
		d['subCategory'] = items.loc['subCategory']
		d['stock'] = items.loc['stock']

		for current_datetime in date_range(release_dates[i], last_date):
			print(current_datetime)
			weekday = current_datetime.weekday()
			age = (current_datetime - release_dates[i]).days

			day = current_datetime.day
			if (day >= 1 and day <= 7): 
				weeknumber = 1
			elif (day > 7 and day <= 14):
				weeknumber = 2
			elif (day > 14 and day <= 21):
				weeknumber = 3
			elif (day > 21 and day <= 28):
				weeknumber = 4
			else:
				weeknumber = 5

			price = prices.loc[current_datetime.strftime("%Y-%m-%d")]
			price_rrp = price/items.loc['rrp']

			d['age'] = age
			d['weekday'] = weekday
			d['weeknumber'] = weeknumber
			d['price'] = price
			d['price_rrp'] = price_rrp
			d['max_rrp'] = max_rrp
			d['target'] = search_results(current_datetime, d['pid'], d['size'], results)

			data.append(d, ignore_index = True)

	data.to_csv(".\\" + "_trainResolved.csv", sep='\t', index=False)

def fill_missing_values(items_data, idx, subCategories):
	n = len(items_data)
	most_common = st.mode(subCategories)
	
	for i in range(0, n):
		if items_data.iloc[i][idx] != items_data.iloc[i][idx]:
			items_data.iloc[i][idx] = most_common

	items_data.to_csv(".\\" + "items_data_modificado.csv", sep='|', index=False)

def get_results(train_data, month):
	dates = train_data['date'].tolist()
	test_results = list()

	if(isinstance(month, int)):
		for i in range(0, len(dates)):
			if (datetime.strptime(dates[i], "%Y-%m-%d").month == month):
				test_results.append(train_data.iloc[i])
	else:
		for i in range(0, len(dates)):
			for m in month:
				if (datetime.strptime(dates[i], "%Y-%m-%d").month == m):
					test_results.append(train_data.iloc[i])

	return test_results

def search_results(date, pid, size, results):
	for i in range(0, len(results)):
		if (datetime.strptime(results[i]['date'], "%Y-%m-%d") == date and results[i]['pid'] == pid and results[i]['size'] == size):
			return 1

	return 0

def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)
main()