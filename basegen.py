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

	items_data = treat_subCategory(items_data)

	vect_color = get_feature_vectorizer(items_data, 'color')
	vect_brand = get_feature_vectorizer(items_data, 'brand')
	vect_mainC = get_feature_vectorizer(items_data, 'mainCategory')
	vect_category = get_feature_vectorizer(items_data, 'category')
	items_data = vectorize_feature(items_data, 'color', vect_color)
	items_data = vectorize_feature(items_data, 'brand', vect_brand)
	items_data = vectorize_feature(items_data, 'mainCategory', vect_mainC)
	items_data = vectorize_feature(items_data, 'category', vect_category)

	items_data = items_data.drop(columns = ['color', 'brand', 'mainCategory', 'category'])

	# consertar size
	generate_base(release_dates, items_data, train_data, last_date_train, train_results, prices_data)

def generate_base(release_dates, items_data, train_data, last_date, results, prices_data):

	d = dict()
	column_names = list()

	for column in items_data.columns:
		column_names.append(column)

	n = len(column_names)

	column_names.append('age')
	column_names.append('weekday')
	column_names.append('weeknumber')
	column_names.append('price')
	column_names.append('price_rrp')
	column_names.append('max_rrp')
	column_names.append('target')

	data = pd.DataFrame(columns = column_names)

	for i in range(0, len(release_dates)):
		print(str(i))
		items = items_data.iloc[i]
		prices = prices_data.iloc[i]
		max_value = prices.tolist()
		max_value = max(max_value[2:])
		max_rrp = max_value/items.loc['rrp']

		for i in range(0, n):
			d[column_names[i]] = items.loc[column_names[i]]

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
			# consertar size
			d['target'] = search_results(current_datetime, d['pid'], d['size'], results)

			data.append(d, ignore_index=True)

	vect_weekday = get_feature_vectorizer(data, 'weekday')
	vect_weeknumber = get_feature_vectorizer(data, 'weeknumber')
	data = vectorize_feature(data, 'weekday', vect_weekday)
	data = vectorize_feature(data, 'weeknumber', vect_weeknumber)
	data = data.drop(columns = ['weekday', 'weeknumber'])

	data.to_csv(".\\" + "_trainResolved.csv", sep='\t', index=False)

# consertar size !!!
def search_results(date, pid, size, results):
	for i in range(0, len(results)):
		if (datetime.strptime(results[i]['date'], "%Y-%m-%d") == date and results[i]['pid'] == pid and results[i]['size'] == size):
			return 1

	return 0

def treat_subCategory(items_data):
	most_common = [3, 32, 21, 14, 25, 8, 16, 6, 5, 22]
	num_cat = len(most_common)
	idx = items_data.columns.get_loc("subCategory")
	n = len(items_data)

	for i in range(0, num_cat):
		items_data.insert(loc=idx+i+1, column='sCat' + str(most_common[i]), value=[0]*n)
	items_data.insert(loc = idx+num_cat+1, column='sCatOutros', value=[1]*n)

	for item in range(0, n):
		for j in range(0, num_cat):
			if (items_data.iloc[item][idx] == most_common[j]):
				column_subC = items_data.columns[idx+j+1]
				column_outros = items_data.columns[idx+num_cat+1]

				items_data.at[item, column_subC] = 1
				items_data.at[item, column_outros] = 0

	items_data = items_data.drop(columns = 'subCategory')

	return items_data

def get_results(train_data, month):
	dates = train_data['date'].tolist()
	test_results = list()

	# if(isinstance(month, int)):
	# 	for i in range(0, len(dates)):
	# 		if (datetime.strptime(dates[i], "%Y-%m-%d").month == month):
	# 			test_results.append(train_data.iloc[i])
	# else:
	for i in range(0, len(dates)):
		for m in month:
			if (datetime.strptime(dates[i], "%Y-%m-%d").month == m):
				test_results.append(train_data.iloc[i])

	return test_results


def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def get_feature_vectorizer(train_data, feature_name):
	all_feature_values = list()
	
	feature_lines = train_data[feature_name].tolist()
	for feature_line in feature_lines:
		if (feature_line == feature_line):
			features =  feature_line.split(',')
			for feature in features:
				all_feature_values.append(feature)

	temp_vectorizer = CountVectorizer()
	temp_vectorizer.fit(all_feature_values)
	return temp_vectorizer

def vectorize_feature(data, feature_name, vectorizer_feature):
	pos = data.columns.get_loc(feature_name)
	features_list = data[feature_name].tolist()
	for feature_line in features_list:
		feature_line = feature_line.split(',')

	new_feature_names = list()
	for old_feature_name in vectorizer_feature.get_feature_names():
		new_feature_names.append(feature_name + '_' + old_feature_name)

	new_features_mtx = vectorizer_feature.transform(features_list).toarray()
	new_df = pd.DataFrame(new_features_mtx, columns=new_feature_names)

	for idx in range(pos, len(new_feature_names)+pos):
		data.insert(loc = idx, column = new_feature_names[idx-pos], value = list(new_features_mtx[:,idx-pos]))

	return data
main()