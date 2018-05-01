from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime, timedelta
from calendar import monthrange
from collections import Counter
import pandas as pd
import numpy as np
import statistics as st
import copy

last_date_train = datetime(2017, 10, 2)
# last_date_test = datetime(2018, 1, 31)

csv_data_folder = '.\\csv_data\\'
items_csv_data = '.\\itemsResolved.csv'
prices_csv_data = '.\\prices.csv'
train_csv_data = '.\\train.csv'

def main():

	items_data = pd.read_csv(items_csv_data, sep=';')
	prices_data = pd.read_csv(prices_csv_data, sep='|')	
	train_data = pd.read_csv(train_csv_data, sep='|')

	release_dates = items_data['releaseDate'].tolist()
	for i in range(0, len(release_dates)):
		release_dates[i] = datetime.strptime(release_dates[i], "%Y-%m-%d")

	# items_data = items_data.drop(columns = 'releaseDate')
	items_data['releaseDate'] = release_dates
	
	train_results = get_results(train_data, [last_date_train.month, last_date_train.month-1, last_date_train.month-2, last_date_train.month-3])
	items_data = treat_subCategory(items_data)
	items_data = vectorize_features(items_data)

	data = generate_base(release_dates, items_data, train_data, last_date_train, train_results, prices_data)

	# train(data)

def vectorize_features(items_data):
	df = items_data[['category']]
	df = df.astype(str)
	dummies = pd.get_dummies(df)
	# print(dummies)

	vect_columns = ['color', 'brand']
	for i in range(0, len(vect_columns)):
		aux_df = items_data[[vect_columns[i]]]
		dummies = pd.get_dummies(aux_df)	
		items_data = items_data.join(dummies)
		items_data = items_data.drop(columns = [vect_columns[i]])	

	aux_df = items_data[['size']]
	dummies = pd.get_dummies(aux_df)	
	items_data = items_data.join(dummies)

	vect_columns = ['category', 'mainCategory']
	for i in range(0, len(vect_columns)):
		aux_df = items_data[[vect_columns[i]]]
		aux_df = aux_df.astype(str)
		dummies = pd.get_dummies(aux_df)	
		items_data = items_data.join(dummies)
		items_data = items_data.drop(columns = [vect_columns[i]])

	return items_data

def generate_base(release_dates, items_data, train_data, last_date, results, prices_data):

	ctr = 0
	column_names = list()

	for column in items_data.columns:
		column_names.append(column)

	n = len(column_names)
	d = dict()

	column_names.append('age')
	column_names.append('weekday')
	column_names.append('weeknumber')
	column_names.append('price')
	column_names.append('price_rrp')
	column_names.append('max_rrp')
	column_names.append('target')

	for column in column_names:
		d[column] = list()

	# with open('test.csv', 'w') as f:

	# 	for column in column_names:
	# 		f.write("%s|" % column)
	# 	f.write("\n")

	data = pd.DataFrame(columns = column_names)

	for i in range(0, int(len(release_dates)/6000)):
		print(str(i))
		items = items_data.iloc[i]
		prices = prices_data.iloc[i]
		max_value = prices.tolist()
		max_value = max(max_value[2:])
		max_rrp = max_value/items.loc['rrp']

		for j in range(0, n):
			d[column_names[j]].append(items.loc[column_names[j]])

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
			d['age'].append(age)
			d['weekday'].append(weekday)
			d['weeknumber'].append(weeknumber)
			d['price'].append(price)
			d['price_rrp'].append(price_rrp)
			d['max_rrp'].append(max_rrp)
			d['target'].append(search_results(current_datetime, d['pid'][ctr], d['size'][ctr], results))

			# for key in d:
			# 	f.write("%s|" % d[key])
			# f.write("\n")
			
	for column in column_names:
		data[column] = d[column]

	vect_columns = ['weeknumber', 'weekday']
	for j in range(0, len(vect_columns)):
		aux_df = data[[vect_columns[j]]]
		aux_df = aux_df.astype(str)
		dummies = pd.get_dummies(aux_df)	
		data = data.join(dummies)
	data = data.drop(columns = ['size', 'weekday', 'weeknumber'])

	data.to_csv(".\\" + "new_data.csv", sep='|', index=False)
	# return data

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

	for i in range(0, len(dates)):
		for m in month:
			if (datetime.strptime(dates[i], "%Y-%m-%d").month == m):
				test_results.append(train_data.iloc[i])

	return test_results


def date_range(start_date, end_date):
    for n in range(int ((end_date - start_date).days)):
        yield start_date + timedelta(n)

def train(data):
	n_columns = len(data.columns)
	X_train = data[0:n_columns-1]
	Y_train = data['target']

	rf_model = RandomForestClassifier(n_estimators = 100, n_jobs = 7)
	score = cross_val_score(rf_model, X_train, Y_train, cv=6, scoring='roc_auc')
	print('roc area: ' + str(score.mean()))


def test(model, new_test, test_results):
	n_columns = len(new_test.columns)
	X_test = new_test[0:n_columns-1]

	Y_test = model.predict(X_test)
	print(accuracy_score(test_results, Y_test))

main()