import pandas as pd
import genre_classifier as gc
import datetime
import multiprocessing

if __name__ == '__main__':

	ignored_words = {
		'',
		'a', 'an', 'the',
		'to', 'of', 'in', 'into', 'with', 'by', 'on', 'for', 'from',
		'as', 'when', 'after', 'at', 'up', 'out', 'about',
		'and', 'but',
		'is', 'are', 'has', 'be', 'must', 'will',
		'find',
		'he', 'she', 'his', 'him', 'her',
		'they', 'their', 'that', 'who', 'it', 'them', 'this',
		'young', 'new',
		'man', 'woman',
		'one', 'two',
		'life',
	}

	train_data = pd.read_csv('train.csv').loc[10000:]
	unique_genres = set(train_data['genre'])

	model = gc.GenreClassifier(ignored_words, unique_genres)

	print('Model training...')

	time_start = datetime.datetime.now()
	model.parallel_train(train_data)
	time_finish = datetime.datetime.now()

	print(f'Model is successfully trained in time {str(time_finish - time_start)}.')

	test_count = 10000
	test_data = pd.read_csv('train.csv').loc[range(test_count)]
	test_synopsises = list(test_data['synopsis'])
	exact_genres = list(test_data['genre'])

	print('Model testing...')

	time_start = datetime.datetime.now()

	# ==== ==== ==== ==== ==== ==== ==== ====
	# Вариант 1 (НЕ ПОДДЕРЖИВАЕТСЯ; ЗАПУСК ПРИВЕДЁТ К ОШИБКЕ; НУЖДАЕТСЯ В ПЕРЕРАБОТКЕ)
	#success_count = 0
	#i = 0
	#for genre, text in zip(exact_genres, test_synopsises):
	#	result = model.apply(text)
	#	if result == genre:
	#		success_count += 1
	#	i += 1
	#	#print(f'Test {i} completed ({result == genre}).')
	# ==== ==== ==== ==== ==== ==== ==== ====

	# ==== ==== ==== ==== ==== ==== ==== ====
	# Вариант 2
	#approx_genres = list(map(model.apply, test_synopsises))
	# ==== ==== ==== ==== ==== ==== ==== ====

	# ==== ==== ==== ==== ==== ==== ==== ====
	# Вариант 3
	#approx_genres = [ model.apply(text) for text in test_synopsises ]
	# ==== ==== ==== ==== ==== ==== ==== ====

	# ==== ==== ==== ==== ==== ==== ==== ====
	# Вариант 4
	with multiprocessing.Pool() as pool:
		approx_genres = list(pool.map(model.apply, test_synopsises))
	# ==== ==== ==== ==== ==== ==== ==== ====

	time_finish = datetime.datetime.now()

	print(f'Model is tested in time {str(time_finish - time_start)}.')

	success_count = sum(map(lambda p: p[0] == p[1], zip(exact_genres, approx_genres)))
	print(f'Total successes: {success_count} / {test_count} ({success_count / test_count * 100}%).')

	print('Analyzing results...')

	time_start = datetime.datetime.now()

	results = pd.DataFrame(
		0,
		columns = [
			'TP', 'FP', 'FN', 'TN',
			'TotalExact', 'CheckExact', 'TotalApprox', 'CheckApprox', 'Total', 'Check',
			'Accuracy', 'Precision', 'Recall|TPR', 'FPR', 'F1',
		],
		index = list(unique_genres)
	)

	for exact, approx in zip(exact_genres, approx_genres):
		if exact == approx:
			results.loc[exact, 'TP'] += 1
			for genre in unique_genres:
				if genre != exact:
					results.loc[genre, 'TN'] += 1
		else:
			results.loc[exact, 'FN'] += 1
			results.loc[approx, 'FP'] += 1
			for genre in unique_genres:
				if genre != exact and genre != approx:
					results.loc[genre, 'TN'] += 1

	results['TotalExact'] = list(map(exact_genres.count, results.index))
	results['CheckExact'] = results['TotalExact'] == results['TP'] + results['FN']

	results['TotalApprox'] = list(map(approx_genres.count, results.index))
	results['CheckApprox'] = results['TotalApprox'] == results['TP'] + results['FP']

	results['Total'] = results['TP'] + results['FP'] + results['FN'] + results['TN']
	results['Check'] = results['Total'] == test_count

	results['Accuracy'] = (results['TP'] + results['TN']) / test_count
	results['Precision']  = results['TP'] / (results['TP'] + results['FP'])
	results['Recall|TPR'] = results['TP'] / (results['TP'] + results['FN'])
	results['FPR']        = results['FP'] / (results['FP'] + results['TN'])
	results['F1'] = 2 * results['Precision'] * results['Recall|TPR'] / (
		results['Precision'] + results['Recall|TPR']
	)

	conf_matrix_mean = results[['TP', 'FP', 'FN', 'TN']].mean()

	micro_metrics = pd.Series(index = ['Accuracy', 'Precision', 'Recall|TPR', 'FPR', 'F1'])

	micro_metrics['Accuracy'] = (conf_matrix_mean['TP'] + conf_matrix_mean['TN']) / test_count
	micro_metrics['Precision']  = conf_matrix_mean['TP'] / (conf_matrix_mean['TP'] + conf_matrix_mean['FP'])
	micro_metrics['Recall|TPR'] = conf_matrix_mean['TP'] / (conf_matrix_mean['TP'] + conf_matrix_mean['FN'])
	micro_metrics['FPR']        = conf_matrix_mean['FP'] / (conf_matrix_mean['FP'] + conf_matrix_mean['TN'])
	micro_metrics['F1'] = 2 * micro_metrics['Precision'] * micro_metrics['Recall|TPR'] / (
		micro_metrics['Precision'] + micro_metrics['Recall|TPR']
	)

	macro_metrics = results[['Accuracy', 'Precision', 'Recall|TPR', 'FPR', 'F1']].mean()

	time_finish = datetime.datetime.now()

	print(f'Results are analyzed in time {str(time_finish - time_start)}.')

	print('==== ==== ==== ==== ==== ==== ==== ====')
	print(results)
	print('==== ==== ==== ==== ==== ==== ==== ====')
	print('Confusion matrix mean:')
	print(conf_matrix_mean)
	print('Micro-averaging metrics (ConfusionMatrix -> Mean -> Metric):')
	print(micro_metrics)
	print('Macro-averaging metrics (ConfusionMatrix -> Metric -> Mean):')
	print(macro_metrics)

	# Далее черновик

	#for genre, dc in model.knowledge.items():
	#	items_list = list(dc.items())
	#	items_list.sort(key = lambda p: p[1], reverse = True)
	#	print(f'FOR \'{genre}\':\n{items_list[:10]}\nDictionary length = {len(items_list)}\n')

	#print(f'TEST SYNOPSISES:\n{test_synopsises}\n')
	#print(f'TEST GENRES:\n{test_genres}\n')

	#for genre in unique_genres:
	#	syns = gc.synopsises_by_genre(train_data, genre)
	#	text = ' '.join(syns)
	#	wdcs = gc.word_counts_in_text(text, ignored_words)
	#	itli = list(wdcs.items())
	#	itli.sort(key = lambda p: p[1], reverse = True)
	#	print(f'FOR \'{genre}\':\n{itli[:10]}\nDictionary length = {len(itli)}\n')

	#word_counts_dictionary = {
	#	genre: gc.word_counts_in_text(
	#		' '.join(gc.synopsises_by_genre(train_data, genre)),
	#		ignored_words
	#	) for genre in unique_genres
	#}
	#normalized_WCD = {
	#	genre: gc.normalized_dictionary(
	#		word_counts_dictionary[genre]
	#	) for genre in unique_genres
	#}

	#for genre, word_counts in word_counts_dictionary.items():
	#	items_list = list(word_counts.items())
	#	items_list.sort(key = lambda p: p[1], reverse = True)
	#	print(f'FOR \'{genre}\':\n{items_list[:10]}\n')
