import re
import numpy as np
import pandas as pd
import multiprocessing
#import concurrent.futures


# keys({ key: value, ... }) := { key, ... }

# { key: value, ... }[key] := value

# dc[key]|alt := if key in keys(dc) then dc[key] else alt

# norm({ key: value, ... }) := norm([ value, ... ])


# (dataframe, genre) -> select 'synopsis' from dataframe
# where
#     'genre' == genre
def synopsises_by_genre(dataframe, genre):
	return list(dataframe.loc[dataframe['genre'] == genre, 'synopsis'])


# (text, ignored_words) -> { word: count, ... }
# where
#     word not in ignored_words
def word_counts_in_text(text, ignored_words):
	words = re.split(' |,|;|\.|\(|\)|\?|!|\n', text.lower())
	words = [ word for word in words if word not in ignored_words ]
	return { word: words.count(word) for word in set(words) }


# { key: value, ... } -> { key: relative_number, ... }
# where
#     relative_number := value / norm({ key: value, ... })
def normalized_dictionary(d):
	d_values_np = np.array(list(d.values()))
	return dict(zip(d.keys(), d_values_np / np.linalg.norm(d_values_np)))


# (dc1, dc2) -> norm({ key: value, ... })
# where
#     { key, ... } := keys(dc1) | keys(dc2)
#     value := abs(dc1[key]|0 - dc2[key]|0)
def dictionary_distance(d1, d2):
	keys_set = set(d1) | set(d2)
	abs_sub_d = { key: np.abs(d1.get(key, 0) - d2.get(key, 0)) for key in keys_set }
	return np.linalg.norm(list(abs_sub_d.values()))


class Trainer:

	def __init__(self, train_data, ignored_words):
		self.train_data = train_data
		self.ignored_words = ignored_words

	def set_train_data(self, train_data):
		self.train_data = train_data

	def set_ignored_words(self, ignored_words):
		self.ignored_words = ignored_words

	def task(self, genre):
		return (
			genre,
			normalized_dictionary(
				word_counts_in_text(
					' '.join(synopsises_by_genre(self.train_data, genre)),
					self.ignored_words
				)
			)
		)


class GenreClassifier:

	# self.knowledge := { genre: { word: relative_number, ... }, ... }
	# where
	#     relative_number := count / norm({ word: count, ... })

	def __init__(self, ignored_words, unique_genres):
		self.ignored_words = ignored_words
		self.unique_genres = unique_genres
		self.knowledge = {}

	def set_ignored_words(self, ignored_words):
		self.ignored_words = ignored_words

	def set_unique_genres(self, unique_genres):
		self.unique_genres = unique_genres

	def train(self, train_data):
		tr = Trainer(train_data, self.ignored_words)
		self.knowledge = dict(map(tr.task, self.unique_genres))

	def parallel_train(self, train_data):
		tr = Trainer(train_data, self.ignored_words)
		with multiprocessing.Pool(3) as pool:
			self.knowledge = dict(pool.map(tr.task, self.unique_genres))

	def apply(self, text):
		nd = normalized_dictionary(word_counts_in_text(text, self.ignored_words))
		distances = [
			(
				genre, dictionary_distance(self.knowledge[genre], nd)
			) for genre in self.unique_genres
		]
		return min(distances, key = lambda p: p[1])[0]
