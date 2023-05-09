# TODO: requirements.txt
# TODO: functiecomments + gewone comments
# TODO: check welke libraries je daadwerkelijk gebruikt

from recommendations import *
from collections import Counter
import itertools
import ast
import pandas as pd
import numpy as np
from kmodes.kmodes import KModes 
from kmodes.kprototypes import KPrototypes 
import matplotlib.pyplot as plt
from kneed import KneeLocator
import random 
from datetime import datetime, date
pd.options.mode.chained_assignment = None
import pickle

def jaccard_similarity(user_set, others_set):
	"""
	Calculate Jaccard similarity.
	"""
	set_intersect = user_set.intersection(others_set)
	set_union = user_set.union(others_set)
	similarity = len(set_intersect) / len(set_union)

	return similarity

def get_diverse_recommendations(user_index, user_data, content_data, sim_genre=True, sim_demo=True, sim_favorites=True):
	# - je gaat over de top10 favorites van de user 
	# - voor deze favorite (bijv. Roxanne) zoek je wat het genre is (bijv. drama en movies) en kiest de meest voorkomende (bijv. drama) 
	# - je kijkt naar de preferred genres van alle users en kijkt welk genre het meest genoemd worden samen met drama (bijv. comedy) 
	# - dan kijk je naar alle content met het genre comedy EN die niet Roxanne is 
	# - en hieruit recommend je de 10 titels die de laagste cosine similarity hebben met de originele titel (met Roxanne dus)
	"""
	Applying the principle of finding dissimilar content within a similar type of content.
	E.g., finding dissimilar content (to the user's favorite) within a genre similar to the user's favorite genre.
	Content has a genre and it has content info. The diverse recommendation's genre is similar, but its content info is dissimilar.
	- We could also turn this around: find content with similar content info but a dissimilar genre
	- A user has preferences and demographics. Maybe we can find users with similar demographics but very different preferences, or vice versa,
	and then recommend these users' favorites.
	"""

	user = user_data.iloc[user_index]

	user_faves = user["top10"]

	if sim_genre:
		list_of_dfs = []

		for fave in user_faves:

			if fave not in content_data["title"].unique():
				continue

			# get all genres associated with this title
			genre_list = content_data[content_data["title"] == fave]["genre"].sum()

			# get this title's genre (by checking which one is most common)
			current_genre = Counter(genre_list).most_common(1)[0][0]

			# get other genres that are preferred by users that prefer the current genre, i.e., similar genres
			genre_in_list_mask = user_data["pref_genre"].apply(lambda genre_list : True if current_genre in genre_list else False)
			similar_genres_list = user_data[genre_in_list_mask]["pref_genre"].sum()

			# prevent crashing in case a genre has not been picked by any user as preferred genre
			if similar_genres_list == 0:
				continue

			# remove all occurrences of the current genre in the list
			similar_genres_list = [genre for genre in similar_genres_list if genre != current_genre]

			# get the genre most commonly preferred alongside the current genre
			most_similar_genre = Counter(similar_genres_list).most_common(1)[0][0] # if there are some with same most common count -> chooses random I think # TODO: 2 most common genres?

			# get all content with the similar genre that is not the user's favorite
			content_has_genre = content_data["genre"].apply(lambda genre_list : True if most_similar_genre in genre_list else False) 
			similar_genre_content = content_data[content_has_genre & (content_data["title"] != fave)]

			# sum all similar content data into a column of sets
			content_info_col = similar_genre_content[["genre", "tags", "tags2", "more", "rating"]].sum(axis=1).apply(set)

			# sum all content data for the fave in a set
			fave_info = set(content_data[content_data["title"] == fave].iloc[0][["genre", "tags", "tags2", "more", "rating"]].sum())

			# compute similarity between all sets of similar content and the fave set
			similar_genre_content["jac_similarity"] = content_info_col.apply(lambda content_info_list : jaccard_similarity(fave_info, content_info_list))

			# concat all dfs (don't remove dupes bc you want the jac score for all comparisons)
			list_of_dfs.append(similar_genre_content)

		if len(list_of_dfs) == 0:
			user_recs_sim_content_df = content_data.sample(n=20) # TODO: pick 20 random rows from content data? Or some other way to fix it if a person only has 1 genre in their faves. Limitations! Maybe still somehow use jaccard sim?
		else:
			user_recs_sim_content_df = pd.concat(list_of_dfs, ignore_index=True)
		
			# sort the content from least to most similar
			user_recs_sim_content_df = user_recs_sim_content_df.sort_values(by=["jac_similarity"])
			# TODO: remove duplicates here? Cause they are in there

	else:
		user_recs_sim_content_df = pd.DataFrame([])
	
	# find users in your demographics cluster and recommend their favorites that are least similar to your favorites
	if sim_demo:
		list_of_dfs = []

		for fave in user_faves:

			if fave not in content_data["title"].unique():
				continue

			# find the assigned cluster for this user
			user_cluster = user["sim_demo_labels"]

			# find other users assigned to this cluster
			cluster_members = user_data[(user_data["sim_demo_labels"] == user_cluster) & (user_data["key"] != user["key"])]

			if cluster_members.empty:
				user_recs_sim_demo_df = pd.DataFrame([])

			else: 

				# add these users' favorite titles to the recommendations if they are not in this user's favorites already
				cluster_member_faves = set(cluster_members["top10"].sum()) # TODO: returns 0 in some cases?
				new_titles = list(cluster_member_faves.difference(user_faves))

				### TODO: write function for this?
				# get the content data for these titles
				similar_demo_content = content_data[content_data["title"].isin(new_titles)]

				# sum all content data into a column of sets
				content_info_col = similar_demo_content[["genre", "tags", "tags2", "more", "rating"]].sum(axis=1).apply(set)

				# sum all content data for the fave in a set
				fave_info = set(content_data[content_data["title"] == fave].iloc[0][["genre", "tags", "tags2", "more", "rating"]].sum())

				# compute similarity between all sets in similar content and the fave set
				similar_demo_content["jac_similarity"] = content_info_col.apply(lambda content_info_list : jaccard_similarity(fave_info, content_info_list))

				# concat all dfs (don't remove dupes bc you want the jac score for all comparisons)
				list_of_dfs.append(similar_demo_content)

		if len(list_of_dfs) == 0:
			user_recs_sim_demo_df = content_data.sample(n=20)
		else:
			user_recs_sim_demo_df = pd.concat(list_of_dfs, ignore_index=True)
				
			# sort the content from least to most similar
			user_recs_sim_demo_df = user_recs_sim_demo_df.sort_values(by=["jac_similarity"])
			# TODO: remove duplicates here? Cause they are in there

	else:
		user_recs_sim_demo_df = pd.DataFrame([])

	# find users in your similar favorites cluster and recommend their favorites that are least similar to your favorites
	if sim_favorites:
		list_of_dfs = []

		for fave in user_faves:

			if fave not in content_data["title"].unique():
				continue

			# find the assigned cluster for this user
			user_cluster = user["sim_fav_labels"]

			# find other users assigned to this cluster
			cluster_members = user_data[(user_data["sim_fav_labels"] == user_cluster) & (user_data["key"] != user["key"])]

			if cluster_members.empty:
				user_recs_sim_faves_df = pd.DataFrame([])
			else:

				# get the cluster members' favorites that are not in the user's favorites
				cluster_member_faves = set(cluster_members["top10"].sum())

				new_titles = list(cluster_member_faves.difference(user_faves))

				### TODO: write function for this?
				# get the content data for these titles
				similar_fav_content = content_data[content_data["title"].isin(new_titles)]

				# sum all content data into a column of sets
				content_info_col = similar_fav_content[["genre", "tags", "tags2", "more", "rating"]].sum(axis=1).apply(set)

				# sum all content data for the fave in a set
				fave_info = set(content_data[content_data["title"] == fave].iloc[0][["genre", "tags", "tags2", "more", "rating"]].sum())

				# compute similarity between all sets in similar content and the fave set
				similar_fav_content["jac_similarity"] = content_info_col.apply(lambda content_info_list : jaccard_similarity(fave_info, content_info_list))

				# concat all dfs (don't remove dupes bc you want the jac score for all comparisons)
				list_of_dfs.append(similar_fav_content)

		if len(list_of_dfs) == 0:
			user_recs_sim_faves_df = content_data.sample(n=20)

		else:
			user_recs_sim_faves_df = pd.concat(list_of_dfs, ignore_index=True)

			# sort the content from least to most similar
			user_recs_sim_faves_df = user_recs_sim_faves_df.sort_values(by=["jac_similarity"])
			# TODO: remove duplicates here? Cause they are in there

	else:
		user_recs_sim_faves_df = pd.DataFrame([])

	return user_recs_sim_content_df, user_recs_sim_demo_df, user_recs_sim_faves_df

	# find users in your preference cluster and recommend their favorites that are least similar to your favorites

	# find users in the cluster that is furthest away from your demographics/preference/favorites cluster (hard to say with kmodes probably) 
	# and recommend their favorites that are most similar to yours (idk if this is possible)

def main():
	### We can combine files in single file -> no double time/efficiency costs (loading data, clustering)

	# load nikki's pickled dataframe so the convert_to_list line can be removed
	user_data = pd.read_pickle("data/ABC/final_users")
	content_data = pd.read_csv("data/ABC/programs_abc.csv")
	content_data = content_data.drop_duplicates() 
	user_data, content_data = data_preprocessing(user_data, content_data)
	


	# TODO: only do these clusterings if the keyword is true
	# cluster users based on demographics (results in very few recommendations)
	users_with_labels = k_prototypes_recs(user_data, determine_best_K=False)

	# TODO: what to do when person is alone in a cluster?

	# cluster users based on favorites
	users_with_labels = k_modes_recs(users_with_labels,
					   							content_data,
												sim_var="similar_favorites",
												determine_best_K=False)

	# cluster content 
	content_with_labels = k_modes_recs(user_data, content_data, sim_var="similar_content") 

	# convert rating column into a list of a single string so we can sum the df values
	content_with_labels["rating"] = content_with_labels["rating"].apply(lambda rating : rating.split(sep=None))

	# get recommendations for a random user
	user_index = random.randint(0, len(user_data) - 1)
	user_recs = get_diverse_recommendations(user_index, users_with_labels, content_with_labels)
	print(user_recs)

	print(len(user_recs[0]), len(user_recs[1]), len(user_recs[2]))
	# 2427 3390 5930
	# 1112 4030 5770
	# 2427 1910 5740
	# 1106 3390 5860
	# 1240 3520 5710

	# Limitations: clustering users on demo results in 3 or 4 clusters, meaning that many of the recommendations are equal for people
	# and that some people can be alone or with one other person in their cluster
	# Maybe it could have been solved if location was in coordinates

if __name__ == "__main__":
   main()