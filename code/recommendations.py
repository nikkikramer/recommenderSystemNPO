
# TODO: requirements.txt
# TODO: functiecomments + gewone comments
# TODO: check welke libraries je daadwerkelijk gebruikt

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

def data_preprocessing(user_df, content_df):
	"""
	"""
	content_df = content_df.replace(np.nan, "") # TODO: fillna?

	# turn DOB (yyyy-mm-dd) into age 
	user_df["age"] = user_df["dob"].apply(calculate_age)

	# replace No preferences TODO
	user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type"]] = replace_no_pref(user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type"]])

	# convert columns of list strings to columns of lists
	# user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type", "top10"]] = convert_to_list(user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type", "top10"]])
	content_df[["tags", "tags2", "genre"]] = convert_to_list(content_df[["tags", "tags2", "genre"]])
	content_df["more"] = content_df["more"].str.replace("Hosts|Host", "", regex=True).str.strip().str.split(", ")

	# alphabetize for kmodes
	user_df["pref_genre"] = user_df["pref_genre"].apply(lambda fav_list : sorted(fav_list))
	user_df["pref_rating"] = user_df["pref_rating"].apply(lambda fav_list : sorted(fav_list))
	user_df["top10"] = user_df["top10"].apply(lambda fav_list : sorted(fav_list))
	user_df["pref_title_type"] = user_df["pref_title_type"].apply(lambda fav_list : sorted(fav_list))

	content_df["tags"] = content_df["tags"].apply(lambda fav_list : sorted(fav_list))
	content_df["tags2"] = content_df["tags2"].apply(lambda fav_list : sorted(fav_list))
	content_df["genre"] = content_df["genre"].apply(lambda fav_list : sorted(fav_list))
	content_df["more"] = content_df["more"].apply(lambda fav_list : sorted(fav_list))

	# expand these columns of lists to separate columns
	user_df = expand_column(user_df, "pref_genre")
	user_df = expand_column(user_df, "pref_rating")
	user_df = expand_column(user_df, "pref_years")
	user_df = expand_column(user_df, "top10")
	user_df = expand_column(user_df, "pref_title_type")

	content_df = expand_column(content_df, "tags")
	content_df = expand_column(content_df, "tags2")
	content_df = expand_column(content_df, "genre")
	content_df = expand_column(content_df, "more")

	return user_df, content_df

def determine_K(data, algorithm="KModes", K_range=100, sim_var=""):
	"""
	Determine optimal K with knee plot.
	"""

	costs = []

	K = range(1, K_range)

	for k in K:
		print("K: ", k)

		if algorithm == "KModes": 
			kmodel = KModes(n_clusters=k, init = "random", n_init = 5, verbose=1)
			kmodel.fit_predict(data)

		elif algorithm == "KPrototypes":
			kmodel = KPrototypes(n_clusters=k, init='Cao') # TODO: Other init? Use other args like with KModes?

			kmodel.fit_predict(data, categorical=[0, 1])
		
		costs.append(kmodel.cost_)

	kn = KneeLocator(K, costs, curve='convex', direction='decreasing')

	# ensure the program doesn't crash if no knee is found
	K_optimal = kn.knee if kn.knee else 5 #TODO: onderbouwen?

	if sim_var == "similar_favorites":
		plotname_suffix = "_cluster_users"
	elif sim_var == "similar_demographics":
		plotname_suffix = "_cluster_users_dem"
	elif sim_var == "similar_preferences":
		plotname_suffix = "_cluster_users_pref"		
	elif sim_var == "similar_content":
		plotname_suffix = "_cluster_content"
	else:
		plotname_suffix = ""
		
	plt.plot(K, costs, 'bx-')
	plt.xlabel('No. of clusters')
	plt.ylabel('Cost')
	plt.title(f'Knee Plot for {algorithm} (K = {K_optimal})')
	plt.vlines(K_optimal, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
	plt.savefig(f"Knee_plot_{algorithm}{plotname_suffix}.png")
	plt.show()

	return K_optimal

def k_modes_recs(user_df, content_df, determine_best_K=False, sim_var="similar_favorites"):
	"""
	"KModes clustering is one of the unsupervised Machine Learning algorithms 
	that is used to cluster categorical variables." 
	https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/#:~:text=KModes%20clustering%20is%20one%20of,distance)%20to%20cluster%20continuous%20data. )
	"""

	if sim_var == "similar_content":
		data = content_df[[col for col in content_df if col.startswith(("genre_", "tags_", "tags2_", "more_", "rating"))]]

		if determine_best_K: 
			K_optimal = determine_K(data, sim_var=sim_var) # running this w/ K_range = 100 takes 9.5 h
			print("K_optimal = ", K_optimal)
		else:
			K_optimal = 10 # updated and done, bc we don't change this data

		km = KModes(n_clusters=K_optimal, init="random", n_init = 5, verbose=1) #TODO: random_state?
		cluster_labels = km.fit_predict(data, sim_var = "similar_content")
		content_df["sim_content_labels"] = cluster_labels
		
		return content_df

	elif sim_var == "similar_favorites":
		data = user_df[[f"top10_{i}" for i in range(10)]]

		if determine_best_K: # this takes 2 seconds
			K_range = round((data.shape[0]) / 10)

			K_optimal = determine_K(data, K_range=K_range, sim_var=sim_var)
			print("K_optimal = ", K_optimal)

		else:
			K_optimal = 4 # or 2? or 4? K_range = 100 -> K = 5
			# TODO: this is too unstable -> We've now chosen K=4 because that gives us more recommendations for the diversity. However, this also means that recommendations are more similar to eachother

		km = KModes(n_clusters=K_optimal, init = "random", n_init = 5, verbose=1) #TODO: random_state?
		cluster_labels = km.fit_predict(data)
		user_df["sim_fav_labels"] = cluster_labels
		
		return user_df
	
	elif sim_var == "similar_preferences":
		data = user_df[[col for col in user_df if col.startswith(("pref_genre_", "pref_rating_", "pref_years_"))]]

		if determine_best_K: # this takes 5 minutes
			K_range = round((data.shape[0]) / 10)

			K_optimal = determine_K(data, K_range=K_range, sim_var=sim_var)
			print("K_optimal = ", K_optimal)

		else:
			K_optimal = 5 # updated, very stable

		km = KModes(n_clusters=K_optimal, init = "random", n_init = 5, verbose=1) #TODO: random_state?
		cluster_labels = km.fit_predict(data)
		user_df["sim_pref_labels"] = cluster_labels

		return user_df

def k_prototypes_recs(user_df, determine_best_K=False, sim_var="similar_demographics"):
	"""
	"""

	data = user_df[["gender", "location", "age"]]

	if determine_best_K: # for K_range = 10, this takes 2 seconds
		K_range = round((data.shape[0]) / 10)
		K_optimal = determine_K(data, algorithm="KPrototypes", K_range=K_range, sim_var=sim_var) 
		print("K_optimal = ", K_optimal)

	else:
		K_optimal = 4 # updated, very stable

	# only 1 numeric feature (age)  -> no need to normalize continuous features 

	kp = KPrototypes(n_clusters=K_optimal, init='Cao') # TODO: Other init? Use other args like with KModes?
	user_df["sim_demo_labels"] = kp.fit_predict(data, categorical=[0, 1])

	return user_df

def get_sim_faves_recs(user, user_df_with_labels):
	"""
	"""

	user_faves = user["top10"]

	# find the assigned cluster for this user
	user_cluster = user["sim_fav_labels"]

	# find other users assigned to this cluster
	cluster_members = user_df_with_labels[(user_df_with_labels["sim_fav_labels"] == user_cluster) & (user_df_with_labels["key"] != user["key"])]

	if cluster_members.empty:
		new_titles = set()

	else: 

		# recommend these users' favorite titles if they are not in the user's favorites already
		cluster_member_faves = set(cluster_members["top10"].sum())
		new_titles = cluster_member_faves.difference(user_faves)

	return new_titles

def get_sim_content_recs(user, content_df_with_labels):
	"""
	"""

	user_faves = user["top10"]

	user_recs = []

	# loop over this user's favorite titles
	for fave in user_faves:

		# TODO because in the user data there is 'bangarra's world' without capital B and Mythbusters without the rest of the title
		if fave not in content_df_with_labels["title"].unique():
			continue

		# find the assigned cluster for this title
		fave_cluster_labels = content_df_with_labels[content_df_with_labels["title"] == fave]["sim_content_labels"]
		fave_cluster = fave_cluster_labels.mode().values[0] # use the most common cluster for this title

		# find other content assigned to this cluster
		cluster_members = content_df_with_labels[(content_df_with_labels["sim_content_labels"] == fave_cluster) \
					   							& (content_df_with_labels["title"] != fave)]
		
		if cluster_members.empty:
			continue

		else: 

			# add unique titles to the recommendations
			user_recs.extend(list(set(cluster_members.title.values)))

	user_recs = set(user_recs)

	return user_recs

def get_sim_prefs_recs(user, user_df_with_labels):

	user_faves = user["top10"]

	# find the assigned cluster for this user
	user_cluster = user["sim_pref_labels"]

	# find other users assigned to this cluster
	cluster_members = user_df_with_labels[(user_df_with_labels["sim_pref_labels"] == user_cluster) & (user_df_with_labels["key"] != user["key"])]

	if cluster_members.empty:
		user_recs = set()

	else: 

		# add these users' favorite titles to the recommendations if they are not in this user's favorites already
		cluster_member_faves = set(cluster_members["top10"].sum())

		new_titles = cluster_member_faves.difference(user_faves)

		user_recs = new_titles
	
	return user_recs

def get_sim_demo_recs(user, user_df_with_labels):
	"""
	"""

	user_faves = user["top10"]

	# find the assigned cluster for this user
	user_cluster = user["sim_demo_labels"]

	# find other users assigned to this cluster
	cluster_members = user_df_with_labels[(user_df_with_labels["sim_demo_labels"] == user_cluster) & (user_df_with_labels["key"] != user["key"])]

	if cluster_members.empty:
		user_recs = set()

	else: 

		# add these users' favorite titles to the recommendations if they are not in this user's favorites already
		cluster_member_faves = set(cluster_members["top10"].sum())
		new_titles = cluster_member_faves.difference(user_faves)

		user_recs = new_titles

	return user_recs

def get_most_faved_recs(user, title_count_per_genre, N=15):
	"""
	"""
	# Limitations: deze rec type geeft de minste recs omdat het ligt aan de hoeveelheid user data

	# for the current user, loop over all its favorite genres
	user_fave_genres = user["pref_genre"]
	user_recs = []

	for fav_genre in user_fave_genres:
		# Limitations: because we have relatively little user data, some users don't get many recommendations and there is not much we can do about it.
		# E.g., people with Music in their favorite genres, will only get a couple of recommendations
		
		# find top N most favorited titles for this favorite genre
		genre_in_list_mask = title_count_per_genre["genre"].apply(lambda genre_list : True if fav_genre in genre_list else False)
		most_favorited_titles = title_count_per_genre[genre_in_list_mask].head(N)["top10"].values

		# add these titles to user_recs
		user_recs.extend(most_favorited_titles)

	return user_recs

def calculate_age(date_str):
	"""
	Convert date of birth into age in years.
	Source: partially from https://www.geeksforgeeks.org/python-program-to-calculate-age-in-year/https://www.geeksforgeeks.org/python-program-to-calculate-age-in-year/
	"""
	
	birth_date = datetime.strptime(date_str, '%Y-%m-%d')
	
	today = date.today()
	
	age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))

	return age

def expand_column(df, col_name):
	"""
	max prefs: max number of genres/ratings... you can choose
	"""
	df_col = df[col_name]

	max_prefs = df_col.str.len().max()
	
	# add empty strings for as many as the max number of genres/ratings... you can choose
	df_col = df_col.apply(lambda list_ : list_ + ["" for _ in range(max_prefs - len(list_))])
	df[[f"{col_name}_{i}" for i in range(max_prefs)]] = pd.DataFrame(df_col.tolist(), index=df.index)

	return df

def convert_to_list(df_subset):
	# TODO: make this obsolete
	for col_name in df_subset:
		df_subset[col_name] = df_subset[col_name].replace("", "[]")
		df_subset[col_name] = df_subset[col_name].apply(lambda list_string: ast.literal_eval(list_string))
	return df_subset

def replace_no_pref(pref_columns):
	for pref_col in pref_columns:
		pref_columns[pref_col] = pref_columns[pref_col].apply(lambda pref_list : [] if pref_list[0] == 'No preference' else pref_list)
	return pref_columns

def get_content(user, list_of_titles, content_df):
	"""
	Get content rows from content_df based on list of titles and ...
	"""
	unfiltered_content = content_df[content_df["title"].isin(list_of_titles)].drop_duplicates(["title"])
	print("unfiltered_content:")
	print(unfiltered_content)

	# filter the content based on user preferences
	filtered_content = unfiltered_content.copy()

	user_rating = user["pref_rating"]
	print("user_rating: ", user_rating)
	print("filtered_content[rating]:")
	print(filtered_content["rating"].values)
	if user_rating: # if the user has preferred rating(s)
		filtered_content = filtered_content[filtered_content["rating"].isin(user_rating)]
		print("filtered_content after rating:")
		print(filtered_content)

	user_title_type = user["pref_title_type"]
	print("user_title_type: ", user_title_type)
	print("filtered_content[title_type]:")
	print(filtered_content["title_type"].values)
	if user_title_type: # if the user has preferred title type(s)
		filtered_content = filtered_content[filtered_content["title_type"].isin(user_title_type)]
		print("filtered_content after title type:")
		print(filtered_content)

	if user["pref_years"]: # if the user has preferred publication year(s)
		print("pref_years: ", user["pref_years"])
		print("filtered_content[publication_date]:")
		print(filtered_content["publication_date"].values)
		user_date_from, user_date_to = user["pref_years"]
		publication_year = filtered_content["publication_date"].str[:4].astype('int')
		filtered_content = filtered_content[(publication_year >= user_date_from) & (publication_year <= user_date_to)]
		print("filtered_content after pref date:")
		print(filtered_content)

	user_genre = user["pref_genre"]
	print("user_genre: ", user_genre)
	print("filtered_content[genre]:")
	print(filtered_content["genre"].values)
	if user_genre: # if the user has preferred genre(s)
		overlapping_genres = filtered_content["genre"].apply(lambda genre_list : True if len(set(user_genre).intersection(set(genre_list))) > 0 else False)
		filtered_content = filtered_content[overlapping_genres]
		print("filtered_content after genre:")
		print(filtered_content)

	# check if filtering yields any content
	if filtered_content.empty:
		return unfiltered_content
	
	else:
		# shuffle the unfiltered content randomly
		unfiltered_content = unfiltered_content.sample(frac=1)#.reset_index()

		print("filtered_content:")
		print(filtered_content)

		print("unfiltered_content:")
		print(unfiltered_content)

		# concatenate filtered and unfiltered content and remove all rows that are in filtered content
		new_content = pd.concat([filtered_content, unfiltered_content]).drop_duplicates(["title"], keep=False)
		
		return new_content

def get_recommendations(user_index, user_df, content_df, sim_favorites=True, sim_content=True, most_favorited_content=True, sim_demographics=True, sim_preferences=True):
	"""
	"""

	# cluster users based on their favorites using kmodes
	if sim_favorites: 
		users_with_sim_fav_labels = k_modes_recs(user_df, 
													content_df, 
													sim_var="similar_favorites",
													determine_best_K=False)

	# cluster content using kmodes
	if sim_content: 
		content_with_labels = k_modes_recs(user_df, 
											content_df, 
											sim_var="similar_content",
											determine_best_K=False)
		
	# ...
	if most_favorited_content: 
		# count how often each title was favorited by users
		favorite_counts = pd.Series(user_df["top10"].sum()).value_counts().to_frame().reset_index()
		favorite_counts.columns = ["top10", "count"]

		# get each title's favorited count and genre
		title_count_per_genre = favorite_counts.merge(content_df[["genre", "title"]], 
														how="left", 
														left_on="top10", 
														right_on="title")[["top10", "count", "genre"]]
		title_count_per_genre = title_count_per_genre.dropna() # removes 3 titles that weren't spelled correctly TODO: fix in csv file!

	# cluster users based on their demographics using kprototypes
	if sim_demographics: # mixed data types -> K-Prototypes bc it's better than using one-hot encoding (https://medium.com/analytics-vidhya/clustering-on-mixed-data-types-in-python-7c22b3898086)
		users_with_sim_demo_labels = k_prototypes_recs(user_df,
																determine_best_K=False)

	# cluster users based on their preferences using kmodes
	if sim_preferences: 
		# we're using kmodes because K-Prototypes wouldn't work with the sparse ratings and genres
		# Limitations -> years are integers but still we use k-modes

		users_with_sim_pref_labels = k_modes_recs(user_df, 
															content_df, 
															sim_var="similar_preferences",
															determine_best_K=False)

	# get current user's recommendations
	user = user_df.iloc[user_index]

	# get recs for every user based on users with similar favorite titles
	if sim_favorites:
		user_recs_sim_faves = get_sim_faves_recs(user, users_with_sim_fav_labels) 
		user_recs_sim_faves_df = get_content(user, user_recs_sim_faves, content_df)
	else:
		user_recs_sim_faves_df = pd.DataFrame([])

	# get recs for every user based on content similar to the user's favorites
	if sim_content:
		user_recs_sim_content = get_sim_content_recs(user, content_with_labels)
		user_recs_sim_content_df = get_content(user, user_recs_sim_content, content_df)
	else:
		user_recs_sim_content_df = pd.DataFrame([])

	# get recs for every user based on the most favorited content from their preferred genre(s) 
	if most_favorited_content:
		user_recs_fav_content = get_most_faved_recs(user, title_count_per_genre)
		user_recs_fav_content_df = get_content(user, user_recs_fav_content, content_df)
	else:
		user_recs_fav_content_df = pd.DataFrame([])

	# get recs for every user based on users with similar demographics
	if sim_demographics:
		user_recs_sim_demo = get_sim_demo_recs(user, users_with_sim_demo_labels) 
		user_recs_sim_demo_df = get_content(user, user_recs_sim_demo, content_df)
	else:
		user_recs_sim_demo_df = pd.DataFrame([])

	# get recs for every user based on users with similar preferences
	if sim_preferences:
		user_recs_sim_pref = get_sim_prefs_recs(user, users_with_sim_pref_labels) 
		user_recs_sim_pref_df = get_content(user, user_recs_sim_pref, content_df)
	else:
		user_recs_sim_pref_df = pd.DataFrame([])	

	return user_recs_sim_faves_df, user_recs_sim_content_df, user_recs_fav_content_df, user_recs_sim_demo_df, user_recs_sim_pref_df
	



def main():

	# load nikki's pickled dataframe so the convert_to_list line can be removed
	user_data = pd.read_pickle("data/ABC/final_users")
	content_data = pd.read_csv("data/ABC/programs_abc.csv")
	content_data = content_data.drop_duplicates() 
	user_data, content_data = data_preprocessing(user_data, content_data)

	# get recommendations for a random user
	user_index = random.randint(0, len(user_data) - 1)
	recs = get_recommendations(user_index, user_data, content_data) 
		  

	print(f"Recommendations for user {user_index}: ")
	print("based on users with similar favorite titles")
	print(recs[0])
	print("based on content similar to the user's favorites:")
	print(recs[1])
	print("based on the most favorited content from their preferred genre(s):")
	print(recs[2])
	print("based on users with similar demographics:")
	print(recs[3])
	print("based on users with similar preferences:")
	print(recs[4])

	print(len(recs[0]), len(recs[1]), len(recs[2]), len(recs[3]), len(recs[4]))


	# met eerst de filter_prefs en dan zonder
	# 570, 639, 8, 331, 290
	# 570, 598, 9, 353, 292
	# 565, 377, 7, 107, 111
	# 579, 229, 2, 108, 187
	# 594, 739, 14, 231, 280



if __name__ == "__main__":
   main()




# TODO: al Nikki's ideeen uitwerken (random picks, top 10 most favorited movies etc.) (lage prio)
# TODO: location in coordinates?
# TODO: alles naar lowercase zodat je zeker weet dat het matcht
# TODO Nikki: rename variables top5 -> top10, age -> dob, genre -> pref_genre, prefDate -> pref_date, rating -> pref_rating, title_type -> pref_title_type
# TODO Nikki: zorgen dat in interface de first name update als je deze wijzigt
# TODO Nikki: genre column in content df bevat soms 2x zelfde genre, bijv rij 12389 (Spartacus) heeft 2x arts & culture 
# TODO Nikki: zorg dat zij voor de 4 clustering algoritmen N RANDOM titels pakt ipv de bovenste N
# TODO after we have enough user data: run sim_faves with determine best K + update K_optimal + update K_range
