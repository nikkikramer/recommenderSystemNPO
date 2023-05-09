from all_recommendations import *
from warnings import simplefilter
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# load data
user_data = pd.read_pickle("data/ABC/final_users")
content_data = pd.read_csv("data/ABC/programs_abc.csv")
content_data = content_data.drop_duplicates() 
user_data, content_data = data_preprocessing(user_data, content_data)

# create empty lists for the ILS scores
ils_normal_recs = []
ils_diverse_recs = []

for i in tqdm(range(30), desc="i"):
	user_index = random.randint(0, len(user_data) - 1)

	# get normal recommendations for a random user
	recs_1, recs_2, recs_3, recs_4, recs_5 = get_recommendations(user_index, user_data, content_data)

	# concatenate the first 3 rows of each recommendation data frame together -> 15 rows in total
	all_recs_df = pd.concat([recs_1.iloc[:3], recs_2.iloc[:3], recs_3.iloc[:3], recs_4.iloc[:3], recs_5.iloc[:3]], ignore_index=True)

	# get diverse recommendations for a random user
	content_with_labels, users_with_labels = get_df_with_labels()
	diverse_recs_1, diverse_recs_2, diverse_recs_3 = get_diverse_recommendations(user_index, users_with_labels, content_with_labels)

	# concatenate the first 5 rows of each diverse recommendation data frame together -> 15 rows in total
	all_diverse_recs_df = pd.concat([diverse_recs_1.iloc[:5], diverse_recs_2.iloc[:5], diverse_recs_3.iloc[:5]], ignore_index=True)

	# create data frames for the normal and diverse recommendations
	one_hot_recs = pd.DataFrame({"title":all_recs_df["title"].values})
	one_hot_diverse_recs = pd.DataFrame({"title":all_diverse_recs_df["title"].values})

	# loop over the content data columns in the recommendations
	for col in ["genre", "tags", "tags2", "more", "rating", "title_type"]:
		
		# get all unique values for this column in both the normal and diverse recommendations
		unique_values = list(set(all_recs_df[col].sum() + all_diverse_recs_df[col].sum()))
		
		# loop over the unique values
		for unique_val in unique_values:
			# create a one-hot encoded matrix for the normal recommendations
			one_hot_recs[f"{col}_{unique_val}"] = all_recs_df[col].apply(lambda item_list : 1 if unique_val in item_list else 0)
			
			# create a one-hot encoded matrix for the diverse recommendations
			one_hot_diverse_recs[f"{col}_{unique_val}"] = all_diverse_recs_df[col].apply(lambda item_list : 1 if unique_val in item_list else 0)
				
	# set title column as index
	one_hot_recs = one_hot_recs.set_index("title")
	one_hot_diverse_recs = one_hot_diverse_recs.set_index("title")

	# compute intra-list similarity (avg cosine similarity) for normal recommendations
	cos_sim = cosine_similarity(one_hot_recs)
	ils_normal = np.mean(cos_sim)
	ils_normal_recs.append(ils_normal)

	# compute intra-list similarity (avg cosine similarity) for diverse recommendations
	diverse_cos_sim = cosine_similarity(one_hot_diverse_recs)
	ils_diverse = np.mean(diverse_cos_sim)
	ils_diverse_recs.append(ils_diverse)

	# store the first iteration in .csv files
	if i == 0:
		cos_sim = np.round(cos_sim, decimals=2)
		normal_titles = all_recs_df["title"].values
		pairwise_cos_sim_df = pd.DataFrame(cos_sim, index=normal_titles, columns=normal_titles)
		pairwise_cos_sim_df.to_csv("plots/cos_sim_pairwise.csv")

		diverse_cos_sim = np.round(diverse_cos_sim, decimals=2)
		diverse_titles = all_diverse_recs_df["title"].values
		pairwise_diverse_cos_sim_df = pd.DataFrame(diverse_cos_sim, index=diverse_titles, columns=diverse_titles)
		pairwise_diverse_cos_sim_df.to_csv("plots/cos_sim_diverse_pairwise.csv")

# plot ILS for normal recommendations
plt.hist(ils_normal_recs)
ils_normal_mean = np.round(np.mean(ils_normal_recs), 2)
plt.axvline(np.mean(ils_normal_recs), color='k', linestyle='dashed', linewidth=1)
plt.title(f"Histogram of Intra-List-Similarity\nfor normal recommendations (M={ils_normal_mean})")
plt.xlabel("ILS")
plt.savefig("plots/histogram_normal_ILS.png")

# plot ILS for diverse recommendations
plt.hist(ils_diverse_recs)
ils_diverse_mean = np.round(np.mean(ils_diverse_recs), 2)
plt.axvline(np.mean(ils_diverse_recs), color='k', linestyle='dashed', linewidth=1)
plt.title(f"Histogram of Intra-List-Similarity\nfor diverse recommendations (M={ils_diverse_mean})")
plt.xlabel("ILS")
plt.savefig("plots/histogram_diverse_ILS.png")
