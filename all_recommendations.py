
import ast
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from collections import Counter
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from kneed import KneeLocator
import random
from datetime import datetime, date
pd.options.mode.chained_assignment = None


def data_preprocessing(user_df, content_df):
    """
    Takes the user data frame and the content data frame and cleans these,
    such that they can be used to generate recommendations with.
    """
    content_df = content_df.fillna("")

    # turn DOB (yyyy-mm-dd) into age
    user_df["age"] = user_df["dob"].apply(calculate_age)

    # replace 'No preference'
    user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type"]] = replace_no_pref(
        user_df[["pref_genre", "pref_rating", "pref_years", "pref_title_type"]])

    # remove functional names
    content_df["more"] = content_df["more"].str.replace("Cast", "")
    content_df["more"] = content_df["more"].str.replace("Director", "")
    content_df["more"] = content_df["more"].str.replace(
        "Hosts|Host", "", regex=True)
    content_df["more"] = content_df["more"].str.replace("Narrator", "")

    # convert columns of strings into columns of single-elemt lists
    content_df["rating"] = content_df["rating"].apply(
        lambda rating: rating.split(sep=None))
    content_df["title_type"] = content_df["title_type"].apply(lambda title_type: title_type.split(
        sep=None))

    # convert columns of list strings to columns of lists
    content_df[["tags", "tags2", "genre"]] = convert_to_list(
        content_df[["tags", "tags2", "genre"]])
    content_df["more"] = content_df["more"].str.strip().str.split(", ")

    # alphabetize user data for kmodes
    user_df["pref_genre"] = user_df["pref_genre"].apply(
        lambda fav_list: sorted(fav_list))
    user_df["pref_rating"] = user_df["pref_rating"].apply(
        lambda fav_list: sorted(fav_list))
    user_df["top10"] = user_df["top10"].apply(
        lambda fav_list: sorted(fav_list))
    user_df["pref_title_type"] = user_df["pref_title_type"].apply(
        lambda fav_list: sorted(fav_list))

    # alphabetize content data for kmodes
    content_df["tags"] = content_df["tags"].apply(
        lambda fav_list: sorted(fav_list))
    content_df["tags2"] = content_df["tags2"].apply(
        lambda fav_list: sorted(fav_list))
    content_df["genre"] = content_df["genre"].apply(
        lambda fav_list: sorted(fav_list))
    content_df["more"] = content_df["more"].apply(
        lambda fav_list: sorted(fav_list))

    # expand user data columns of lists into separate columns
    user_df = expand_column(user_df, "pref_genre")
    user_df = expand_column(user_df, "pref_rating")
    user_df = expand_column(user_df, "pref_years")
    user_df = expand_column(user_df, "top10")
    user_df = expand_column(user_df, "pref_title_type")

    # expand content data columns of lists into separate columns
    content_df = expand_column(content_df, "tags")
    content_df = expand_column(content_df, "tags2")
    content_df = expand_column(content_df, "genre")
    content_df = expand_column(content_df, "more")
    content_df = expand_column(content_df, "rating")
    content_df = expand_column(content_df, "title_type")

    return user_df, content_df


def determine_K(data, algorithm="KModes", K_range=100, sim_var=""):
    """
    Determine and return optimal K using a knee plot.
    """

    # create empty list to keep track of each model's cost
    costs = []

    K = range(1, K_range)

    # loop over the different values of k
    for k in tqdm(K, desc="K_range"):
        print("K: ", k)

        # run k-modes
        if algorithm == "KModes":
            kmodel = KModes(n_clusters=k, init="random", n_init=5, verbose=1)
            kmodel.fit_predict(data)

            # run k-prototypes
        elif algorithm == "KPrototypes":
            kmodel = KPrototypes(n_clusters=k, init='Cao')
            kmodel.fit_predict(data, categorical=[0, 1])

        costs.append(kmodel.cost_)

        # determine knee
    kn = KneeLocator(K, costs, curve='convex', direction='decreasing')

    # ensure the program doesn't crash if no knee is found
    K_optimal = kn.knee if kn.knee else 5

    # distinguish between plot file names
    if sim_var == "similar_favorites":
        plotname_suffix = "_cluster_users_fav"
        cluster_type = "users on top 10 favorites"
    elif sim_var == "similar_demographics":
        plotname_suffix = "_cluster_users_dem"
        cluster_type = "users on demographics"
    elif sim_var == "similar_preferences":
        plotname_suffix = "_cluster_users_pref"
        cluster_type = "users on content preferences"
    elif sim_var == "similar_content":
        plotname_suffix = "_cluster_content"
        cluster_type = "content on similar metadata"
    else:
        plotname_suffix = ""

        # create knee plot
    plt.plot(K, costs, 'bx-')
    plt.xlabel('No. of clusters')
    plt.ylabel('Cost')
    plt.title(
        f'Knee Plot for {algorithm} for clustering\n{cluster_type} (K = {K_optimal})')
    plt.vlines(K_optimal, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
    plt.savefig(f"plots/Knee_plot_{algorithm}{plotname_suffix}.png")
    plt.show()

    return K_optimal


def k_modes_recs(user_df, content_df, determine_best_K=False, sim_var="similar_favorites"):
    """
    Predict labels using the K-Modes algorithm for clustering categorical variables.
    """

    # cluster content on content characteristics
    if sim_var == "similar_content":
        data = content_df[[col for col in content_df if col.startswith(
            ("genre_", "tags_", "tags2_", "more_", "rating_", "title_type_"))]]

        # determine the optimal value of k
        if determine_best_K:
            K_optimal = determine_K(data, sim_var=sim_var)
        else:
            K_optimal = 8

            # run clustering and add labels to data frame
        km = KModes(n_clusters=K_optimal, init="random", n_init=5, verbose=1)
        cluster_labels = km.fit_predict(data, sim_var="similar_content")
        content_df["sim_content_labels"] = cluster_labels

        return content_df

        # cluster users on their favorites
    elif sim_var == "similar_favorites":
        data = user_df[[f"top10_{i}" for i in range(10)]]

        # determine the optimal value of k
        if determine_best_K:
            K_optimal = determine_K(data, sim_var=sim_var)
        else:
            K_optimal = 10

            # run clustering and add labels to data frame
        km = KModes(n_clusters=K_optimal, init="random", n_init=5, verbose=1)
        cluster_labels = km.fit_predict(data)
        user_df["sim_fav_labels"] = cluster_labels

        return user_df

        # cluster users on their preferences
    elif sim_var == "similar_preferences":
        data = user_df[[col for col in user_df if col.startswith(
            ("pref_genre_", "pref_rating_", "pref_years_"))]]

        # determine the optimal value of k
        if determine_best_K:
            K_optimal = determine_K(data, sim_var=sim_var)
        else:
            K_optimal = 19

            # run clustering and add labels to data frame
        km = KModes(n_clusters=K_optimal, init="random", n_init=5, verbose=1)
        cluster_labels = km.fit_predict(data)
        user_df["sim_pref_labels"] = cluster_labels

        return user_df


def k_prototypes_recs(user_df, determine_best_K=False, sim_var="similar_demographics"):
    """
    Predict labels using the K-Prototypes algorithm for clustering variables of 
    mixed data types.
    """

    data = user_df[["gender", "location", "age"]]

    # determine the optimal value of k
    if determine_best_K:
        K_optimal = determine_K(
            data, algorithm="KPrototypes", K_range=35, sim_var=sim_var)
    else:
        K_optimal = 4

        # run clustering and add labels to data frame
    kp = KPrototypes(n_clusters=K_optimal, init='Cao')
    user_df["sim_demo_labels"] = kp.fit_predict(data, categorical=[0, 1])

    return user_df


def get_sim_faves_recs(user, user_df_with_labels):
    """
    Takes the user data frame with the cluster labels for the users' favorites and
    recommends the current user the favorites of its cluster members.
    """

    # get this user's favorite titles
    user_faves = user["top10"]

    # find the assigned cluster for this user
    user_cluster = user["sim_fav_labels"]

    # find other users assigned to this cluster
    cluster_members = user_df_with_labels[(user_df_with_labels["sim_fav_labels"] == user_cluster) & (
        user_df_with_labels["key"] != user["key"])]

    # if the user is alone in its cluster, recommend no new titles
    if cluster_members.empty:
        new_titles = set()
    else:
        # recommend these users' favorite titles if they are not in the user's favorites already
        cluster_member_faves = set(cluster_members["top10"].sum())
        new_titles = cluster_member_faves.difference(user_faves)

    return new_titles


def get_sim_content_recs(user, content_df_with_labels):
    """
    Takes the content data frame with the cluster labels for content characteristics
    and recommends the current user...
    """

    # get this user's favorite titles
    user_faves = user["top10"]

    user_recs = []

    # loop over this user's favorite titles
    for fave in user_faves:

        # prevent crashing
        if fave not in content_df_with_labels["title"].unique():
            continue

        # find the assigned cluster for this title
        fave_cluster_labels = content_df_with_labels[content_df_with_labels["title"]
                                                     == fave]["sim_content_labels"]
        # use the most common cluster for this title
        fave_cluster = fave_cluster_labels.mode().values[0]

        # find other content assigned to this cluster
        cluster_members = content_df_with_labels[(content_df_with_labels["sim_content_labels"] == fave_cluster)
                                                 & (content_df_with_labels["title"] != fave)]

        # if the user is alone in its cluster, recommend no new titles
        if cluster_members.empty:
            continue
        else:
            # add unique titles to the recommendations
            user_recs.extend(list(set(cluster_members.title.values)))

    user_recs = set(user_recs)

    return user_recs


def get_sim_prefs_recs(user, user_df_with_labels):
    """
    Takes the user data frame with the cluster labels for the users' preferences 
    and recommends the current user the favorites of its cluster members.
    """

    # get this user's favorite titles
    user_faves = user["top10"]

    # find the assigned cluster for this user
    user_cluster = user["sim_pref_labels"]

    # find other users assigned to this cluster
    cluster_members = user_df_with_labels[(user_df_with_labels["sim_pref_labels"] == user_cluster) & (
        user_df_with_labels["key"] != user["key"])]

    # if the user is alone in its cluster, recommend no new titles
    if cluster_members.empty:
        user_recs = set()
    else:
        # add these users' favorite titles to the recommendations if they are not in this user's favorites already
        cluster_member_faves = set(cluster_members["top10"].sum())
        new_titles = cluster_member_faves.difference(user_faves)

    return new_titles

def get_sim_demo_recs(user, user_df_with_labels):
    """
    Takes the user data frame with the cluster labels for the users' demographics 
    and recommends the current user the favorites of its cluster members.
    """

    # get this user's favorite titles
    user_faves = user["top10"]

    # find the assigned cluster for this user
    user_cluster = user["sim_demo_labels"]

    # find other users assigned to this cluster
    cluster_members = user_df_with_labels[(user_df_with_labels["sim_demo_labels"] == user_cluster) & (
        user_df_with_labels["key"] != user["key"])]

    # if the user is alone in its cluster, recommend no new titles
    if cluster_members.empty:
        user_recs = set()
    else:
        # add these users' favorite titles to the recommendations if they are not in this user's favorites already
        cluster_member_faves = set(cluster_members["top10"].sum())
        new_titles = cluster_member_faves.difference(user_faves)

    return new_titles

def get_most_faved_recs(user, title_count_per_genre):
    """

    """
    # get this user's preferred genre(s)

    # for the current user, loop over all its favorite genres
    user_fave_genres = user["pref_genre"]

    user_recs = []

    # loop over this user's preferred genre(s)
    for fav_genre in user_fave_genres:

        # get the titles in this favorite genre, sorted by how often they were favorited
        genre_in_list_mask = title_count_per_genre["genre"].apply(
            lambda genre_list: True if fav_genre in genre_list else False)
        
        # since the first 15 rows are selected in the interface, this will yield the 15 most favorited titles
        most_favorited_titles = title_count_per_genre[genre_in_list_mask]["top10"].values

        # add these titles to user_recs
        user_recs.extend(most_favorited_titles)

    return user_recs

def calculate_age(date_str):
    """
    Convert date of birth into age in years. 
    """

    birth_date = datetime.strptime(date_str, '%Y-%m-%d')

    today = date.today()

    age = today.year - birth_date.year - \
        ((today.month, today.day) < (birth_date.month, birth_date.day))

    return age

def expand_column(df, col_name):
    """
    Takes a column from a data frame in which each row is a list, and transforms
    this column into the maximum number of list elements. This way, a row like
    ['a', 'b', 'c'] will now turn into three columns containing 'a', 'b', and 'c', 
    respectively.
    """

    # get the column from the data frame
    df_col = df[col_name]

    # get the maximum number of elements a row in the column contains
    max_prefs = df_col.str.len().max()

    # add empty strings for each value that is missing
    df_col = df_col.apply(lambda list_: list_ +
                          ["" for _ in range(max_prefs - len(list_))])
    
    # create new columns
    df[[f"{col_name}_{i}" for i in range(max_prefs)]] = pd.DataFrame(
        df_col.tolist(), index=df.index)

    return df

def convert_to_list(df_subset):
    """
    Converts each column in the subset of data frame columns from a string 
    representation of a list into an actual list.
    """

    # loop over columns in data frame
    for col_name in df_subset:

        # ensure empty strings will become string representations of empty list
        df_subset[col_name] = df_subset[col_name].replace("", "[]")

        # turn string representation of list into list
        df_subset[col_name] = df_subset[col_name].apply(
            lambda list_string: ast.literal_eval(list_string))
    return df_subset

def replace_no_pref(pref_columns):
    """
    Replaces any occurrence of 'No preference' in the preference columns with
    an empty list.
    """

    # loop over preference columns
    for pref_col in pref_columns:

        # replace 'No preference' with an empty list
        pref_columns[pref_col] = pref_columns[pref_col].apply(
            lambda pref_list: [] if pref_list[0] == 'No preference' else pref_list)
        
    return pref_columns

def get_content(user, data, content_df, diverse=False):
    """
    Get content rows from content_df based on list of titles and ...
    """

    # get the recommendation rows from the content data frame
    if diverse:  # data is a df of titles
        unfiltered_content = data
    else:  # data is a list of titles
        unfiltered_content = content_df[content_df["title"].isin(data)]

    # remove any titles that are duplicated
    unfiltered_content = unfiltered_content.drop_duplicates(["title"])

    # copy the unfiltered content and apply a filter based on user preferences
    filtered_content = unfiltered_content.copy()

    # only keep rows that comply to the user's preferred rating
    user_rating = user["pref_rating"]
    if user_rating: 
        filtered_content = filtered_content[filtered_content["rating_0"].isin(
            user_rating)]

    # only keep rows that comply to the user's preferred title type
    user_title_type = user["pref_title_type"]
    if user_title_type: 
        filtered_content = filtered_content[filtered_content["title_type"].isin(
            user_title_type)]

    # only keep rows that comply to the user's preferred publication year interval
    if user["pref_years"]:
        user_date_from, user_date_to = user["pref_years"]
        publication_year = filtered_content["publication_date"].str[:4].astype('int')
        filtered_content = filtered_content[(publication_year >= user_date_from) & (
            publication_year <= user_date_to)]

    # only keep rows that comply to the user's preferred genre
    user_genre = user["pref_genre"]
    if user_genre: 
        overlapping_genres = filtered_content["genre"].apply(lambda genre_list: True if len(
            set(user_genre).intersection(set(genre_list))) > 0 else False)
        filtered_content = filtered_content[overlapping_genres]

    # check if filtering yields any content
    if filtered_content.empty:
        if diverse:    
            # sort the content from least to most similar and by it being preferred content (1) or not (0)
            unfiltered_content = unfiltered_content.sort_values(by=["jac_similarity"])
        else:
            # shuffle the unfiltered content randomly
            unfiltered_content = unfiltered_content.sample(frac=1)
        return unfiltered_content

    if diverse:
        # add column that indicates whether the content complies to user preferences
        filtered_content["preferred_content"] = 1
        unfiltered_content["preferred_content"] = 0

        # concatenate filtered and unfiltered content and remove all rows that are in filtered content
        new_content = pd.concat(
            [filtered_content, unfiltered_content]).drop_duplicates(["title"])

        # sort the content from least to most similar and by it being preferred content (1) or not (0)
        new_content = new_content.sort_values(
            by=["jac_similarity", "preferred_content"], ascending=[True, False])

        # remove the newly added binary column
        new_content = new_content.drop(columns=["preferred_content"])

        return new_content

    else:
        # shuffle the unfiltered content randomly
        unfiltered_content = unfiltered_content.sample(frac=1)

        # concatenate filtered and unfiltered content and remove all rows that are in filtered content
        new_content = pd.concat(
            [filtered_content, unfiltered_content]).drop_duplicates(["title"])

        return new_content

def get_recommendations(user_index, user_df, content_df, sim_favorites=True, sim_content=True, most_favorited_content=True, sim_demographics=True, sim_preferences=True):
    """
    Returns five data frames containing recommendations of five different types,
    depending on whether any recommendations were found.

    If sim_favorites == True, the recommendations are based on the favorites of
    users with similar favorite titles, e.g. 'Roxanne'.

    If sim_content == True, the recommendations are based on content that is 
    similar to the user's favorite titles.

    If most_favorited_content == True, the recommendations are based on the 
    most favorited titles from the user's favorite genre(s).

    If sim_demographics == True, the recommendations are based on the favorites of
    users with similar demographics, e.g., age.

    If sim_preferences == True, the recommendations are based on the favorites of
    users with similar preferences, e.g., pref_genre.
    """

    # get recs for every user based on content similar to the user's favorites
    if sim_content:

        # cluster content based on metadata using kmodes
        content_with_labels = k_modes_recs(
            user_df, content_df, sim_var="similar_content")
        content_with_labels.to_pickle(f"pickles/content_with_labels.pkl")

        # get current user's recommendations
        user = user_df.iloc[user_index]

        # get recommendations
        user_recs_sim_content = get_sim_content_recs(user, content_with_labels)
        user_recs_sim_content_df = get_content(
            user, user_recs_sim_content, content_df)
    else:
        user_recs_sim_content_df = pd.DataFrame([])

    # get recs for every user based on users with similar favorite titles
    if sim_favorites:

        # cluster users based on their favorites using kmodes
        users_with_sim_fav_labels = k_modes_recs(
            user_df, content_df, sim_var="similar_favorites")
        users_with_sim_fav_labels.to_pickle(
            f"pickles/users_with_sim_fav_labels.pkl")

        # get current user's recommendations
        user = users_with_sim_fav_labels.iloc[user_index]

        # get recommendations
        user_recs_sim_faves = get_sim_faves_recs(
            user, users_with_sim_fav_labels)
        user_recs_sim_faves_df = get_content(
            user, user_recs_sim_faves, content_df)
    else:
        user_recs_sim_faves_df = pd.DataFrame([])

    # get recs for every user based on the most favorited content from their preferred genre(s)
    if most_favorited_content:

        # count how often each title was favorited by users
        favorite_counts = pd.Series(
            user_df["top10"].sum()).value_counts().to_frame().reset_index()
        favorite_counts.columns = ["top10", "count"]

        # get each title's favorited count and genre
        title_count_per_genre = favorite_counts.merge(content_df[["genre", "title"]],
                                                      how="left",
                                                      left_on="top10",
                                                      right_on="title")[["top10", "count", "genre"]]
        title_count_per_genre = title_count_per_genre.dropna()

        # get recommendations
        user_recs_fav_content = get_most_faved_recs(
            user, title_count_per_genre)
        user_recs_fav_content_df = get_content(
            user, user_recs_fav_content, content_df)
    else:
        user_recs_fav_content_df = pd.DataFrame([])

    # get recs for every user based on users with similar demographics
    if sim_demographics:

        # cluster users based on their demographics using kprototypes
        users_with_sim_demo_labels = k_prototypes_recs(user_df)
        users_with_sim_demo_labels.to_pickle(
            f"pickles/users_with_sim_demo_labels.pkl")

        # get current user's recommendations
        user = users_with_sim_demo_labels.iloc[user_index]

        # get recommendations
        user_recs_sim_demo = get_sim_demo_recs(
            user, users_with_sim_demo_labels)
        user_recs_sim_demo_df = get_content(
            user, user_recs_sim_demo, content_df)
    else:
        user_recs_sim_demo_df = pd.DataFrame([])

    # get recs for every user based on users with similar preferences
    if sim_preferences:

        # cluster users based on their preferences using kmodes
        users_with_sim_pref_labels = k_modes_recs(
            user_df, content_df, sim_var="similar_preferences")
        users_with_sim_pref_labels.to_pickle(
            f"pickles/users_with_sim_pref_labels.pkl")

        # get current user's recommendations
        user = users_with_sim_pref_labels.iloc[user_index]

        # get recommendations
        user_recs_sim_pref = get_sim_prefs_recs(
            user, users_with_sim_pref_labels)
        user_recs_sim_pref_df = get_content(
            user, user_recs_sim_pref, content_df)
    else:
        user_recs_sim_pref_df = pd.DataFrame([])

    return user_recs_sim_faves_df, user_recs_sim_content_df, user_recs_fav_content_df, user_recs_sim_demo_df, user_recs_sim_pref_df

def jaccard_similarity(user_set, others_set):
    """
    Calculate Jaccard similarity between the current user's set of information and
    the other user's set of information.
    """

    # get intersection
    set_intersect = user_set.intersection(others_set)

    # get union
    set_union = user_set.union(others_set)

    # compute Jaccard similarity
    similarity = len(set_intersect) / len(set_union)

    return similarity


def get_diverse_recommendations(user_index, user_data, content_data, sim_genre=True, sim_demo=True, sim_favorites=True):
    """
    Returns three data frames containing diverse recommendations of three 
    different types, depending on whether any recommendations were found.
    Diversity was implemented by applying the principle of finding dissimilar 
    content within a similar type of content.

    If sim_genre == True, the recommendations are based on the content from the
    genre that is most similar to the user's favorite genre, but content that is 
    least similar to the user's favorite titles.

    If sim_demo == True, the recommendations are based on the least similar 
    favorites of the users with similar demographics to the user's demographics.

    If sim_favorites == True, the recommendations are based on the least similar 
    favorites of the users with similar favorites to the user's favorite titles.
    """

    # get the current user
    user = user_data.iloc[user_index]

    # get the current user's favorites
    user_faves = user["top10"]

    # find content with the genre that is similar to each of the user's favorites  
    # and recommend the content that is least similar to each of the user's favorites
    if sim_genre:
        list_of_dfs = []

        # loop over the user's favorite titles
        for fave in user_faves:

            # prevent crashing
            if fave not in content_data["title"].unique():
                continue

            # get all genres associated with this title
            genre_list = content_data[content_data["title"]
                                      == fave]["genre"].sum()

            # get this title's genre (by checking which one is most common)
            current_genre = Counter(genre_list).most_common(1)[0][0]

            # get other genres that are preferred by users that prefer the current genre, i.e., similar genres
            genre_in_list_mask = user_data["pref_genre"].apply(
                lambda genre_list: True if current_genre in genre_list else False)
            similar_genres_list = user_data[genre_in_list_mask]["pref_genre"].sum(
            )

            # prevent crashing in case a genre has not been picked by any user as preferred genre
            if similar_genres_list == 0:
                continue

            # remove all occurrences of the current genre in the list
            similar_genres_list = [
                genre for genre in similar_genres_list if genre != current_genre]

            # get the genre most commonly preferred alongside the current genre
            most_similar_genre = Counter(
                similar_genres_list).most_common(1)[0][0]

            # get all content with the similar genre that is not the user's favorite
            content_has_genre = content_data["genre"].apply(
                lambda genre_list: True if most_similar_genre in genre_list else False)
            similar_genre_content = content_data[content_has_genre & (
                content_data["title"] != fave)]

            # sum all similar content data into a column of sets
            content_info_col = similar_genre_content[[
                "genre", "tags", "tags2", "more", "rating", "title_type"]].sum(axis=1).apply(set)

            # sum all content data for the fave in a set
            fave_info = set(content_data[content_data["title"] == fave].iloc[0][[
                            "genre", "tags", "tags2", "more", "rating", "title_type"]].sum())

            # compute similarity between all sets of similar content and the fave set
            similar_genre_content["jac_similarity"] = content_info_col.apply(
                lambda content_info_list: jaccard_similarity(fave_info, content_info_list))

            # add content to list of data frames
            list_of_dfs.append(similar_genre_content)

        # no similar content was found
        if len(list_of_dfs) == 0:

            # pick 100 random titles
            user_recs_sim_content_df = content_data.sample(n=100)
        else:

            # concatenate the data frames
            user_recs_sim_content_df = pd.concat(
                list_of_dfs, ignore_index=True)

            # sort content by jaccard similarity and preferences; 
            # least similar and most preferred content at the top
            user_recs_sim_content_df = get_content(
                user, user_recs_sim_content_df, content_data, diverse=True)

    else:
        user_recs_sim_content_df = pd.DataFrame([])

    # find users in the user's demographics cluster and recommend these users' 
    # favorites that are least similar to each of the user's favorites
    if sim_demo:
        list_of_dfs = []

        # loop over the user's favorites
        for fave in user_faves:

            # prevent crashing
            if fave not in content_data["title"].unique():
                continue

            # find the assigned cluster for this user
            user_cluster = user["sim_demo_labels"]

            # find other users assigned to this cluster
            cluster_members = user_data[(user_data["sim_demo_labels"] == user_cluster) & (
                user_data["key"] != user["key"])]

            # no cluster members were found
            if cluster_members.empty:
                user_recs_sim_demo_df = pd.DataFrame([])

            else:

                # add these users' favorite titles to the recommendations if they are not in this user's favorites already
                cluster_member_faves = set(cluster_members["top10"].sum())
                new_titles = list(cluster_member_faves.difference(user_faves))

                # get the content data for these titles
                similar_demo_content = content_data[content_data["title"].isin(
                    new_titles)]

                # sum all content data into a column of sets
                content_info_col = similar_demo_content[[
                    "genre", "tags", "tags2", "more", "rating", "title_type"]].sum(axis=1).apply(set)

                # sum all content data for the fave in a set
                fave_info = set(content_data[content_data["title"] == fave].iloc[0][[
                                "genre", "tags", "tags2", "more", "rating", "title_type"]].sum())

                # compute similarity between all sets in similar content and the fave set
                similar_demo_content["jac_similarity"] = content_info_col.apply(
                    lambda content_info_list: jaccard_similarity(fave_info, content_info_list))

                # add content to list of data frames
                list_of_dfs.append(similar_demo_content)

        # no similar content was found
        if len(list_of_dfs) == 0:

            # pick 100 random titles
            user_recs_sim_demo_df = content_data.sample(n=100)
        else:

            # concatenate the data frames
            user_recs_sim_demo_df = pd.concat(list_of_dfs, ignore_index=True)

            # get content sorted by jaccard similarity and preferences; 
            # least similar and most preferred content at the top
            user_recs_sim_demo_df = get_content(
                user, user_recs_sim_demo_df, content_data, diverse=True)

    else:
        user_recs_sim_demo_df = pd.DataFrame([])

    # find users in the user's similar favorites cluster and recommend their 
    # favorites that are least similar to each of the user's favorites
    if sim_favorites:
        list_of_dfs = []

        # loop over the user's favorites
        for fave in user_faves:

            # prevent crashing
            if fave not in content_data["title"].unique():
                continue

            # find the assigned cluster for this user
            user_cluster = user["sim_fav_labels"]

            # find other users assigned to this cluster
            cluster_members = user_data[(user_data["sim_fav_labels"] == user_cluster) & (
                user_data["key"] != user["key"])]

            # no cluster members were found
            if cluster_members.empty:
                user_recs_sim_faves_df = pd.DataFrame([])
            else:

                # get the cluster members' favorites that are not in the user's favorites
                cluster_member_faves = set(cluster_members["top10"].sum())
                new_titles = list(cluster_member_faves.difference(user_faves))

                # get the content data for these titles
                similar_fav_content = content_data[content_data["title"].isin(
                    new_titles)]

                # sum all content data into a column of sets
                content_info_col = similar_fav_content[[
                    "genre", "tags", "tags2", "more", "rating", "title_type"]].sum(axis=1).apply(set)

                # sum all content data for the fave in a set
                fave_info = set(content_data[content_data["title"] == fave].iloc[0][[
                                "genre", "tags", "tags2", "more", "rating", "title_type"]].sum())

                # compute similarity between all sets in similar content and the fave set
                similar_fav_content["jac_similarity"] = content_info_col.apply(
                    lambda content_info_list: jaccard_similarity(fave_info, content_info_list))

                # add content to list of data frames
                list_of_dfs.append(similar_fav_content)

        # no similar content was found
        if len(list_of_dfs) == 0:

            # pick 100 random titles
            user_recs_sim_faves_df = content_data.sample(n=100)

        else:

            # concatenate the data frames
            user_recs_sim_faves_df = pd.concat(list_of_dfs, ignore_index=True)

            # get content sorted by jaccard similarity and preferences; 
            # least similar and most preferred content at the top
            user_recs_sim_faves_df = get_content(
                user, user_recs_sim_faves_df, content_data, diverse=True)

    else:
        user_recs_sim_faves_df = pd.DataFrame([])

    return user_recs_sim_content_df, user_recs_sim_demo_df, user_recs_sim_faves_df


def get_df_with_labels():
    """
    Load the .pkl files for the content data frame including the clustering labels, 
    and the user data frame including the clustering labels for similar favorites
    and similar demographics.

    Needed to generate the diverse recommendations without having to rerun the
    clustering algorithm.
    """

    # load content .pkl file
    content_with_labels = pd.read_pickle("pickles/content_with_labels.pkl")

    # load user .pkl files for similar favorites and similar demographics
    sim_faves_df = pd.read_pickle("pickles/users_with_sim_fav_labels.pkl")
    sim_demo_df = pd.read_pickle("pickles/users_with_sim_demo_labels.pkl")

    # create user data frame with both label columns
    users_with_labels = sim_demo_df.copy()
    users_with_labels["sim_fav_labels"] = sim_faves_df.iloc[:, -1:]

    return content_with_labels, users_with_labels


def main():

    # load data
    user_data = pd.read_pickle("data/ABC/final_users")
    content_data = pd.read_csv("data/ABC/programs_abc.csv")
    content_data = content_data.drop_duplicates()
    user_data, content_data = data_preprocessing(user_data, content_data)

    # get normal recommendations for a random user
    user_index = random.randint(0, len(user_data) - 1)
    recs = get_recommendations(user_index, user_data, content_data)
    print("normal recs:")
    print(recs)

    # get diverse recommendations for a random user
    content_with_labels, users_with_labels = get_df_with_labels()
    diverse_recs = get_diverse_recommendations(
        user_index, users_with_labels, content_with_labels)
    print("diverse recs:")
    print(diverse_recs)


if __name__ == "__main__":
    main()
