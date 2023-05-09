# import libraries
import ast
import numpy as np
import pandas as pd
import random
import datetime as dt
import math
from collections.abc import Iterable

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)
pd.set_option('display.float_format', '{:20,.2f}'.format)
pd.set_option('display.max_colwidth', None)

# to convert value to lowercase except for the first character
def lowercase_except_first(s):
    return s[:1] + s[1:].lower()

# to flatten a list
def flatten(lis):
     for item in lis:
         if isinstance(item, Iterable) and not isinstance(item, str):
             for x in flatten(item):
                 yield x
         else:        
             yield item

# import data
content_data = pd.read_csv("data/ABC/programs_abc.csv")

df_password= pd.read_csv("data/userdata/passwords.csv", sep=";")
df_boys= pd.read_csv("data/userdata/boynames.csv", sep=";")
df_girls= pd.read_csv("data/userdata/girlnames.csv", sep=";")
df_location_out= pd.read_csv("data/userdata/nl_smallcity.csv", sep=";")
df_location= pd.read_csv("data/userdata/nl_bigcity.csv", sep=";")

# exclude title named:'No title found' from the dataset
content_data = content_data[content_data.title != 'No title found']


start_date_young = dt.date(1953, 1, 1)
end_date_young  = dt.date(2005, 1, 1)

start_date_old = dt.date(1943, 1, 1)
end_date_old = dt.date(1953, 1, 1)

# convert the dataframes to lists
password = df_password.Password.tolist()
boys = df_boys.name.tolist()
girls = df_girls.name.tolist()
location_out = df_location_out.city.tolist()
location = df_location.city.tolist()

# get the different type of titles and put this in a list
title_type = content_data.title_type.unique().tolist()

# convert the column data to get the years of publication of the titles
content_data.publication_date = pd.to_datetime(content_data.publication_date)
pref_date = content_data.publication_date.dt.year.unique().tolist()
pref_date = [x for x in pref_date if not math.isnan(x)]
pref_date = [int(x) for x in pref_date]

# Get the differenct genres and put this in a list
genre = content_data.genre.values.tolist()
genre = list(flatten(genre))
genre = [ast.literal_eval(s) for s in genre]
genre = list(set([item for sublist in genre for item in sublist]))

# clean the string; remove brackets and quotation marks
def convert(string):
    string = string.strip("][").replace("' ", "").replace(" '", "").replace("'","") 
    lis = list(string.split(","))
    return lis

def add_data(df_genres, gender, location, start_date, end_date, aantal, sex, genre, pref_date, title_type, password):
  
    rating = ['PG', 'M', 'MA', 'No preference', 'G']
    key=['tje','00','01','tjuuh','x', 'XX', '123', '759', '']
    email=['@outlook.com','@outlook.nl','@live.nl','@hotmail.com','@hotmail.nl', '@gmail.com', '@icloud.com']

    df = pd.DataFrame(columns = ['age', 'email', 'first_name','gender', 'genre','key','location','password','prefDate','rating', 'title_type','genre2','top5'])
    
    # generate random values for in the dataframe
    for i in range(aantal):
        time_between_dates = end_date - start_date
        days_between_dates = time_between_dates.days
        random_number_of_days = random.randrange(days_between_dates)
        random_date = start_date + dt.timedelta(days=random_number_of_days)
        random_date = random_date.strftime('%Y-%m-%d')
        
        # get random a number between 1 and 3 for determining the amount of favourite genres/ratings and title_type
        num_genres = random.randint(1, 3)
        num_ratings = random.randint(1, 3)
        num_type = random.randint(1, 3)
        
        df.at[i, 'first_name'] = random.choice(gender)
        df.at[i, 'location'] = random.choice(location)
        df.at[i, 'genre'] = random.sample(genre, num_genres) 
        df.at[i, 'rating'] = random.sample(rating, num_ratings)
        df.at[i, 'prefDate'] = random.sample(pref_date, 2)
        df.at[i, 'title_type'] = random.sample(title_type, num_type)
        df.at[i, 'password'] = random.choice(password)
        df.at[i, 'key'] =df.at[i, 'first_name'] + random.choice(key)
        df.at[i, 'email'] =  df.at[i, 'key'] + random.choice(email)
        df.at[i, 'age'] = random_date

        # depending on the gender, add in the column gender the sex.
        if sex == 'boy':
            df.at[i, 'gender'] = 'Male'
        elif sex =='girl': 
            df.at[i, 'gender'] = 'Female'

    # if a row value in the column ratings contains no preference then that wil be the only value 
    for index, row in enumerate(df.rating):
        if 'No preference' in row:
            df.at[index, 'rating'] = ['No preference']
    
    # get favorite titles based on title type en genre. Users with a favoriete genre/title only get a favoriete title which is in their favoriete genre or titletype
    for index1, user_row in df.iterrows():
        genre_user = user_row.genre
        titletype_user = user_row.title_type
        list_titles=[]
        genre_user = [element.lower() for element in genre_user]     
        for index2, rij in df_genres.iterrows():    
            genre_title = rij.genre
            titletype = rij.title_type
            lijst = convert(genre_title)
            for genre in genre_user:
                if genre in lijst or titletype in  titletype_user:
                    title = df_genres.at[index2, 'title']
                    if title not in list_titles:
                        list_titles.append(title)
        
        list_to_edit = random.sample(list_titles, 10)
        favo_genre = '[' + ', '.join(['"' + s.replace('"', '\\"') + '"' for s in list_to_edit]) + ']'
        df.at[index1, 'top5'] = favo_genre
    df.genre2 = df.genre.copy()
    df = df.rename(columns={'genre2': 'genre'})
    return df

df1 = add_data(content_data, girls, location, start_date_young, end_date_young, 1, 'girl', genre, pref_date, title_type, password)
df2 = add_data(content_data, boys, location, start_date_young, end_date_young, 1, 'boy', genre, pref_date, title_type, password)
df3 = add_data(content_data, girls, location_out, start_date_old, end_date_old, 1, 'girl', genre, pref_date, title_type, password)
df4 = add_data(content_data, boys, location_out, start_date_old, end_date_old, 1, 'boy', genre, pref_date, title_type, password)


# df1 = add_data(content_data, girls, location, start_date_young, end_date_young, 400, 'girl', genre, pref_date, title_type, password)
# df2 = add_data(content_data, boys, location, start_date_young, end_date_young, 400, 'boy', genre, pref_date, title_type, password)
# df3 = add_data(content_data, girls, location_out, start_date_old, end_date_old, 100, 'girl', genre, pref_date, title_type, password)
# df4 = add_data(content_data, boys, location_out, start_date_old, end_date_old, 100, 'boy', genre, pref_date, title_type, password)

frames = [df1, df2, df3, df4]

merged_df = pd.concat(frames, ignore_index=True)
merged_df['prefDate'] = merged_df['prefDate'].apply(lambda x: sorted(x))

print(merged_df.head())

# save dataframe as csv file
merged_df.to_csv("user_data.csv", sep=';')

