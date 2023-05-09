#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 15:34:01 2023

@author: nikkikramer
"""
import pandas as pd
from deta import Deta
import streamlit_authenticator as stauth

DETA_KEY = "a0ddw3hdwsr_x59cuXqC7mjnetCZ21hdoEMgWQZLVA95"

deta = Deta(DETA_KEY)

db = deta.Base("recommender")

def insert_user(username, name, lname, email, gender, age, location,password, top10_shows, genre, rating, pref_date, title_type):
    return db.put({"key": username, "first_name": name, "last_name": lname, "email": email, "gender": gender,
                   "dob": age, "location": location, "password": password, "top10": top10_shows, "pref_genre": genre,
                   "pref_rating": rating, "pref_years": pref_date, "pref_title_type": title_type})
#top5_shows, genre, rating, pref_date
def fetch_all_users():
    res = db.fetch()
    return res.items

def get_user(username):
    return db.get(username)

def update_user(username, updates):
    return db.update(updates, username)

def delete_user(username):
    return db.delete(username)

# -- LOAD SYNTHESIZED DATA INTO ONLINE DATABASE --
# user_data = pd.read_pickle("final_users")
# for index, row in user_data.iterrows():
#     password = row['password']
#     row['password'] = stauth.Hasher([str(password)]).generate()
#     user = row.to_dict()
#     db.put(user)

    


