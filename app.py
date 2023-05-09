import streamlit as st
import pandas as pd
#import streamlit_authenticator as stauth
import streamlit_authenticator2 as stauth2
import database as db
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import numpy as np
#import datetime
from datetime import date
#from streamlit.components.v1 import html
#from streamlit_javascript import st_javascript
import all_recommendations as rec
#from streamlit_extras.stoggle import stoggle
import warnings


st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
warnings.filterwarnings("ignore")


# -- FUNCTIONS --
def show_selected_image_from_carousel(selectedImageUrl):
    if selectedImageUrl is not None:
        selected_showName = df_shows[df_shows['image']==selectedImageUrl]['title'].values[0]
        selected_showDescription = df_shows[df_shows['image']==selectedImageUrl]['description'].values[0]
        selected_showDescription2 = df_shows[df_shows['image']==selectedImageUrl]['description2'].values[0]
        st.markdown(
            """
            <style>
            .container {
                display: flex;
            }
            .logo-text {
                font-size:28px;
                color: #FFFFFF;
                font-family: sans-serif;
            }
            .descrip-text {
                font-size:15px;
                color: #FFFFFF;
                font-family: sans-serif;
            }
            .descrip-text2 {
                font-size:12px;
                color: #FFFFFF;
                font-family: sans-serif;
                font-style: italic;
            }
            .logo-img {
                float:right;
                width: 40%;
                height: 40%;
                display: grid;
                border-radius: 10%
            }
            .show_text {
               padding: 20px; 
            }
           
            </style>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="container">
                <img class="logo-img" src={selectedImageUrl}>
                <div class="show_text">
                    <p class="logo-text">{selected_showName}</p>
                    <p class="descrip-text">{selected_showDescription}</p>
                    <p class="descrip-text2">{selected_showDescription2}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# -- USER AUTHENTICATION --
users = db.fetch_all_users()
usernames = [user["key"] for user in users]
names = [user["first_name"] for user in users]

# passwords are now hashed in the database python script to reduce runtime
hashed_passwords = [str(user["password"])[2:-2] for user in users]


# -- CREDENTIALS --
credentials = {"usernames":{}}
pre_aut = {"emails":[]}

for un, name, pw in zip(usernames, names, hashed_passwords):
    user_dict = {"name":name,"password":pw}
    credentials["usernames"].update({un:user_dict})
    

authenticator = stauth2.Authenticate(credentials,"program_recommender","abcdef",30,pre_aut)

# -- READ IN DATA --
df_shows = pd.read_csv("programs_abc.csv")
df_shows = df_shows.drop_duplicates()
df_users = pd.DataFrame(users)

# -- SESSION STATES --
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False
if 'name' not in st.session_state:
    st.session_state['name'] = ""
if 'username' not in st.session_state:
    st.session_state['username'] = ""
if 'date' not in st.session_state:
    st.session_state['date'] = date.today()
if 'randomPick1' not in st.session_state:
    st.session_state['randomPick1'] = df_shows.sample(1)
if 'randomPick2' not in st.session_state:
    st.session_state['randomPick2'] = df_shows.sample(1)
if 'prevUser' not in st.session_state:
    st.session_state['prevUser'] = None
if 'diversityFilter1' not in st.session_state:
    st.session_state['diversityFilter1'] = 100
if 'diversityFilter2' not in st.session_state:
    st.session_state['diversityFilter2'] = 100
if 'diversityFilter3' not in st.session_state:
    st.session_state['diversityFilter3'] = 100
    
  # -- INTERFACE --  
authentication_status = st.session_state['authentication_status']
name = st.session_state['name']
username = st.session_state['username']
prevUser = st.session_state['prevUser']
date_random_pick = st.session_state['date']
randomPick1 = st.session_state['randomPick1']
randomPick2 = st.session_state['randomPick2']

with st.sidebar:
    st.markdown(
        """
        <style>
            .img_logo_side{
                width:100%;
                heigth: 200px;
            }
            .logo_abc{
                width: 60%;
                height: 60%;}
            .img_logo_side:hover img {
                transform: scale(1);}
            .abc_logo_letters{
                font-size:80px;
                color: #FFFFFF;
                font-weight: 600;}
        </style>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
            <div class="img_logo_side">
                <center><img class="logo_abc" src="https://www.transparency.gov.au/sites/default/files/logos/lissajous-white-1574982067269.png">
                <div class="abc_logo_letters">ABC</div></center>
            </div>
                    
        """,unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Log in","Sign Up"],
        icons=["key","envelope"],
        menu_icon="cast",
        default_index=0)
   
if selected == "Log in":
    name, authentication_status, username = authenticator.login("Login","main")

    if authentication_status: 
        authenticator.logout("Logout", "sidebar")
        
        #-- GET RECOMMENDATIONS --
        @st.cache_data 
        def get_recommendations_interface(username, df_users, df_shows):
            user_data, content_data = rec.data_preprocessing(df_users, df_shows)
            user_index = user_data[user_data['key']==username].index[0]
            recs = rec.get_recommendations(user_index, user_data, content_data)
            
            content_with_labels, users_with_labels = rec.get_df_with_labels()
            diverse_recs = rec.get_diverse_recommendations(user_index, users_with_labels, content_with_labels)
            #content_with_labels, users_with_labels = rec.get_users_with_labels()
            #content_with_labels["rating"] = content_with_labels["rating"].apply(lambda rating : rating.split(sep=None))
            #div_recs = rec.get_diverse_recommendations(user_index, users_with_labels, content_with_labels)
            return recs, diverse_recs
        
        selected_hor = option_menu(
            menu_title=None,
            options=["Home","Movies","Tv shows","Diversity","Settings"],
            icons=["house","film","camera-reels",'globe','sliders'],
            default_index=0,
            orientation="horizontal")
        
        if selected_hor == "Home":
            st.title(f"Welcome {name}") 
            
            #different dates or different user, different random picks
            if (date_random_pick != date.today()) | (prevUser != username):
                st.session_state['prevUser'] = username
                randomPick1 = df_shows.sample(1)
                randomPick2 = df_shows.sample(1)
                st.session_state['randomPick1'] = randomPick1
                st.session_state['randomPick2'] = randomPick2

            show1Url = randomPick1['image'].values[0]
            show2Url = randomPick2['image'].values[0]
           
            show1Title = randomPick1['title'].values[0]
            show2Title = randomPick2['title'].values[0]
            
            curr_watch = df_shows.iloc[615]
            curr_watch_url = curr_watch['image']
            curr_watch_title = curr_watch['title']
            curr_watch_descrip = curr_watch['description']
            
            st.markdown(
                """
                <style>
                body {
                 padding: 0;
                 margin: 0;
                }
                .begin {
                    display: flex;
                }
                .begin_img {
                    width: 60%;
                    position: relative;
                }
                .begin_img img {
                    width: 100%;
                    height: auto;
                    border-radius: 10%;
                    position: absolute;
                    left: 0;
                    bottom: 40px;
                    border: 2px solid red;
                }
                .begin_img2{
                    width: 40%;
                    text-align: center;
                }
                .begin_img2 img {
                    float: right;
                    width: 80%;
                    height: auto;
                    margin: 15px;
                    border-radius: 10%;
                }
                .today_pick {
                    font-size:28px;
                    color: #FFFFFF;
                    font-family: sans-serif;
                }
                .curr_watch {
                    font-size:32px;
                    color: #FFFFFF;
                    font-family: sans-serif;
                }
                .title_cur {
                    font-size: 50px;}
                img:hover {
                    transform: scale(1.1);
                }
                .overlay {
                  position: absolute;
                  top: 106px;
                  bottom: 39px;
                  left: 0;
                  right: 0;
                  height: auto%;
                  width: 100%;
                  opacity: 0;
                  transition: .5s ease;
                  background-color: rgba(0, 0, 0, 0.5);
                  border-radius: 10%;
                  
                }
                .overlay2 {
                  position: absolute;
                  top: 12%;
                  bottom:40%;
                  right: 15px;
                  width: 32%;
                  height: 42%;
                  opacity: 0;
                  transition: .5s ease;
                  background-color: rgba(0, 0, 0, 0.5);
                  border-radius: 8%;
                  
                }
                .overlay3 {
                  position: absolute;
                  top: 55%;
                  bottom: 0%;
                  right: 15px;
                  width: 32%;
                  heigth: auto;
                  opacity: 0;
                  transition: .5s ease;
                  background-color: rgba(0, 0, 0, 0.5);
                  border-radius: 8%;
                  
                }
                .begin_img:hover .overlay, .check2:hover .overlay3, .check1:hover .overlay2{
                    opacity: 1;
                }
    
                .descrip_begin_img1 {
                    position: absolute;
                    top: 40%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: #fff;
                    font-size:1.8vw;
                    font-family: sans-serif;
                    -webkit-transform: translate(-50%, -50%);
                  -ms-transform: translate(-50%, -50%);
                  transform: translate(-50%, -50%);
                  text-align: center;
                }
                .descrip_begin_img11 {
                    position: absolute;
                    top: 60%;
                    left: 50%;
                    transform: translate(-50%, -50%);
                    color: #fff;
                    font-size: 0.9vw;
                    font-family: sans-serif;
                    -webkit-transform: translate(-50%, -50%);
                  -ms-transform: translate(-50%, -50%);
                  transform: translate(-50%, -50%);
                  text-align: center;
                }
             
                .descrip_begin_img2 {
                    position: absolute;
                    top: 90px;
                    right: 150px;
                    color: white;
                    font-size:1.5vw;
                    opacity: 0;
                    text-align: center;
                    }
                
                .descrip_begin_img3 {
                    position: absolute;
                    bottom: 100px;
                    right: 110px;
                    color: white;
                    font-size:1.5vw;
                    opacity:0;
                  text-align: center;
                    }
                .check2:hover .descrip_begin_img3, .check1:hover .descrip_begin_img2 {
                    opacity:1;}
                .begin_img31 {
                    border: 2px solid green;}
                .begin_img32 {
                    border: 2px solid orange;}
             
               
               
                </style>
                """,
                unsafe_allow_html=True
            )
    
            st.markdown(
                f"""
                <div class="begin">
                    <div class="begin_img">
                        <p class="curr_watch">Currently Watching</p>
                        <img class="begin_img1" style="vertical-align: bottom" src={curr_watch_url}>
                        <div class="overlay">
                            <div class="descrip_begin_img1">{curr_watch_title}</div>
                            <div class="descrip_begin_img11">{curr_watch_descrip}</div>
                        </div>
                    </div>
                    <div class="begin_img2">
                        <p class="today_pick">Today's Random Pick:</p>
                        <div class="check1">
                            <img class="begin_img31" src={show1Url}>
                            <div class="overlay2">
                                <div class="descrip_begin_img2">{show1Title}</div>
                            </div>
                        </div>
                        <div class="check2">
                            <img class="begin_img32" src={show2Url}>
                            <div class="overlay3">
                                <div class="descrip_begin_img3">{show2Title}</div>
                            </div>
                        </div>
                    </div>
       
                </div>
                """,
                unsafe_allow_html=True
            )
            
       
            #first recommender 
            st.subheader("")
            st.subheader("Your Recommendations Based On Your Favorites")
            
            imageCarouselComponent = components.declare_component("image-carousel-component1", path="frontend/public")
            
            df_recom, df_recom_div = get_recommendations_interface(username, df_users, df_shows)
            first_recom = df_recom[1]['image'][:15].to_list()
            imageUrls = first_recom
            selectedImageUrl = imageCarouselComponent(imageUrls=imageUrls, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl)
        
            #recommender zero
            st.subheader("")
            st.subheader("Recommendations Based On Favorite Shows By Similar Users")
                
            imageCarouselComponent_zero = components.declare_component("image-carousel-component0", path="frontend/public")
                
            zero_recom = df_recom[0]['image'][:15].to_list()
            selectedImageUrl_zero = imageCarouselComponent_zero(imageUrls=zero_recom, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_zero)
        
            #second recommender
            st.subheader("")
            st.subheader("Recommendations Based On Your Favorite Genres")
                
            imageCarouselComponent_second = components.declare_component("image-carousel-component2", path="frontend/public")
            second_recom = df_recom[2]['image'][:15].to_list()
            selectedImageUrl_second = imageCarouselComponent_second(imageUrls=second_recom, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_second)
            
            #third recommender
            st.subheader("")
            st.subheader("Recommendation Based On Users With Similar Demographics")
                
            imageCarouselComponent_third = components.declare_component("image-carousel-component3", path="frontend/public")
            third_recom = df_recom[3]['image'][:15].to_list()
            selectedImageUrl_third = imageCarouselComponent_third(imageUrls=third_recom, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_third)
            
            #fouth recommender
            st.subheader("")
            st.subheader("Recommendation Based On Users With Similar Preferences")
                
            imageCarouselComponent_fouth = components.declare_component("image-carousel-component4", path="frontend/public")
            fouth_recom = df_recom[4]['image'][:15].to_list()
            selectedImageUrl_fouth = imageCarouselComponent_fouth(imageUrls=fouth_recom, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_fouth)
            
            st.subheader("")
            st.subheader("")
            authenticator.logout(f"Logout {name}", "main")
                
                
        if selected_hor == "Settings":
            st.header(f"Change settings and preferences for user: {name}") 
            st.subheader("User Preferences")
            with st.form(key="preferences", clear_on_submit=True):
                user_check = db.get_user(username)
                diff_shows = df_shows['title'].unique()
                top10_shows_curr = str(user_check['top10']).replace('"',"")
                top10_shows = st.multiselect("Select your top 10 programs",set(diff_shows),max_selections=10)
                st.caption(f"Currently selected: {top10_shows_curr[1:-1]}")
                
                genres = ['DOCUMENTARY','ARTS & CULTURE','REGIONAL AUSTRALIA','PANEL & DISCUSSION','LIFESTYLE','EDUCATION','FAMILY','COMEDY','INDIGENOUS','DRAMA','SCIENCE','KIDS','SPORT']
                previous_genres = str(user_check['pref_genre']).replace("'","")
                genre = st.multiselect("Genre",set(genres), max_selections=(len(genres)-1))
                st.caption(f"Currently selected: {previous_genres[1:-1]}")
                
                #title type
                title_types = df_shows['title_type'].unique()
                previous_titles = str(user_check['pref_title_type']).replace("'","")
                title_type = st.multiselect("Title type", title_types, max_selections=(len(title_types)-1))
                st.caption(f"Currently selected: {previous_titles[1:-1]}")
                
                ratings = df_shows['rating'].unique()
                previous_ratings = str(user_check['pref_rating']).replace("'","")
                ratings = ["No preference" if x is np.nan else x for x in ratings]
                rating = st.multiselect("Rating", set(ratings), max_selections=3)
                st.caption(f"Currently selected: {previous_ratings[1:-1]}")
                
                dates_col_split = df_shows['publication_date'].apply(lambda x: str(x).split())
                dates_shows = dates_col_split.apply(lambda x: x[0])
                dates_shows_datetime = pd.to_datetime(dates_shows)
                min_date = min(dates_shows_datetime).year
                max_date = max(dates_shows_datetime).year + 1
                user_pref_date = tuple(user_check['pref_years'])
                pref_date = st.slider("Year of release",min_date,max_date,user_pref_date)
                sub_pref = st.form_submit_button(label="Update preferences")
                
                #button clicked update changed
                if sub_pref:
                    if len(top10_shows) == 0 or len(top10_shows) == 10:
                        if top10_shows != []:
                            db.update_user(username, updates={"top10": top10_shows})
                        if genre != []:
                            db.update_user(username, updates={"pref_genre": genre})
                        if rating != []:
                            db.update_user(username, updates={"pref_rating": rating})
                        if pref_date != user_pref_date:
                            db.update_user(username, updates={"pref_years": pref_date})
                        if title_type != []:
                            db.update_user(username, updates={"pref_title_type": title_type})
                        st.success("User preferences succesfully updated")
                    else: 
                        st.error("You have selected less than 10 shows")      
            
            st.subheader("User Demographics")
            with st.form("user_demographics",clear_on_submit=True):
                f_name = st.text_input("First Name")
                l_name = st.text_input("Last Name")
                age = st.date_input("Date of birth")
                today = date.today()
                gender = st.selectbox('Gender',('','Male','Female','Other'))
                location = st.text_input("Location")
                sub_demo = st.form_submit_button(label="Update demographics")
                if sub_demo:
                    st.success("User demographics succesfully updated")
                    if f_name != "":
                        db.update_user(username, updates={"first_name": f_name})
                        name = f_name
                        st.session_state['name'] = f_name
                    if l_name != "":
                        db.update_user(username, updates={"last_name": l_name})
                    if location != "":
                        db.update_user(username, updates={"location": location})
                    if gender != "":
                        db.update_user(username, updates={"gender": gender})
                    if age != today:
                        db.update_user(username, updates={"dob": str(age)})
        
        if selected_hor == "Movies":
            #trailer does only work one day since IMDB changes source of trailer everyday
            js_code = '''
                 <style>
                     .vid {
                         position: relative;
                            }
                     video {
                         width: 90%;
                         height: 90%;
                         border-radius:30px;}
                     .vid_title {
                         position: absolute;
                         top: 80%;
                         left: 50%;
                         transform: translate(-50%,-50%);
                         color: #fff;
                         opacity: 0;
                         font-size:70px;
                         font-family: sans-serif;
                    }   
                     .vid_contain:hover .vid_title{
                         opacity: 0;}
                 </style>
                 <meta name="viewport" content="width=device-width, initial-scale=1.0">
                 <div class="vid_contain">
                 <video  controls class="vid" style="width:100%;">
                    <source class="trailer" src="https://imdb-video.media-imdb.com/vi3481443353/MV5BODMyMjFlM2MtMTIzOS00OWM5LTlhOWMtZWVhZWI5N2FjNzQyXkExMV5BbXA0XkFpbWRiLWV0cy10cmFuc2NvZGU@.mp4?Expires=1680683097&Signature=Hd~qgHpT5THgBfAFIrUBIhkfAPFYDkEsBposEu9BNM~~LOhKMYNjGsE0ZOvOe7q4f6KAlI1Hl-FMCKfLSNxXPvd5~xSKffnplwVjmfUblBCGBxfvrAp~TBCobcPpV6NzWyj59MmNYOlgadzVO1PzNEA6CzVHtUrZbrmhwmTkQWsPpkKvDP4RsXYmO1kyYBEQnGKr0ZcDme4o5OZ~SWd7~oM9oCD9PHLeEewUhgqb5pb6eqm1K8x-YdO0fRinW85yWVeIkLN4pGjgS0iduFDHsr3ex~5bIQW3KOwMX-~U31E1rM1VlhD-Md~jYZuJKLsFiWNpyihyzVSWMtz93fTkaA__&Key-Pair-Id=APKAIFLZBVQZ24NQH3KA#t=30" type="video/mp4">
                 </video>
                 <div class="vid_title">Griff the Invisible</div>
                     
                 </div>
                 <script>
                     let clip = document.querySelector(".vid")
                     
                     clip.addEventListener("mouseover", function (e) {
                         clip.play();
                     })
               
                     clip.addEventListener("mouseout", function (e) {
                         clip.pause();
                     })
             
                  </script>
                      '''
            components.html(js_code, width=1500, height=700)

            st.subheader("")
            st.subheader("Movies")
                
            imageCarouselComponent3 = components.declare_component("image-carousel-component", path="frontend/public")
                
            imageUrls_movies = df_shows[df_shows['title_type']=='Movie']['image'][:15].tolist()
            selectedImageUrl_movie = imageCarouselComponent3(imageUrls=imageUrls_movies, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_movie)
        
        if selected_hor == "Tv shows":
            js_code = '''
                 <style>
                     .vid2 {
                         position: relative;
                         width: 100%;
                         height: 100%;}
                     .vid_title2 {
                         position: absolute;
                         top: 80%;
                         left: 50%;
                         transform: translate(-50%,-50%);
                         color: #fff;
                         font-size:70px;
                         font-family: sans-serif;
                         opacity: 0;
                    }   
                     .vid_contain2:hover .vid_title2 {
                         opacity: 0;}
                 </style>
                 
                 <div class="vid_contain2">
                 <video  controls class="vid2">
                     <source src="https://imdb-video.media-imdb.com/vi1629602329/1434659607842-pgv4ql-1657924246665.mp4?Expires=1680683368&Signature=vRViHxShunbFFVpnSeGYu1~BbDBu9qZC2uR1hdSqxDrRWvRMVNvMeP~RUUYblbKJw~RGsvWk2jzyET1ZGJC66-yGN3W086-NY61vLjWEEpL~5AeCP8vQvbdfS10K6YasWnMfMnQ8T4ZVpi4AVhVQcYjSgze372G1gIgL4CYuLOyMHx36Yki9~psiD4F3KWuW3fLFU8eV7Q0bxTBViEb7Ye5HaN2aqBykNR5ngY7GLHDOFWXJr8ZiRsy0Spr0mfvz9huiwF8eSY4iwZh63GXlCjBn9igdo6aNKMuoEVIOBVjlQZye3DfNxZt4MMjZTD2KBirjSunPfD5JdiKTkoLz4w__&Key-Pair-Id=APKAIFLZBVQZ24NQH3KA" type="video/ogg">
                 </video>
                 <div class="vid_title2">London Spy</div>
              
                 </div>
                 <script>
                     let clip = document.querySelector(".vid2")
                     
                     clip.addEventListener("mouseover", function (e) {
                         clip.play();
                     })
               
                     clip.addEventListener("mouseout", function (e) {
                         clip.pause();
                     })
             
                  </script>
                      '''
            components.html(js_code, width=1500, height=700)
            
            st.subheader("")
            st.subheader("Tv shows")
                
            imageCarouselComponent4 = components.declare_component("image-carousel-component", path="frontend/public")
                
            imageUrls_series = df_shows[df_shows['title_type']=='tv']['image'][2000:2015].tolist()
            selectedImageUrl_movie = imageCarouselComponent4(imageUrls=imageUrls_series, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_movie)
        
        if selected_hor == "Diversity":
            st.markdown(
                """
                <style>
                .why_diver {
                    text-align: center;
                    }
                .div_text {
                    font-size: 40px;}
                </style>
                """, unsafe_allow_html=True
                )
            st.markdown(
                f"""
                <h1 style="text-align: center; color: white;"><i>Why Diversity?</i></h1>
                <div class="why_diver">
                    <p class"div_text">At ABC Australia, we strive to recommend users content that is as diverse as our modern day society. This way, we hope to inspire you to discover new interests and experiences.</p>
                    <p class="quote"><i>"Without diversity, algorithms recommend a self-fulfulling prophecy of sameness that reinforces and amplifies biases we already have."</i></p>
                    <p class-"quote_author">- Cathy O'Neil</p>
                </div>
                
                """, unsafe_allow_html=True)
            
            st.subheader("")
        
            col1, col2 = st.columns([3,1])
            col1.subheader("Diverse Recommendations Based On Genres Similar To Your Favorite Genres")
            with col2:
                with st.expander("Diversity filter", expanded=False):
                    diversity_slider = st.slider('How diverse would you like your recommendations?', 0, 100, 100, key="divFilter1", help="Setting this filter to 100% will make your recommendations as diverse as possible. Setting it to 0% will simply provide you with your regular recommendations")
                    st.session_state['diversityFilter1'] = diversity_slider

            #first diverse recommender
            df_recom, df_recom_div = get_recommendations_interface(username, df_users, df_shows)
            imageCarouselComponent_div1 = components.declare_component("image-carousel-component_div1", path="frontend/public")
            rows_div1 = len(df_recom_div[0])
            if diversity_slider == 0:
                end_div1 = rows_div1
                begin_div1 = end_div1 - 15
            else:
                end_div1 = rows_div1 - int((diversity_slider/100)*rows_div1) + 15
                begin_div1 = end_div1 - 15
            
            first_recom_div = df_recom_div[0]['image'][begin_div1:end_div1].to_list()
            selectedImageUrl_div1 = imageCarouselComponent_div1(imageUrls=first_recom_div, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_div1)
            st.caption(f"Current diversity filter: {diversity_slider}%")
            
            #second diverse recommender
            st.subheader("")
            col3, col4 = st.columns([3,1])
            col3.subheader("Diverse Recommendations Based On Your Demographics")
            with col4:
                with st.expander("Diversity filter", expanded=False):
                    diversity_slider2 = st.slider('How diverse would you like your recommendations?', 0, 100, 60, key="divFilter2")
                    st.session_state['diversityFilter2'] = diversity_slider2
     
            imageCarouselComponent_div2 = components.declare_component("image-carousel-component_div2", path="frontend/public")
            rows_div2 = len(df_recom_div[1])
            if diversity_slider2 == 0:
                end_div2 = rows_div2
                begin_div2 = end_div2 - 15
            else:
                end_div2 = rows_div2 - int((diversity_slider2/100)*rows_div2) + 15
                begin_div2 = end_div2 - 15
            
            second_recom_div = df_recom_div[1]['image'][begin_div2:end_div2].to_list()
            selectedImageUrl_div2 = imageCarouselComponent_div2(imageUrls=second_recom_div, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_div2)
            st.caption(f"Current diversity filter: {diversity_slider2}%")
            
            #third diverse recommender
            st.subheader("")
            col5, col6 = st.columns([3,1])
            col5.subheader("Diverse Recommendations Based On Content Similar To Your Favorites")
            with col6:
                with st.expander("Diversity filter", expanded=False):
                    diversity_slider3 = st.slider('How diverse would you like your recommendations?', 0, 100, 60, key="divFilter3")
                    st.session_state['diversityFilter3'] = diversity_slider3
                
            imageCarouselComponent_div3 = components.declare_component("image-carousel-component_div3", path="frontend/public")
            rows_div3 = len(df_recom_div[2])
            if diversity_slider3 == 0:
                end_div3 = rows_div3
                begin_div3 = end_div3 - 15
            else:
                end_div3 = rows_div3 - int((diversity_slider3/100)*rows_div3) + 15
                begin_div3 = end_div3 - 15
            third_recom_div = df_recom_div[2]['image'][begin_div3:end_div3].to_list()
            selectedImageUrl_div3 = imageCarouselComponent_div3(imageUrls=third_recom_div, height=200)
            
            show_selected_image_from_carousel(selectedImageUrl_div3)
            
            st.caption(f"Current diversity filter: {diversity_slider3}%")
            
            st.subheader("")
            st.subheader("")

            
    if authentication_status == False:
        st.error("Username/password is incorrect")
    if authentication_status == None:
        st.warning("Please enter your username and password or sign up")

    

if selected == "Sign Up":
    if authentication_status:
        st.title(f"You are already signed in as {name}")
        authenticator.logout(f"Logout {name}", "sidebar")
    else:
        try:
            if authenticator.register_user('Register user', "main",preauthorization=False):
                st.success('User registered successfully, now you can log in')
        except Exception as e:
            st.error(e)
    

