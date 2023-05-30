import streamlit as st
import pickle
import pandas as pd
import requests

def fetch_poster(movie_id):
    url = 'https://api.themoviedb.org/3/movie/{}?api_key=6e5aec0d002eb2338c934ecb5c14e915&language=en-US'.format(movie_id.movie_id)
    respone = requests.get(url)
    data = respone.json()
    return "https://image.tmdb.org/t/p/original"+data['poster_path']

def recommend(name):
    movie_index = movies[movies['title']==name].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    recommend_movies = []
    recommended_movies_posters = []
    for i in movie_list:
        movie_id = movies.iloc[i[0]]
        recommend_movies.append(movie_id.title)
        recommended_movies_posters.append(fetch_poster(movie_id))
    return recommend_movies,recommended_movies_posters


movies_list = pickle.load(open('movies.pkl','rb'))
movies = pd.DataFrame(movies_list)
movies_list = movies_list['title'].values
similarity = pickle.load(open('similarity.pkl','rb'))

st.title('Movie Recommender System')
selected_movies_name = st.selectbox('Movies List',(movies['title'].values))

if st.button('Recommend'):
    recommended_movies,posters = recommend(selected_movies_name)

    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        st.subheader(recommended_movies[0])
        st.image(posters[0])
    with col2:
        st.subheader(recommended_movies[1])
        st.image(posters[1])
    with col3:
        st.subheader(recommended_movies[2])
        st.image(posters[2])
    with col4:
        st.subheader(recommended_movies[3])
        st.image(posters[3])
    with col5:
        st.subheader(recommended_movies[4])
        st.image(posters[4])
    