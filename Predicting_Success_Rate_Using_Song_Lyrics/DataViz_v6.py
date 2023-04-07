# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


###############################################################################
# SPOTIFY POPULARITY DASHBOARD
###############################################################################

#==============================================================================
# LIBRARIES
#==============================================================================

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from streamlit_option_menu import option_menu
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from readability import Readability
from gensim.models import LdaModel
import plotly.graph_objects as go
from transformers import pipeline
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from textblob import TextBlob
from sklearn.svm import SVC
import plotly.express as px
from gensim import corpora
import streamlit as st
import pandas as pd
import contractions
import numpy as np
import gensim
import spacy
import nltk
import time
import re
nltk.download('punkt')
nltk.download('vader_lexicon')
#==============================================================================
# IMPORT DATA
#==============================================================================

# read the dataset 
songs = pd.read_csv('Predicting_Success_Rate_Using_Song_Lyrics/ALL_SONGS_with_lyrics.csv', encoding='iso-8859-1')
top_songs = songs.sort_values(by='popularity', ascending=False).head(20)
bow_popularity = pd.read_excel('Predicting_Success_Rate_Using_Song_Lyrics/BOW_per_popularity.xlsx')
bow_song = pd.read_excel('Predicting_Success_Rate_Using_Song_Lyrics/BOW_per_song.xlsx')

#Reading the data and subsetting those values without lyrics
data=pd.read_csv('Predicting_Success_Rate_Using_Song_Lyrics/lyrics_preprocessed.csv')\
.drop('Unnamed: 0',axis=1)\
.dropna(subset='cleaned_lyrics')\
.drop_duplicates(subset=['song_name'])

#We will keep only the id and the lyrics
df=data[['id','cleaned_lyrics']]

# Reading the dataset for sentiment analysis
sentiment_songs = pd.read_excel('Predicting_Success_Rate_Using_Song_Lyrics/df_pre_modelling.xlsx')

# Reading the pre-modelling dataset
df_pre_modelling = pd.read_excel('Predicting_Success_Rate_Using_Song_Lyrics/df_pre_modelling - 01.xlsx')

#==============================================================================
# HEADER
#==============================================================================

# Set primary color to green
st.set_page_config(page_title='Spotify Dashboard', page_icon='🎵', initial_sidebar_state = 'auto', layout="wide")
st.markdown("""<style>.stProgress .st-bo {background-color: green;}</style>""", unsafe_allow_html=True)

# Add dashboard title and description
st.title("SPOTIFY ROCKSTARS 🎸")
st.write("Built by Maria Karakoulian, Arunkkumar Karthikayan, Fabrizio Lucero, Fernando Diaz")

#==============================================================================
# SIDEBAR
#==============================================================================

# Add an image
st.sidebar.image('https://logodownload.org/wp-content/uploads/2020/03/listen-on-spotify.png')
songs = songs.drop_duplicates(subset=['song_name'])
songs['song_name'] = songs['song_name'].str.title()

#==============================================================================
# BODY
#==============================================================================

# Create tabs and add tab titles
with st.sidebar:
    
    active_tab = option_menu(menu_title=None,
                    options=["Key Attributes", "Song Information", "Success Factors", "Popularity Prediction"],
                    icons=['key-fill', 'info-circle-fill', 'trophy-fill', 'list-stars'],
                    orientation='vertical',
                    styles={"nav-link-selected": {"background-color": "lightgreen"},})

# Deploy botton

if active_tab == 'Success Factors' or active_tab == 'Popularity Prediction':
    search_song = st.sidebar.selectbox('Select a Song', songs['song_name'], disabled=True,key='popo')
else:
    search_song = st.sidebar.selectbox('Select a Song', songs['song_name'])

# ------------------------------------TAB 1------------------------------------  

def Song_Attributes():
    
    # Select the row corresponding to the selected song
    song_row = songs.loc[songs['song_name'] == search_song]
    
    if not song_row.empty:
        
        col1, col2 = st.columns([3,3])
        
        with col1:
             
             # Insert a radar plot for audio features
             #st.header('**Audio Features**')
             st.markdown('<div style="text-align:center;font-weight: bold;font-size: 28px;">Audio Features</div>', unsafe_allow_html=True)  
             
             # Define the variables and their respective values
             variables = ['danceability', 'speechiness', 'acousticness', 'instrumentalness', 'liveness']
             values = song_row[variables].values[0].tolist()
             
             # Define the layout for the radar chart
             layout = go.Layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1])),
                legend=dict(
                        x=0,
                        y=1.35),
                        showlegend=True,)
                
             # Define the trace for the radar chart with updated variable labels
             variable_labels = ['Danceability', 'Speechiness', 'Acousticness', 'Instrumentalness', 'Liveness']
             #theta_labels = [f'<span style="font-weight:bold;color:black">{label}</span>' for label in variable_labels]
            
             # Define the trace for the radar chart
             trace = go.Scatterpolar(
                r=values,
                theta=variable_labels,
                fill='toself',
                name=search_song,
                line=dict(color='black'))
        
             # Combine the layout and trace into a figure object
             fig = go.Figure(data=[trace], layout=layout)
             
             # Create a new trace for the radar chart using the values of the audio features for the top song
             top_song_values = top_songs[variables].mean().tolist()
             top_song_trace = go.Scatterpolar(
                                  r=top_song_values,
                                  theta=variable_labels,
                                  fill='toself',
                                  name='Top 20 Songs by Popularity',
                                  line=dict(color='green'))

             # Add the new trace to the figure object
             fig.add_trace(top_song_trace)
       
             # Display the figure
             st.plotly_chart(fig, use_container_width=True, width=50, height=50)
    
        with col2:
            
            # Insert a quadrant plot for song emotions
            st.markdown('<div style="text-align:center;font-weight: bold;font-size: 28px;">Mood Meter</div>', unsafe_allow_html=True)
            
            # Define the variables
            x = songs['energy']
            y = songs['valence']
            names = songs['song_name']
            
            # Create a scatter plot with energy and valence on the x and y axes
            fig = px.scatter(top_songs, x='energy', y='valence', hover_name="song_name",color_discrete_sequence=['lightgray'])

            # Highlight the selected song
            if search_song in names.tolist():
                idx = names.tolist().index(search_song)
                fig.add_trace(go.Scatter(
                    x=[x[idx]],
                    y=[y[idx]],
                    mode='markers',
                    marker=dict(
                        color='black',
                        size=15
                    ),
                    name=search_song
                ))

            # Add a trace for Scatter Plot
            fig.add_trace(go.Scatter(
                x=top_songs[:-1],
                y=top_songs[:-1],
                mode='markers',
                marker=dict(
                    color='lightgray',
                    size=10
                ),
                name='Top 20 Songs by Popularity'
            ))
            
            # Define the layout for the quadrant plot
            fig.update_layout(
            xaxis_title="Energy",
            yaxis_title="Valence",
            xaxis=dict(range=[0, 1], constrain='domain', title_font=dict(size=13), showgrid=False),
            yaxis=dict(range=[0, 1], scaleanchor='x', scaleratio=1, title_font=dict(size=13), showgrid=False),
            showlegend=True,
            legend=dict(
                x=-0.25,
                y=1.2),
            shapes=[
                # Create a rectangle for the top right quadrant
                dict(
                    type='rect',
                    xref='x',
                    yref='y',
                    x0=0.5,
                    y0=0.5,
                    x1=1,
                    y1=1,
                    fillcolor='yellow',
                    opacity=0.2,
                    line=dict(width=0)),
                
                # Create a rectangle for the top left quadrant
                dict(
                    type='rect',
                    xref='x',
                    yref='y',
                    x0=0,
                    y0=0.5,
                    x1=0.5,
                    y1=1,
                    fillcolor='red',
                    opacity=0.2,
                    line=dict(width=0)),
                
                # Create a rectangle for the bottom left quadrant
                dict(
                    type='rect',
                    xref='x',
                    yref='y',
                    x0=0,
                    y0=0,
                    x1=0.5,
                    y1=0.5,
                    fillcolor='blue',
                    opacity=0.2,
                    line=dict(width=0)),
                
                # Create a rectangle for the bottom right quadrant
                dict(
                    type='rect',
                    xref='x',
                    yref='y',
                    x0=0.5,
                    y0=0,
                    x1=1,
                    y1=0.5,
                    fillcolor='green',
                    opacity=0.2,
                    line=dict(width=0))],
      
                annotations=[
                
                # Add an annotation for the top right quadrant
                dict(
                    x=0.75,
                    y=0.75,
                    xref='x',
                    yref='y',
                    text='HAPPY 😃',
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='black',
                    )
                ),
                # Add an annotation for the top left quadrant
                dict(
                    x=0.25,
                    y=0.75,
                    xref='x',
                    yref='y',
                    text='ANGRY 😡',
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='black'
                    )
                ),
                # Add an annotation for the bottom left quadrant
                dict(
                    x=0.25,
                    y=0.25,
                    xref='x',
                    yref='y',
                    text='SAD 😢',
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='black'
                    )
                ),
                # Add an annotation for the bottom right quadrant
                dict(
                    x=0.75,
                    y=0.25,
                    xref='x',
                    yref='y',
                    text='RELAXED 😌',
                    showarrow=False,
                    font=dict(
                        size=14,
                        color='black'))])
                
            # Display the plot in Streamlit
            st.plotly_chart(fig, use_container_width=True, width=50, height=50)
                
        
        # Insert a gauge chart for tempo and loudness
        st.markdown('<div style="text-align:center;font-weight: bold;font-size: 28px;">Dynamics</div>', unsafe_allow_html=True)  
        
        col1, col2 = st.columns([3,3])
        
        # Define a function to create a gauge chart
        def create_gauge_chart(value, title, range_min, range_max, color, average_value):
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"<b>{title}</b>", 'font': {'size': 14, 'color': 'black'}},
                gauge = {'axis': {'range': [range_min, range_max]},
                         'bar': {'color': color},
                         'steps' : [
                             {'range': [range_min, range_max], 'color': 'lightgray'}],
                         'threshold' : {'line': {'color': 'black', 'width': 4}, 'thickness': 0.75, 'value': average_value},
                         'bordercolor': '#ff0000', 
                         'borderwidth': 0}))
            return fig
        
        # Define a callback function to update the charts when the dropdown selection changes
        def update_charts(song):
            tempo_value = songs.loc[songs['song_name'] == song, 'tempo'].values[0]
            loudness_value = songs.loc[songs['song_name'] == song, 'loudness'].values[0]
            
            
            # Calculate average tempo and loudness for top 20 songs
            avg_tempo = top_songs['tempo'].mean()
            avg_loudness = top_songs['loudness'].mean()
    
            # Create tempo and loudness gauge charts
            tempo_chart = create_gauge_chart(tempo_value, "Tempo", 0, 200, "green", avg_tempo)
            loudness_chart = create_gauge_chart(loudness_value, "Loudness", -60, 0, "green", avg_loudness)
            
            return tempo_chart, loudness_chart
        
        # Update the charts based on the selected song
        tempo_chart, loudness_chart = update_charts(search_song)
    
        
        # Display the gauge charts
        
        with col1:
            st.plotly_chart(tempo_chart, use_container_width=True, width=50, height=10)
        with col2:
            st.plotly_chart(loudness_chart, use_container_width=True, width=50, height=10)


# ------------------------------------TAB 2------------------------------------

def Song_Information():            
    
    # Define a function to add a star icon based on song popularity
    def add_star_icon(song_popularity):
        if song_popularity >= 75:
            return '⭐⭐⭐⭐'
        elif song_popularity >= 50:
            return '⭐⭐⭐'
        elif song_popularity >= 25:
            return '⭐⭐'
        else:
            return '⭐'

    # define a function to get the max sentiment and corresponding emoji
    def display_sentiment(df):
        # get the row for the given song_name
        song_row = df.loc[df['song_name'] == search_song]

        if not song_row.empty: 
        
            emotion_scores = {
            'Sadness': song_row['sentiment_sadness'].values[0],
            'Joy': song_row['sentiment_joy'].values[0],
            'Love': song_row['sentiment_love'].values[0],
            'Anger': song_row['sentiment_anger'].values[0],
            'Fear': song_row['sentiment_fear'].values[0],
            'Surprise': song_row['sentiment_surprise'].values[0]
            }
            max_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            max_emotion_name = max_emotion[0]
            max_emotion_score = max_emotion[1]
            
            if max_emotion_name == 'Sadness':
                emoji = '😔'
            elif max_emotion_name == 'Joy':
                emoji = '😄'
            elif max_emotion_name == 'Love':
                emoji = '❤️'
            elif max_emotion_name == 'Anger':
                emoji = '🤬'
            elif max_emotion_name == 'Fear':
                emoji = '😱'
            elif max_emotion_name == 'Surprise':
                emoji = '😲'
            
            return f"Sentiment: {max_emotion_score*100:.2f}% {emoji}"
        
    # Select the row corresponding to the selected song
    song_row = songs.loc[songs['song_name'] == search_song] 
    data_filter = data.loc[data['song_name'] == search_song]
    
    if not song_row.empty:
        # Extract the relevant information from the selected row
        name = song_row['song_name'].values[0]
        artist_name = song_row['artist_name'].values[0]
        popularity = song_row['popularity'].values[0]
        artist_followers = song_row['artist_followers'].values[0]
        lyrics = song_row['lyrics'].values[0]
        
        # Add a star icon based on song popularity
        star_icon = add_star_icon(popularity)

        # Call the sentiment value
        sentiment = display_sentiment(sentiment_songs)
        
        # Add a link
        spotify_id = song_row['id']
        spotify_url = f"https://open.spotify.com/track/{spotify_id.values[0]}"
        
        # Display the song information
        #text = f"<h1 style='display: inline-block; vertical-align: middle; line-height: 0.5;'>{name}</h1><p style='display: inline-block; vertical-align: middle; line-height: 0.5; margin-left: 10px;'><a href='{spotify_url}' style='font-size: 16px;'>Spotify Link</a></p>"
        container = """
           <div style='display: flex; align-items: center;'>
                <h1 style='margin: 1; font-size: 36px;'>{}</h1>
                <a href='{}' style='margin-left: -40px; font-size: 16px; padding-top: 12px;'>Spotify Link</a>
            </div>
        """

        st.markdown(container.format(name, spotify_url), unsafe_allow_html=True)
        
        col1, col2 = st.columns([3, 3])
        with col1:
            st.write('**Popularity:**', star_icon)
            st.write(sentiment)
        with col2:
            st.write('**Artist 👨‍🎤:**', artist_name)
            st.write('**Followers 🎉:**', '{:,}'.format(artist_followers))
        
        st.markdown('<div style="text-align:left;font-weight: bold;font-size: 22px;">Lyrics</div>', unsafe_allow_html=True) 
        for section in lyrics.split('\n\n'):
          if section.startswith('[') and section.endswith(']'):
              st.write(section)
          else:
              section = section.strip().lstrip('>')
              st.write(section)
        
    else:
        st.write("Sorry, we couldn't find any information for the selected song.")
    
    # Define a function to recommend songs based on the selected song
    nlp = spacy.load("en_core_web_sm")
    stop_words = nltk.download('stopwords')
    #stop_words = set(stopwords.words('english'))
    p_stemmer = PorterStemmer()

    def preprocess_text(text):
        '''Function to pass text into it in order to start the preprocessing. This includes lowecarse conversion, 
        re.substitution and deletion of special characters, tokenizations, stop word removal and stemming. 
        returns the lyrics paste up together after the preprocessing. '''
        
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        #Fix words
        text = contractions.fix(text)
        
        #Substitute 'til with until
        text = re.sub(r"\'til", 'until', text)
        
        #Remove everything inside brackets e.g '[Intro]'
        text = re.sub(r'\[.*?\]', '', text)
           
        # Keep only alphanumeric characters and some punctuation marks
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Tokenization
        text = nltk.word_tokenize(text)
        
        # Remove stop words
        text = [word for word in text if not word in stop_words]
        
        #Remove words with less than one length
        text = [word for word in text if len(word) > 1] 
        
        #Stemming
        text = [PorterStemmer().stem(t) for t in text]
        
        print ("tokens created correctly!")
        
        #Detokenize
        text = ' '.join(text)
        
        return text.strip()

    st.markdown('<div style="text-align:left;font-weight: bold;font-size: 22px;">Reccomended Songs for You...</div>', unsafe_allow_html=True) 

    def transformation_weighted_avg_emb(dataframe: pd.DataFrame, lyric: str =None):
        '''Function that serves the purpose of collecting the lyrics of 500+ songs and  create unique tokens 
        (appearing more than once) and furtherly creates word embeddings that will be multiplied by the frequency of the
        token / how many times it appears in a sentence. (token_weights * token_embeddings)''' 
        if lyric !=None: 
            #Pass the text (lyric by the preprocessing function!) For more info ?preprocess_text
            new_lyric=preprocess_text(lyric)
    
            # Append the new row to the dataframe
            new_song= {'id': 'newsong001', 'cleaned_lyrics':new_lyric }
            # Convert the new row to a dataframe
            new_song_df = pd.DataFrame(new_song, index=[0])
            
            #concat (append them) together, one below the rest.
            dataframe = pd.concat([dataframe, new_song_df], axis=0, ignore_index=True)
            #debug
            print("1 new lyric was added to the dataset!")
        
        # concatenate all the cleaned lyrics into a single string
        lyrics_string = ' '.join(dataframe['cleaned_lyrics'].tolist())
    
        # split the string into individual words
        words = lyrics_string.split()
    
        # count the frequency of each word
        word_counts = Counter(words)
    
        # remove the most frequent word and the words that appear only once
        most_common_word = word_counts.most_common(1)[0][0]
        unique_tokens = set(word for word in word_counts if word_counts[word] > 1 and word != most_common_word)
    
    
        # print the number of unique tokens
        print("example of tokens:", list(unique_tokens)[:10])
        
        
        #INITIAL MATRIX 
        
        #df with the necessary info for creating the matrix after...
        matrix=dataframe[['id','cleaned_lyrics']]
    
        
        #TOKENS AND ID
        # create a DataFrame with an id column and a token column
        df_tokens = pd.DataFrame(zip(range(len(unique_tokens)), unique_tokens), columns=['id', 'token'])
    
        # will contain all tokens with their 
        df_tokens = df_tokens.set_index('id') 
        
        #WORD2VEC AND FREQUENCY ( if tokens are words it will only take into account embeddings as frequency =1)
        
        # train a Word2Vec model on your lyrics data
        sentences = [lyrics.split() for lyrics in matrix['cleaned_lyrics']]
        embedding_size = len(df_tokens['token'])
        min_word_count = 5
        context_size = 5
        embedding_model = gensim.models\
        .Word2Vec(sentences, vector_size=embedding_size, min_count=min_word_count, window=context_size)
        
        # Print the vocabulary size
        print("vocabulary size: ", len(embedding_model.wv))
        
        # initialize an empty array to hold the weighted average embeddings
        embeddings = np.zeros((len(matrix), embedding_size))
    
        # loop over each token in the token list
        embedding_cols = []
        
        print('creating word embeddings')
        bar = st.progress(0)
   
        for i, token in enumerate(df_tokens['token']):
            # get the embeddings for this token
            try:
                token_embeddings = embedding_model.wv[token]
            except KeyError:
                # handle case where token is not in vocabulary
                token_embeddings = np.zeros(embedding_size)
            # calculate the weighted average embedding for this token
            counts = matrix['cleaned_lyrics'].str.count(token)
            token_weights = np.array(counts / counts.sum())
            weighted_embeddings = token_weights[:, np.newaxis] * token_embeddings
            # add this token's weighted embeddings to the overall embeddings array
            embeddings += weighted_embeddings

            # add the token name to the embedding_cols list
            embedding_cols.append(token)
            bar.progress(i/len((df_tokens['token'])))
    
        # add the embeddings array to the original matrix as new columns
        df_embeddings = pd.DataFrame(embeddings, columns=embedding_cols)
        matrix_df = pd.concat([matrix['id'], df_embeddings], axis=1)
    
        matrix_df=matrix_df.set_index('id').T
        #debugging
        print("Yay it worked!")
        
        return {'matrix': matrix_df, 'tokens': df_tokens}
    
    
    # Define similarity metric
    def cosine_similarity(u, v):
        '''the resulting value gives a measure of how similar the two vectors are in terms of direction
        or orientation. It's a commonly used measure in machine learning and natural language processing
        tasks, such as document similarity, recommender systems, and clustering'''
        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)
        if norm_u == 0 or norm_v == 0:
            similarity = 0
        else:
            similarity = np.dot(u, v) / (norm_u * norm_v)
        return similarity
    
    
    def recommendation(matrix, target_id:str= 'newsong001', N=10):
        '''The function first finds the index of the target song in the matrix and computes the cosine similarity
        between the target song and all other songs in the matrix. Cosine similarity is a measure of similarity between
        two vectors, and in this case, it is used to compare the attributes of songs in the matrix.
    
        The function then sorts the similarity scores in descending order and retrieves the indices of the top N most similar songs.
        These indices are used to retrieve the IDs of the recommended songs from the matrix, which are then returned
        in a DataFrame with their corresponding artist names and song names. '''
        # Compute similarity between target song and all other songs
        
        target_index = int(list(matrix.columns).index(target_id))
        similarity_scores = []
        for i in range(matrix.shape[1]):
            if target_index!=i:
                similarity_scores.append(cosine_similarity(matrix.iloc[:, i], matrix.iloc[:, target_index]))
            else: 
                similarity_scores.append(0)
    
            # Sort songs by similarity to target song
        sorted_indices = np.argsort(similarity_scores)[::-1]
    
        # Recommend top N most similar songs to target song
        Num= N
        recommendations = []
        for i in range(1, Num+1):
            recommendations.append(list(matrix.columns)[sorted_indices[i]])
        if target_id!='newsong001':
            print("Target song: ", data['song_name'][data['id']==target_id])
        
        return data[['artist_name','song_name']][data['id'].isin(recommendations)]
    
    # We will pass our initial dataframe and the new lyrics into our transformation function.
    Transformed=transformation_weighted_avg_emb(dataframe=df)
    
    selected_song= data[data['song_name']==search_song]
    target_id = selected_song['id'].values[0]
    recom = recommendation(matrix=Transformed['matrix'],target_id=target_id, N=5)

    st.table(recom.style.set_table_styles([{'selector': 'th', 'props': [('font-weight', 'bold')]}]).hide_index())
# ------------------------------------TAB 3------------------------------------

def Success_Factors(): 

    # Define a function to generate a word cloud for a given popularity label
   
    def generate_wordcloud(popularity_label):
        # Filter the data based on the popularity label
        filtered_df = bow_popularity[bow_popularity['popularity_label'] == popularity_label]

        # Group by word and sum the counts to get the frequency
        word_freq = filtered_df.groupby('word')['count'].sum().reset_index()
    
        # Convert the word_freq dataframe to a dictionary for wordcloud input
        word_dict = dict(zip(word_freq['word'], word_freq['count']))
    
        # Generate the word cloud
        wc = WordCloud(width=1000, height=400, background_color='white').generate_from_frequencies(word_dict)
    
        # Display the word cloud
        st.markdown('<div style="text-align:center;font-weight: bold;font-size: 28px;">Wordcloud</div>', unsafe_allow_html=True)  
        st.image(wc.to_array())
    
    # Create a slider for the popularity labels
    popularity_label = st.select_slider(
        "Select a Popularity Label",
        options=['very low', 'low', 'medium', 'high'],
        value='medium',
        format_func=lambda x: x.capitalize())
    
    # Generate the word cloud for the selected popularity label
    generate_wordcloud(popularity_label)

# ------------------------------------TAB 4------------------------------------
def Pop_Predict():

    def recco():

        # Define a function to recommend songs based on the selected song
        nlp = spacy.load("en_core_web_sm")
        stop_words = set(stopwords.words('english'))
        p_stemmer = PorterStemmer()

        def preprocess_text(text):
            '''Function to pass text into it in order to start the preprocessing. This includes lowecarse conversion, 
            re.substitution and deletion of special characters, tokenizations, stop word removal and stemming. 
            returns the lyrics paste up together after the preprocessing. '''
            
            # Convert to string
            text = str(text)
            
            # Convert to lowercase
            text = text.lower()
            
            #Fix words
            text = contractions.fix(text)
            
            #Substitute 'til with until
            text = re.sub(r"\'til", 'until', text)
            
            #Remove everything inside brackets e.g '[Intro]'
            text = re.sub(r'\[.*?\]', '', text)
                
            # Keep only alphanumeric characters and some punctuation marks
            text = re.sub(r"[^a-zA-Z\s]", "", text)
            
            # Tokenization
            text = nltk.word_tokenize(text)
            
            # Remove stop words
            text = [word for word in text if not word in stop_words]
            
            #Remove words with less than one length
            text = [word for word in text if len(word) > 1] 
            
            #Stemming
            text = [PorterStemmer().stem(t) for t in text]
            
            print ("tokens created correctly!")
            
            #Detokenize
            text = ' '.join(text)
            
            return text.strip()

        st.markdown('<div style="text-align:left;font-weight: bold;font-size: 22px;">Reccomended Songs for You...</div>', unsafe_allow_html=True)   

        def transformation_weighted_avg_emb(dataframe: pd.DataFrame, lyric: str =None):
            '''Function that serves the purpose of collecting the lyrics of 500+ songs and  create unique tokens 
            (appearing more than once) and furtherly creates word embeddings that will be multiplied by the frequency of the
            token / how many times it appears in a sentence. (token_weights * token_embeddings)''' 
            if lyric !=None: 
                #Pass the text (lyric by the preprocessing function!) For more info ?preprocess_text
                new_lyric=preprocess_text(lyric)

                # Append the new row to the dataframe
                new_song= {'id': 'newsong001', 'cleaned_lyrics':new_lyric }
                # Convert the new row to a dataframe
                new_song_df = pd.DataFrame(new_song, index=[0])
                
                #concat (append them) together, one below the rest.
                dataframe = pd.concat([dataframe, new_song_df], axis=0, ignore_index=True)
                #debug
                print("1 new lyric was added to the dataset!")
            
            # concatenate all the cleaned lyrics into a single string
            lyrics_string = ' '.join(dataframe['cleaned_lyrics'].tolist())

            # split the string into individual words
            words = lyrics_string.split()

            # count the frequency of each word
            word_counts = Counter(words)

            # remove the most frequent word and the words that appear only once
            most_common_word = word_counts.most_common(1)[0][0]
            unique_tokens = set(word for word in word_counts if word_counts[word] > 1 and word != most_common_word)


            # print the number of unique tokens
            print("example of tokens:", list(unique_tokens)[:10])
            
            
            #INITIAL MATRIX 
            
            #df with the necessary info for creating the matrix after...
            matrix=dataframe[['id','cleaned_lyrics']]

            
            #TOKENS AND ID
            # create a DataFrame with an id column and a token column
            df_tokens = pd.DataFrame(zip(range(len(unique_tokens)), unique_tokens), columns=['id', 'token'])

            # will contain all tokens with their 
            df_tokens = df_tokens.set_index('id') 
            
            #WORD2VEC AND FREQUENCY ( if tokens are words it will only take into account embeddings as frequency =1)
            
            # train a Word2Vec model on your lyrics data
            sentences = [lyrics.split() for lyrics in matrix['cleaned_lyrics']]
            embedding_size = len(df_tokens['token'])
            min_word_count = 5
            context_size = 5
            embedding_model = gensim.models\
            .Word2Vec(sentences, vector_size=embedding_size, min_count=min_word_count, window=context_size)
            
            # Print the vocabulary size
            print("vocabulary size: ", len(embedding_model.wv))
            
            # initialize an empty array to hold the weighted average embeddings
            embeddings = np.zeros((len(matrix), embedding_size))

            # loop over each token in the token list
            embedding_cols = []
            
            print('creating word embeddings')
            bar = st.progress(1)

            for i, token in enumerate(df_tokens['token']):
                # get the embeddings for this token
                try:
                    token_embeddings = embedding_model.wv[token]
                except KeyError:
                    # handle case where token is not in vocabulary
                    token_embeddings = np.zeros(embedding_size)
                # calculate the weighted average embedding for this token
                counts = matrix['cleaned_lyrics'].str.count(token)
                token_weights = np.array(counts / counts.sum())
                weighted_embeddings = token_weights[:, np.newaxis] * token_embeddings
                # add this token's weighted embeddings to the overall embeddings array
                embeddings += weighted_embeddings

                # add the token name to the embedding_cols list
                embedding_cols.append(token)
                bar.progress(i/len((df_tokens['token'])))

            # add the embeddings array to the original matrix as new columns
            df_embeddings = pd.DataFrame(embeddings, columns=embedding_cols)
            matrix_df = pd.concat([matrix['id'], df_embeddings], axis=1)

            matrix_df=matrix_df.set_index('id').T
            #debugging
            print("Yay it worked!")
            
            return {'matrix': matrix_df, 'tokens': df_tokens}


        # Define similarity metric
        def cosine_similarity(u, v):
            '''the resulting value gives a measure of how similar the two vectors are in terms of direction
            or orientation. It's a commonly used measure in machine learning and natural language processing
            tasks, such as document similarity, recommender systems, and clustering'''
            norm_u = np.linalg.norm(u)
            norm_v = np.linalg.norm(v)
            if norm_u == 0 or norm_v == 0:
                similarity = 0
            else:
                similarity = np.dot(u, v) / (norm_u * norm_v)
            return similarity


        def recommendation(matrix, target_id:str= 'newsong001', N=10):
            '''The function first finds the index of the target song in the matrix and computes the cosine similarity
            between the target song and all other songs in the matrix. Cosine similarity is a measure of similarity between
            two vectors, and in this case, it is used to compare the attributes of songs in the matrix.

            The function then sorts the similarity scores in descending order and retrieves the indices of the top N most similar songs.
            These indices are used to retrieve the IDs of the recommended songs from the matrix, which are then returned
            in a DataFrame with their corresponding artist names and song names. '''
            # Compute similarity between target song and all other songs
            
            target_index = int(list(matrix.columns).index(target_id))
            similarity_scores = []
            for i in range(matrix.shape[1]):
                if target_index!=i:
                    similarity_scores.append(cosine_similarity(matrix.iloc[:, i], matrix.iloc[:, target_index]))
                else: 
                    similarity_scores.append(0)

                # Sort songs by similarity to target song
            sorted_indices = np.argsort(similarity_scores)[::-1]

            # Recommend top N most similar songs to target song
            Num= N
            recommendations = []
            for i in range(1, Num+1):
                recommendations.append(list(matrix.columns)[sorted_indices[i]])
            if target_id!='newsong001':
                print("Target song: ", data['song_name'][data['id']==target_id])
            
            return data[['artist_name','song_name']][data['id'].isin(recommendations)]

        # We will pass our initial dataframe and the new lyrics into our transformation function.
        Transformed=transformation_weighted_avg_emb(dataframe=df, lyric=lyrics_input)
        recom = recommendation(matrix=Transformed['matrix'],target_id='newsong001', N=5)
        return recom
    
    def preprocess_text(text):
        # Convert to string
        text = str(text)
        
        # Convert to lowercase
        text = text.lower()
        
        #Fix words
        text = contractions.fix(text)
        
        #Substitute some words with expansions
        text = re.sub(r"\'til", 'until', text)
        text = re.sub(r"\'posed", 'supposed', text)
        text = re.sub(r"\'Don't", 'do not', text)
        text = re.sub(r"\'lyin'", 'lying', text)
        text = re.sub(r"\'dyin'", 'dying', text)
        text = re.sub(r"\'I'm'", 'I am', text)
        text = re.sub(r"\''Cause'", 'because', text)
        text = re.sub(r"\''you're'", 'you are', text)
        text = re.sub(r"\''swimmin''", 'swimming', text)
        
        #Remove everything inside brackets e.g '[Intro]'
        text = re.sub(r'\[[^]]*\]', "", text)
        text = re.sub(r'\([^()]*\)', "", text)
        
        #Replace dash punctuation
        text = text.replace("", " ")
        
        # Keep only alphanumeric characters and some punctuation marks
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        
        # Remove extra white spaces or line breaks
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r'\r', '', text) #get rid of \r
        text = re.sub(r'\n', '', text) #Get rid of \n

        # Tokenization
        text = word_tokenize(text)
        
        # Remove stop words
        additional_stop_words = ["yeah","Yee","eh", "ay","ya","na","wan","uh","gon","ima","mm","uhhuh","bout","em","nigga","niggas","got","ta","lil","ol","hey",
          "oooh","ooh","oh","youre","dont","im","youve","ive","theres","ill","yaka","lalalala","la","da","di","yuh",
          "shawty","oohooh","shoorah","mmmmmm","ook","shh","bro","ho","aint","cant","know","shitll","tonka"]
        stop_words = set(stopwords.words('english'))
        stop_words.update(additional_stop_words)
        text = [word for word in text if not word in stop_words]
        
        #Remove words with less than one length
        text = [word for word in text if len(word) > 1] 
        
        #Stemming
        text = [PorterStemmer().stem(t) for t in text]    
        
        #Detokenize
        text = ' '.join(text)
        
        return text.strip()

    def readability_1(text):
        y=len(text.split())
        while y<=100:
            text=text*2
            y=len(text.split())
        r = Readability(text)
        dc = r.dale_chall()
        return dc.score

    def sentiment_analysis(text):
        analyzer = SentimentIntensityAnalyzer()
        score = analyzer.polarity_scores(text)
        return score['compound']

     # defining classifier
    classifier = pipeline("text-classification",model='bhadresh-savani/distilbert-base-uncased-emotion', return_all_scores=True)

    # Define the bins and labels for each category
    bins = [0, 25, 50, 75, 100]
    labels = ['very low', 'low', 'medium', 'high']

    df_copy=songs.sort_values(by='popularity', ascending=False)
    df_copy['popularity_label'] = pd.cut(df_copy['popularity'], bins=bins, labels=labels, include_lowest=True)
    
    # Create LabelEncoder object
    le = LabelEncoder()

    # Fit and transform the target variable
    df_copy['target'] = le.fit_transform(df_copy['popularity_label'])

    def get_sentiment(text, n=1):
        if len(text)>1000:
            text=text[:1000]
        prediction = classifier(text)
        sentiment_score = prediction[0][n-1]['score']
        return sentiment_score

    def lexical_richness(text):
        tokens = nltk.word_tokenize(text)
        lexical_richness = len(set(tokens)) / len(tokens)
        return lexical_richness

    def prediction():

        #Preprocess
        clean_yl=preprocess_text(lyrics_input)

        #Sentiment analysis and readability 
        sentiment=sentiment_analysis(lyrics_input)
        readability=readability_1(lyrics_input)

        #Convert to data frame
        df_yl={'your_lyric':[lyrics_input],'cleaned_yl':[clean_yl],'readability_score':[readability], 'sentiment':[sentiment]}
        df_yl = pd.DataFrame(data=df_yl)

        #Get the sentiments
        df_yl['sentiment_sadness']=df_yl['your_lyric'].apply(get_sentiment, n=1)
        df_yl['sentiment_joy']=df_yl['your_lyric'].apply(get_sentiment, n=2)
        df_yl['sentiment_love']=df_yl['your_lyric'].apply(get_sentiment, n=3)
        df_yl['sentiment_anger']=df_yl['your_lyric'].apply(get_sentiment, n=4)
        df_yl['sentiment_fear']=df_yl['your_lyric'].apply(get_sentiment, n=5)
        df_yl['sentiment_surprise']=df_yl['your_lyric'].apply(get_sentiment, n=6)

        #Tokenize the lyrics
        df_yl['token_yl'] = df_yl['cleaned_yl'].apply(word_tokenize)

        df_yl['lexical_richness'] = df_yl['cleaned_yl'].apply(lexical_richness)

        #
        #Create dictionary
        dict_yl = corpora.Dictionary(df_yl['token_yl'])
        tempyl = dict_yl[0] #to initialize dictionary
        id2word_yl = dict_yl.id2token

        #Get the bag of words
        bow_yl = [dict_yl.doc2bow(doc) for doc in df_yl['token_yl']]

        # Define lda_model

        lda_model= LdaModel(corpus=bow_yl,
                         id2word=id2word_yl,
                         num_topics=13,
                         update_every=1,
                         chunksize=len(bow_yl),
                         passes=20,
                         alpha='auto',
                         random_state=42)

        # Get document topics for this lyrics with the pretrained model in the LDA part
        doc_topic_matrix = []
        for i, doc in enumerate(bow_yl):
            topic_probs = lda_model.get_document_topics(doc, minimum_probability=0)
            doc_topic_matrix.append([topic_probs[j][1] for j in range(lda_model.num_topics)])

        # Convert the matrix to a numpy array for easier manipulation
        doc_topic_matrix = np.array(doc_topic_matrix)

        #Convert matrix to DF to merge it with past data

        # Create column names for the TF-IDF matrix
        column_names = ['topic' + str(i) for i in range(1,14)]

        # Create a new dataframe with the TF-IDF matrix
        df_topics_yl = pd.DataFrame(doc_topic_matrix, columns=column_names)

        #Join with past dataframe
        df_yl = pd.concat([df_yl, df_topics_yl], axis=1)

        #Get the columns to get the prediction
        df_yl=df_yl[['readability_score','lexical_richness','sentiment','sentiment_sadness',
                     'sentiment_joy','sentiment_love','sentiment_anger','sentiment_fear',
                     'sentiment_surprise','topic1','topic2','topic3','topic4','topic5',
                     'topic6','topic7','topic8','topic9','topic10','topic11','topic12','topic13']]

        #Get only the columns we are interested in
        df_pred=df_pre_modelling.copy()[['readability_score','lexical_richness','sentiment','sentiment_sadness',
                   'sentiment_joy','sentiment_love','sentiment_anger','sentiment_fear',
                   'sentiment_surprise','target','topic1','topic2','topic3','topic4','topic5',
                   'topic6','topic7','topic8','topic9','topic10','topic11','topic12','topic13']]

        #Read ready to model X and Y set. 
        X=df_pred.copy().drop('target', axis=1)
        y=df_pred.copy()['target']

        #scikit-learn train-val split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
                
        # Best Model
        model = RandomForestClassifier(criterion='entropy', max_depth=8)

        # Fit the model with train set
        train_set = model.fit(X_train, y_train) 

        #Do prediction with model we trained in the modelling part
        prediction = model.predict(df_yl)

        #Get the pridction to a list to get the correct prediction based on the label
        list_pred=[]
        list_pred.append(prediction)

        return f"Your prediction for this lyric is {str(le.inverse_transform(list_pred)[0])} \n\
                The polarity is {str(sentiment)} \n\
                The readability is {str(readability)} \n\
                The sadness percentage is {str(df_yl['sentiment_sadness'][0])} 😔 \n\
                The joy percentage is {str(df_yl['sentiment_joy'][0])} 😊 \n\
                The love percentage is {str(df_yl['sentiment_love'][0])} ❤️ \n\
                The anger percentage is {str(df_yl['sentiment_anger'][0])} 😠 \n\
                The fear percentage is {str(df_yl['sentiment_fear'][0])} 😨 \n\
                The surprise percentage is {str(df_yl['sentiment_surprise'][0])} 😲"

    # Create the Streamlit interface
    st.header('Song Popularity Predictor')
    lyrics_input = st.text_area('Enter the lyrics:')
    submit_button = st.button('Submit')

    # When the user submits the lyrics, tokenize them and calculate the popularity score
    if submit_button:
        with st.spinner('Processing your lyrics...'):
            progress_bar = st.progress(0)
            result_prediction = None
            num_chunks = 10
            for i in range(num_chunks):
                # simulate some work
                time.sleep(1)
                progress_bar.progress((i+1) * 100 // num_chunks)
                if i == num_chunks - 1:
                    result_prediction = prediction()
            st.write(result_prediction)
        result = recco()
        st.table(result)      
        
# Run the tab informations               
if active_tab == "Key Attributes":
    Song_Attributes()
elif active_tab == "Song Information":
    Song_Information()
elif active_tab == "Success Factors":
    Success_Factors()
elif active_tab == "Popularity Prediction":
    Pop_Predict()
