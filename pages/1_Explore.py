import streamlit as st
import cv2
import pickle
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import os
from tensorflow.keras.models import load_model


st.set_page_config(page_title = "Pokédex", layout = "wide")

# css file for displaying Pokemon type (fire, water etc.)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        
@st.cache_resource
def get_data():
    conn = st.connection('pokemon_db', type='sql')
    with conn.session as s:
        # Query and display the data you inserted
        df = pd.DataFrame(conn.query('select * from pokemon'))
        s.close()
    return df

# load css file and get data
local_css('style.css')
df = get_data()

# sidebar configuration for searching Pokemon by name
st.sidebar.title('Pokédex')
name = st.sidebar.text_input('Search Name', '').lower() # input name
# find names that matches input and return it in a list
matches = list(df[df['Name'].str.lower().str.contains(name)]['Name'])
# dropdown menu with names that matches input
if len(matches) >= 1:
    name = st.sidebar.selectbox('Pokemon Matches', matches).lower()
else: # if no name matches input
    name = st.sidebar.selectbox('Pokemon Matches', ['No match'])

# filter row of data that matches Pokemon selected in dropdown menu
match = df[df['Name'].str.lower() == name]

# select information to view
info_list = ['Basic Information', 'Base Stats & Type Defenses', 'Radar Chart']
selected_info = st.sidebar.multiselect('View Information', info_list, default = info_list)

model = load_model('model.h5')
uploaded_file = st.sidebar.file_uploader("Choose a image file", type=["jpg","png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    test_img = cv2.imdecode(file_bytes, 1)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR)
    test_img = cv2.resize(test_img, (64,64))
    #plt.imshow(test_img)
    test_img = test_img[np.newaxis,...]

    output = model.predict(test_img)
    output_index = np.argmax(output)
    with open('target_classes.pickle', 'rb') as handle:
        target_classes = pickle.load(handle)
    output_name = target_classes[output_index][1]
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(test_img, width=300)
        st.subheader(f"Output class: {output_name}")

# search Pokemon using min and max base stats (speed, special defense etc.)
with st.sidebar.form(key="my_form"):
    st.subheader('Search Base Stats Range')
    min_speed, max_speed = st.select_slider('Speed', range(251), value = [0, 250])
    min_sp_def, max_sp_def = st.select_slider('Special Defense', range(251), value = [0, 250])
    min_sp_atk, max_sp_atk = st.select_slider('Special Attack', range(251), value = [0, 250])
    min_def, max_def = st.select_slider('Defense', range(251), value = [0, 250])
    min_atk, max_atk = st.select_slider('Attack', range(251), value = [0, 250])
    min_hp, max_hp = st.select_slider('HP', range(251), value = [0, 250])
    
    # pressed is a Boolean that becomes True when the "Search Pokemon" button is pressed
    # code to handle button pressing is all the way at the bottom
    pressed = st.form_submit_button("Search Pokemon")

# display credits on sidebar
st.sidebar.subheader('Info')
st.sidebar.markdown('Pokemon images taken from <a href="https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset">this Kaggle link</a>.', unsafe_allow_html = True)

# use Pokemon name and id to get image path, refer to 'pokemon_images' folder to see how images are named
def get_image_path(name, id):
    if 'Mega' in name:
        if name.endswith(' X'):
            path = 'pokemon_images/' + str(id) + '-mega-x.png'
        elif name.endswith(' Y'):
            path = 'pokemon_images/' + str(id) + '-mega-y.png'
        else:
            path = 'pokemon_images/' + str(id) + '-mega.png'
    elif 'Rotom' in name:
        rotom_type = name.split()[0].lower()
        path = 'pokemon_images/' + str(id) + '-' + rotom_type + '.png'
    elif 'Forme' in name or 'Cloak' in name  or 'Form' in name:
        if 'Zygarde' in name: # only 1 image present for Zygarde
            path = 'pokemon_images/' + str(id) + '.png'			
        else:
            type = name.split()[1].lower()
            path = 'pokemon_images/' + str(id) + '-' + type + '.png'
    elif 'Primal' in name:
        type = name.split()[0].lower()
        path = 'pokemon_images/' + str(id) + '-' + type + '.png'
    elif 'Arceus' in name: 
        path = 'pokemon_images/' + str(id) + '-normal.png' # this is just how Arceus is named in the image file
    else:
        path = 'pokemon_images/' + str(id) + '.png'
    return path
    

def display_basic_info(match):
    # get basic info data
    name = match['Name'].iloc[0]
    id = match['National_no'].iloc[0]
    height = str(match['Height'].iloc[0])
    weight = str(match['Weight'].iloc[0])
    species = ' '.join(match['Species'].iloc[0].split(' ')[:-1])
    type1 = match['type_1'].iloc[0]
    type2 = match['type_2'].iloc[0]
    type_number = match['type_number'].iloc[0]
    ability1 = match['ability_1'].iloc[0]
    ability2 = match['ability_2'].iloc[0]
    ability_hidden = match['ability_hidden'].iloc[0]
    
    st.title(name + ' #' + str(id).zfill(3))
    col1, col2, col3 = st.columns(3)
    
    # leftmost column col1 displays pokemon image
    try:
        path = get_image_path(name, id)
        image = Image.open(path)
        col1.image(image)	
    except: # output 'Image not available' instead of crashing the program when image not found
        col1.write('Image not available.')
    
    # middle column col2 displays nicely formatted Pokemon type using css loaded earlier
    with col2.container():		
        col2.write('Type')
        # html code that loads the class defined in css, each Pokemon type has a different style color
        type_text = f'<span class="icon type-{type1.lower()}">{type1}</span>'
        if type_number == 2:
            type_text += f' <span class="icon type-{type2.lower()}">{type2}</span>'
        # markdown displays html code directly
        col2.markdown(type_text, unsafe_allow_html=True)
        col2.metric("Height", height + " m")
        col2.metric("Weight", weight + " kg")
    
    # rightmost column col3 displays Pokemon abilities
    with col3.container():
        col3.metric("Species", species)
        col3.write('Abilities')
        # print(ability1, type(ability1))
        # print(ability2, type(ability2))
        # print(ability_hidden, type(ability_hidden))
        if ability1 != '' and ability1 != None:
            col3.subheader(ability1)
        if ability2 != '' and ability2 != None:
            col3.subheader(ability2)
        if ability_hidden != '' and ability_hidden != None:
            col3.subheader(ability_hidden + ' (Hidden)')

def display_base_stats_type_defenses(match):					
    with st.container():	
        col1, col2 = st.columns(2)	
        
        # left column col1 displays horizontal bar chart of base stats
        col1.subheader('Base Stats')
        # get base stats of Pokemon and rename columns nicely
        df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
        df_stats = df_stats.T
        df_stats.columns=['stats']
        
        # plot horizontal bar chart using matplotlib.pyplot
        fig, ax = plt.subplots()
        ax.barh(y = df_stats.index, width = df_stats.stats)
        plt.xlim([0, 250])
        col1.pyplot(fig)
            
def display_radar_chart(match):
    st.header('Radar Chart of Base Stats')
    
    # get base stats of Pokemon and rename columns nicely
    df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
    df_stats = df_stats.T
    df_stats.columns=['stats']
    
    # use plotly express to plot out radar char of stats
    fig = px.line_polar(df_stats, r='stats', theta=df_stats.index, line_close=True, range_r=[0, 250])
    st.plotly_chart(fig)
    
    if st.button('Search for Pokemons with Similar Base Stats'):
        display_similar_pokemons(match)

def display_similar_pokemons(match):
    # get base stats of Pokemon and rename columns nicely
    df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
    
    # get stats of all other Pokemon in the full dataframe
    df_stats_all = df[['Name', 'HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']].set_index('Name')
    
    # find difference between stat of Pokemon and each of the other Pokemons
    diff_df = pd.DataFrame(df_stats_all.values - df_stats.values, index = df_stats_all.index)
    
    # find norm 'distance' between this Pokemon and all other Pokemon
    norm_df = diff_df.apply(np.linalg.norm, axis=1)
    # find 20 other Pokemon with smallest distance, i.e. with most similar base stats to this Pokemon
    similar_pokemons = norm_df.nsmallest(21)[1:22].index # index [1:22] so it does not show itself	
    # store all similar Pokemon with their stats in df
    similar_pokemons_df = df_stats_all.loc[similar_pokemons]
    st.write(similar_pokemons_df)
    # display name, image, radar chart of each similar Pokemon
    for row in similar_pokemons_df.iterrows():
        name = row[0]
        st.subheader(name) # display Pokemon name
        id = df[df.Name == name]['National_no'].iloc[0]
        # display Pokemon image
        try:
            path = get_image_path(name, id)
            image = Image.open(path)
            st.image(image)
        except:
            st.write('Image not available.')
        
        # display radar chart	
        fig = px.line_polar(row[1], r=name, theta=row[1].index, line_close=True, range_r=[0, 255])
        st.plotly_chart(fig)
    
    # display full table of all 20 similar Pokemons and their stats
    st.subheader('20 Most Similar Pokemons')
    st.table(similar_pokemons_df)

# if "Search Pokemon" button is not pressed,
# i.e. we are searching Pokemon by name instead of by base stats slider section	
if not pressed:
    if len(match) == 0:
        st.write('Enter name to search for details.')
    
    # display information of Pokemon according to the information the user selects in the sidebar multiselector
    elif len(match) == 1:
        if 'Basic Information' in selected_info:
            display_basic_info(match)
        if 'Base Stats & Type Defenses' in selected_info:
            display_base_stats_type_defenses(match)
        if 'Radar Chart' in selected_info:
            display_radar_chart(match)
            
# if "Search Pokemon" button IS PRESSED,
# filter Pokemon according to the base stats in the slider
# display all the matched Pokemon and their stats in a table
# we do not display every bit of information about the search Pokemons as the list may be huge and the app can hang
else:
    # get base stats of all Pokemon
    df_stats_all = df[['Name', 'HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']].set_index('Name')
    # df_stats_all = df_stats_all.rename(columns={'hp': 'HP', 'attack': 'Attack', 'defense': 'Defense', 'sp_attack': 'Special Attack', 'sp_defense': 'Special Defense', 'speed': 'Speed'})
    # filter stats according to search criteria from the sliders
    searched_pokemons_df = df_stats_all[
        (df_stats_all['HP'] >= min_hp) & (df_stats_all['HP'] <= max_hp) &
        (df_stats_all['Attack'] >= min_atk) & (df_stats_all['Attack'] <= max_atk) &
        (df_stats_all['Defense'] >= min_def) & (df_stats_all['Defense'] <= max_def) &
        (df_stats_all['Special_attack'] >= min_sp_atk) & (df_stats_all['Special_attack'] <= max_sp_atk) &
        (df_stats_all['Special_defense'] >= min_sp_def) & (df_stats_all['Special_defense'] <= max_sp_def) &
        (df_stats_all['Speed'] >= min_speed) & (df_stats_all['Speed'] <= max_speed)										
    ]
    st.header('Pokemon Search Using Base Stats')
    st.table(searched_pokemons_df)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) 