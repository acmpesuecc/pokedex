import streamlit as st
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd


st.set_page_config(page_title="Stat Search", layout="wide")

# css file for displaying Pokemon type
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

@st.cache_resource
def get_data():
    conn = st.connection('pokemon_db', type='sql')
    with conn.session as s:
        df = pd.DataFrame(conn.query('select * from pokemon'))
        s.close()
    return df

local_css('style.css')
df = get_data()

st.title('📊 Pokémon Stat-Based Search')

# use Pokemon name and id to get image path
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
    elif 'Forme' in name or 'Cloak' in name or 'Form' in name:
        if 'Zygarde' in name:
            path = 'pokemon_images/' + str(id) + '.png'
        else:
            type = name.split()[1].lower()
            path = 'pokemon_images/' + str(id) + '-' + type + '.png'
    elif 'Primal' in name:
        type = name.split()[0].lower()
        path = 'pokemon_images/' + str(id) + '-' + type + '.png'
    elif 'Arceus' in name:
        path = 'pokemon_images/' + str(id) + '-normal.png'
    else:
        path = 'pokemon_images/' + str(id) + '.png'
    return path


# Sidebar for searching by stats range
with st.sidebar.form(key="stat_search_form"):
    st.subheader('Search Base Stats Range')
    min_speed, max_speed = st.select_slider('Speed', range(251), value=[0, 250])
    min_sp_def, max_sp_def = st.select_slider('Special Defense', range(251), value=[0, 250])
    min_sp_atk, max_sp_atk = st.select_slider('Special Attack', range(251), value=[0, 250])
    min_def, max_def = st.select_slider('Defense', range(251), value=[0, 250])
    min_atk, max_atk = st.select_slider('Attack', range(251), value=[0, 250])
    min_hp, max_hp = st.select_slider('HP', range(251), value=[0, 250])
    pressed = st.form_submit_button("Search Pokemon")

# Sidebar for finding similar Pokemon
st.sidebar.subheader('Find Similar Pokémon')
name_similar = st.sidebar.text_input('Enter Pokémon Name', '').lower()
matches_similar = list(df[df['Name'].str.lower().str.contains(name_similar)]['Name'])

if len(matches_similar) >= 1:
    selected_pokemon = st.sidebar.selectbox('Select Pokemon', matches_similar)
    find_similar = st.sidebar.button('Find Similar Pokémon')
else:
    find_similar = False


def display_similar_pokemons(match):
    df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
    df_stats_all = df[['Name', 'HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']].set_index('Name')
    
    diff_df = pd.DataFrame(df_stats_all.values - df_stats.values, index=df_stats_all.index)
    norm_df = diff_df.apply(np.linalg.norm, axis=1)
    similar_pokemons = norm_df.nsmallest(21)[1:22].index
    similar_pokemons_df = df_stats_all.loc[similar_pokemons]
    
    st.subheader(f'20 Most Similar Pokémon to {match["Name"].iloc[0]}')
    st.table(similar_pokemons_df)
    
    for row in similar_pokemons_df.iterrows():
        name = row[0]
        st.subheader(name)
        id = df[df.Name == name]['National_no'].iloc[0]
        
        try:
            path = get_image_path(name, id)
            image = Image.open(path)
            st.image(image, width=200)
        except:
            st.write('Image not available.')
        
        fig = px.line_polar(row[1], r=name, theta=row[1].index, line_close=True, range_r=[0, 255])
        st.plotly_chart(fig)


# Handle stat range search
if pressed:
    df_stats_all = df[['Name', 'HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']].set_index('Name')
    searched_pokemons_df = df_stats_all[
        (df_stats_all['HP'] >= min_hp) & (df_stats_all['HP'] <= max_hp) &
        (df_stats_all['Attack'] >= min_atk) & (df_stats_all['Attack'] <= max_atk) &
        (df_stats_all['Defense'] >= min_def) & (df_stats_all['Defense'] <= max_def) &
        (df_stats_all['Special_attack'] >= min_sp_atk) & (df_stats_all['Special_attack'] <= max_sp_atk) &
        (df_stats_all['Special_defense'] >= min_sp_def) & (df_stats_all['Special_defense'] <= max_sp_def) &
        (df_stats_all['Speed'] >= min_speed) & (df_stats_all['Speed'] <= max_speed)
    ]
    st.header('Search Results')
    st.write(f'Found {len(searched_pokemons_df)} Pokémon matching the criteria')
    st.table(searched_pokemons_df)

# Handle similarity search
elif find_similar and len(matches_similar) >= 1:
    match = df[df['Name'] == selected_pokemon]
    display_similar_pokemons(match)

else:
    st.info('👈 Use the sidebar to search for Pokémon by stat ranges or find similar Pokémon')

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
