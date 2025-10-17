import streamlit as st
import pickle
from PIL import Image
import numpy as np
import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt


st.set_page_config(page_title="Explore Pokémon", layout="wide")

# css file for displaying Pokemon type (fire, water etc.)
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

# load css file and get data
local_css('style.css')
df = get_data()

# sidebar configuration for searching Pokemon by name
st.sidebar.title('🔍 Explore Pokémon')
name = st.sidebar.text_input('Search Name', '').lower()
matches = list(df[df['Name'].str.lower().str.contains(name)]['Name'])

if len(matches) >= 1:
    name = st.sidebar.selectbox('Pokemon Matches', matches).lower()
else:
    name = st.sidebar.selectbox('Pokemon Matches', ['No match'])

match = df[df['Name'].str.lower() == name]

# select information to view
info_list = ['Basic Information', 'Base Stats & Type Defenses', 'Radar Chart']
selected_info = st.sidebar.multiselect('View Information', info_list, default=info_list)

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


def display_basic_info(match):
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

    try:
        path = get_image_path(name, id)
        image = Image.open(path)
        col1.image(image)
    except:
        col1.write('Image not available.')

    with col2.container():
        col2.write('Type')
        type_text = f'<span class="icon type-{type1.lower()}">{type1}</span>'
        if type_number == 2:
            type_text += f' <span class="icon type-{type2.lower()}">{type2}</span>'
        col2.markdown(type_text, unsafe_allow_html=True)
        col2.metric("Height", height + " m")
        col2.metric("Weight", weight + " kg")

    with col3.container():
        col3.metric("Species", species)
        col3.write('Abilities')
        if ability1 != '' and ability1 != None:
            col3.subheader(ability1)
        if ability2 != '' and ability2 != None:
            col3.subheader(ability2)
        if ability_hidden != '' and ability_hidden != None:
            col3.subheader(ability_hidden + ' (Hidden)')


def display_base_stats_type_defenses(match):
    with st.container():
        col1, col2 = st.columns(2)
        col1.subheader('Base Stats')
        df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
        df_stats = df_stats.T
        df_stats.columns = ['stats']

        fig, ax = plt.subplots()
        ax.barh(y=df_stats.index, width=df_stats.stats)
        plt.xlim([0, 250])
        col1.pyplot(fig)


def display_radar_chart(match):
    st.header('Radar Chart of Base Stats')
    df_stats = match[['HP', 'Attack', 'Defense', 'Special_attack', 'Special_defense', 'Speed']]
    df_stats = df_stats.T
    df_stats.columns = ['stats']

    fig = px.line_polar(df_stats, r='stats', theta=df_stats.index, line_close=True, range_r=[0, 250])
    st.plotly_chart(fig)


# Display information
if len(match) == 0:
    st.write('Enter name to search for details.')
elif len(match) == 1:
    if 'Basic Information' in selected_info:
        display_basic_info(match)
    if 'Base Stats & Type Defenses' in selected_info:
        display_base_stats_type_defenses(match)
    if 'Radar Chart' in selected_info:
        display_radar_chart(match)

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
