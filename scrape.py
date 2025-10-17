import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

base = "https://pokemondb.net"
url = "https://pokemondb.net/pokedex/all"
response = requests.get(url) 
soup = BeautifulSoup(response.content, 'html.parser')

header = soup.find('h1')
print(header)
#obtain webpage intro
intro = soup.find('div', attrs={"class": "panel panel-intro"})
print(intro)

mapping_pokemon_all_table = {
    0: 'National_no',
    1: 'Name',
    2: 'Types',
    3: 'Total',
    4: 'HP',
    5: 'Attack',
    6: 'Defense',
    7: 'Special_attack',
    8: 'Special_defense',
    9: 'Speed',
}

mapping_vitals_table = {
    2: 'Species',
    3: 'Height',
    4: 'Weight',
    5: 'Abilities'
}

#not using special words currently
special_words = ['Mega', 'Alolan', 'Partner', 'Galarian', 'Hisuian', 'Combat', 'Blaze', 'Aqua', 
                 'Paldean', 'Sunny', 'Rainy', 'Snowy', 'Primal', 'Normal', 'Attack', 'Defense', 'Speed', 'Plant', 'Sandy', 'Tra']

pokedex = soup.find('table', id="pokedex").find_all('tr')

pokemon_all_table = []
for i in range(1,len(pokedex)):
    temp = pokedex[i]
    record = dict()
    for index, colname in mapping_pokemon_all_table.items():
        record[colname] = temp.find_all('td')[index].get_text()
    link = base+temp.find_all('a', href=True)[0]['href']
    # print(link)
    sub_response = requests.get(link) 
    sub_soup = BeautifulSoup(sub_response.content, 'html.parser')
    vitals_table = sub_soup.find('table', class_="vitals-table").find_all('tr')
    for sub_index, sub_colname in mapping_vitals_table.items():
        record[sub_colname] = vitals_table[sub_index].find('td').get_text()
    pokemon_all_table.append(record)


pokedf = pd.DataFrame(pokemon_all_table)
pokedf.to_csv('pokedex_newer.csv')