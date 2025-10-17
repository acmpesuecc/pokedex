import pandas as pd
import streamlit as st
import os

if not os.path.exists('pokemon.db'):
    pokemon_db_csv_dict = pd.read_csv('pokedex_newer.csv').to_dict(orient='records')
    # Create the SQL connection to db as specified in your secrets file.
    conn = st.connection('pokemon_db', type='sql')
    # Insert some data with conn.session.
    with conn.session as s:
        s.execute('CREATE TABLE IF NOT EXISTS pokemon \
                (National_no TEXT,\
                Name TEXT,\
                type_number NUM,\
                type_1 TEXT,\
                type_2 TEXT,\
                Total NUM,\
                HP NUM,\
                Attack NUM,\
                Defense NUM,\
                Special_attack NUM,\
                Special_defense NUM,\
                Speed NUM,\
                Species TEXT,\
                Height TEXT,\
                Weight TEXT,\
                abilities_number NUM,\
                ability_1 TEXT,\
                ability_2 TEXT,\
                ability_hidden);')

        for record in pokemon_db_csv_dict:
            s.execute(
                'INSERT INTO pokemon (National_no, Name, type_number, type_1, type_2, Total, HP, Attack, Defense, Special_attack, Special_defense, Speed, Species, Height, Weight, abilities_number, ability_1, ability_2, ability_hidden) VALUES (:national_no, :name, :type_number, :type_1, :type_2, :total, :hp, :attack, :defense, :sp_attack, :sp_defense, :speed, :species, :height, :weight, :abilities_number, :ability_1, :ability_2, :ability_hidden);',
                params=
                dict(
                    national_no = record['National_no'], 
                    name = record['Name'], 
                    type_number = record['type_number'],
                    type_1 = record['type_1'],
                    type_2 = record['type_2'],
                    total = record['Total'],
                    hp = record['HP'],
                    attack = record['Attack'],
                    defense = record['Defense'],
                    sp_attack = record['Special_attack'],
                    sp_defense = record['Special_defense'],
                    speed = record['Speed'],
                    species = record['Species'],
                    height = record['Height'],
                    weight = record['Weight'],
                    abilities_number = record['abilities_number'],
                    ability_1 = record['ability_1'],
                    ability_2 = record['ability_2'],
                    ability_hidden = record['ability_hidden'],
                    )
            )
        s.commit()
        s.close()