import streamlit as st

st.set_page_config(
    page_title="Pokedex",
    # page_icon="",
)

# st.write("# This is a Pokedex web app!")

# st.sidebar.success("Select a demo above.")

# st.markdown(
#     """
#     Pokémon, electronic game series from Nintendo that debuted in Japan in February 1996 as Pokémon Green and Pokémon Red. The franchise later became wildly popular in the United States and around the world. The series, originally produced for the company’s Game Boy line of handheld consoles, was introduced in 1998 to the United States with two titles, known to fans as Red and Blue. In the games, players assume the role of Pokémon trainers, obtaining cartoon monsters and developing them to battle other Pokémon. Pokémon became one of the most successful video game franchises in the world, second only to Nintendo’s Super Mario Bros.

#     The original Pokémon is a role-playing game based around building a small team of monsters to battle other monsters in a quest to become the best. Pokémon are divided into types, such as water and fire, each with different strengths. Battles between them can be likened to the simple hand game rock-paper-scissors. For example, to gain an advantage over a Pokémon that cannot beat an opponent’s Charizard character because of a weakness to fire, a player might substitute a water-based Pokémon. With experience, Pokémon grow stronger, gaining new abilities. By defeating Gym Leaders and obtaining Gym Badges, trainers garner acclaim.

# """
# )

page_bg_img = '''
<style>
.stApp {
background-image: url("https://images.unsplash.com/photo-1638613067237-b1127ef06c00?auto=format&fit=crop&q=80&w=2982&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
background-size: 100%;
background-repeat: no-repeat;
background-position: center;

}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)