# Smart Pokédex: First Generation Pokémon Classifier

The Smart Pokédex is a web application designed to provide information of all the avaiable pokemon (around 1000) and image recognize the first 151 (Generation 1). It uses **TensorFlow** for image recognition, **SQLite** for data management, and **Streamlit** for a user-friendly interface. The project uses a Convolutional Neural Network (CNN) trained on Pokémon images to classify them accurately.

(If you are going to proceed with this project, please note you will need a GPU (be it local or cloud based for the running and training the CNN model)
(PS: having a local GPU will be preferred instead of u using cloud alternatives like colab or kaggle, but hey it's up to u!!)


## Features

- **Image Recognition**: Classifies Pokémon images using a trained CNN model.
- **Pokémon Information**: Displays detailed information about each Pokémon, including stats, types, abilities, and more.
- **User-Friendly Interface**: Built with Streamlit for an intuitive and interactive experience.
- **Database Management**: Uses SQLite to store and manage Pokémon data efficiently.


## Technologies Used

- **TensorFlow**: For building and training the CNN model for image classification.
- **SQLite**: For storing and querying Pokémon data.
- **Streamlit**: For creating the web application interface.
- **Pandas**: For data manipulation and preprocessing.
- **BeautifulSoup**: For web scraping Pokémon data.
- **Matplotlib**: For visualizing training results and data.
- **OpenCV**: For image processing tasks.


## Installation

To run the Smart Pokédex locally, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/smart-pokedex.git
   cd smart-pokedex

2. **Create a virtual environment**:
Make sure you have Python 3.10+ installed. Create and activate a python environment to run this project in.

   ```bash
   python -m venv <name_of_your_virtual_environment>
   source <name_of_your_virtual_environment>/bin/activate

3. **Install dependencies**:
   Install the required libraries using pip by reading through source code
   (And if a requirements.txt file has been created, install that directly using the below cmd)
   
      ```bash
      pip install -r requirements.txt

5. **Download Dataset**:
Download the Pokémon image dataset for the main webpage from [here](https://www.kaggle.com/datasets/kvpratama/pokemon-images-dataset) and save it in a folder named pokemon_images in the main directory of the project.
Download the Pokémon image training dataset for the CNN from [here](https://www.kaggle.com/datasets/thedagger/pokemon-generation-one) and save it in the main directory of the project.

7. **Web scrape the data**
Run the scrape.py script to create the csv file pokemon_newer.csv

4. **Set Up the Database**:
Run the create_db_table.py script to create and populate the SQLite database:
    ```bash
    python create_db_table.py
    
5. **Train the Model**:
1st update the file paths in the model_training.py file (to reflect your local file architecture).
Then run this file to get a model.h5 file
    ```bash
    python model_training.py

7. **Run the Application**:
Start the Streamlit app by running:

    ```bash
    streamlit run Pokedex.py

## Usage
- **Launch the Application**:
After running the Streamlit app, open the provided URL in your browser to access the Pokédex.

- **View Pokémon Information**:
Browse through the database to view detailed information about all Pokémon.

- **Classify Pokémon Images**:
Use the interface to upload an image of a Pokémon. The app will classify the image only if it belongs to one of the first 151 Pokémon. If the Pokémon is not in the first generation, the classifier will not recognize it (in the sense it will give its approxiamte guess within the 151 pokemon it's abel to recognize).


## Limitations
- **Image Classifier**: The image classifier is trained only on the first 151 Pokémon (Generation 1). It will not recognize Pokémon from other generations.
- **Dataset**: Ensure the training and testing datasets are correctly placed in the PokemonData_train and PokemonData_test directories.


## Future Improvements
- Expand the image classifier to include Pokémon from other generations.
- Add more advanced features, such as battle simulations or team-building tools.
- Improve the UI/UX with additional interactive elements.




