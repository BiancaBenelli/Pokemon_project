# py -m streamlit run pokemon_streamlit.py

import pandas as pd
import numpy as np
import clean_dataset as cl # I import the Python file that cleans the dataframe
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
import squarify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score

# Create a dynamic page
st.set_page_config(layout= 'centered')
st.header('Pokemon project')
st.subheader('To be like Professor Oak!')
#st.write('Use the sidebar to choose what to visualize.')

pokemon_df = pd.read_csv('pokemon.csv')

# I use a sidebar to keep the Eda part of the project separate from the plots and models
# I split the EDA between before and after the cleaning
st.sidebar.write('What do you want to see?')
if st.sidebar.checkbox('EDA'):
    st.title('EDA')
    st.markdown("""The dataframe analyzed in this project deals with Pokemon and its columns can be split into 4 big groups:
##### Pokedex Data:
    - pokedex_number: The entry number of the Pokemon in the National Pokedex
    - name: The English name of the Pokemon
    - german_name: The German name of the Pokemon
    - japanese_name: The Original Japanese name of the Pokemon
    - generation: The numbered generation which the Pokemon was first introduced
    - status: Denotes if the Pokemon is normal, sub-legendary, legendary or mythical
    - species: The Categorie of the Pokemon
    - type_number: Number of types that the Pokemon has
    - type_1: The Primary Type of the Pokemon
    - type_2: The Secondary Type of the Pokemon if it has it
    - height_m: Height of the Pokemon in meters
    - weight_kg: The Weight of the Pokemon in kilograms
    - ability_1: The Primary Ability of the Pokemon
    - ability_2: The Secondary Ability of the Pokemon if it has it
    - ability_hidden: Name of the hidden ability of the Pokemon if it has one
##### Base stats:
    - total_points: Total number of Base Points
    - hp: The Base HP of the Pokemon
    - attack: The Base Attack of the Pokemon
    - defense: The Base Defense of the Pokemon
    - sp_attack: The Base Special Attack of the Pokemon
    - sp_defense: The Base Special Defense of the Pokemon
    - speed: The Base Speed of the Pokemon
##### Training:
    - catch_rate: Catch Rate of the Pokemon
    - base_friendship: The Base Friendship of the Pokemon
    - base_experience: The Base experience of a wild Pokemon when caught
    - growth_rate: The Growth Rate of the Pokemon
##### Breeding:
    - egg_type_number: Number of groups where a Pokemon can hatch
    - egg_type_?: Names of the egg groups where a Pokemon can hatch
    - percentage_male: The percentage of the species that are male. Blank if
    the Pokemon is genderless
    - egg_cycles: The number of cycles (255-257 steps) required to hatch an
    egg of the Pokemon
##### Type defenses:
    - against_?: Eighteen features that denote the amount of damage taken against an
    attack of a particular type
"""
)
    st.write('Choose to see the EDA of the dataframe before or after its cleaning:')
    if st.checkbox('BEFORE CLEANING'):
        st.write('Pokemon dataframe:')
        st.write(pokemon_df)
        st.write('Rows and columns:', pokemon_df.shape)
        
        st.write('Dataframe head and tail:')
        st.write(pokemon_df.head())
        st.write(pokemon_df.tail())

        st.write('Some numerical informations:')
        st.write(pokemon_df.describe())
    
    # Now I work with the clean dataset -> no null values
    pokemon_df = cl.return_clean_dataset()
    if st.checkbox('AFTER CLEANING'):
        st.write('How the dataframe was cleaned and why can be seen on the Google Colab file on GitHub')
        st.write('Pokemon dataframe:')
        st.write(pokemon_df)
        st.write('Rows and columns:', pokemon_df.shape)

        st.write('Pokemon head and tail:')
        st.write(pokemon_df.head())
        st.write(pokemon_df.tail())

        st.write('Some numerical informations:')
        st.write(pokemon_df.describe())

# Plots and models are made working on the clean dataframe
pokemon_df = cl.return_clean_dataset()
if st.sidebar.checkbox('Plots'):
    st.title('PLOTS')
    # I use a selectbox to create a drop-down menu to be able to chose which plot to see
    plot = st.selectbox('Which plot would you like to be see?',
        ('How many Pokemon per generation', 'Pokemon statuses per generation', 'Type 1 frequency', 'Type 2 frequency',
        'Ability 1 frequency', 'Ability 2 frequency', 'Strongest Pokemon', 'Correlation', 'Pokedex number and generation',
        'Height and weight', 'Top 10 correlations'))
    if plot == 'How many Pokemon per generation':
        st.write('As the title says, this is a graph to show how many Pokemon each generation has:')
        # [8, 2] is to chose the dimension of the 2 columns
        col_1, col_2 = st.columns([8, 2])
        pokemon_per_generation = pokemon_df['generation'].value_counts()
        # In the first column I have the plot
        with col_1:
            fig = plt.figure(figsize = (10, 8))
            plt.title('How many Pokemon per generation?', fontsize= 18, fontweight='bold')
            plt.barh(pokemon_per_generation.index, pokemon_per_generation.values, color=['firebrick', 'turquoise', 'gold', 'limegreen', 'darkorange', 'palevioletred', 'darkviolet', 'cornflowerblue'])
            plt.xlabel('Number of Pokemon')
            plt.ylabel('Generation')
            st.write(fig)
        # In the second column I have the value counts of the above plot
        with col_2:
            st.write(pokemon_per_generation)
        st.write('As it could be expected, the first generation has more Pokemon than the others, but it is closely followed by the fifth and third.')

    status_list = pokemon_df['status'].value_counts().index
    if plot == 'Pokemon statuses per generation':
        st.write('A study of how in each generation the statuses of Pokemon are distributed among the eight generations')
        st.write('"Status" means Normal, Mythical, Sub Legendary and Legendary Pokemon.')
        st.markdown("""-A Donut chart was utilized:""")
        fig, axs = plt.subplots(nrows = 4, ncols = 1, figsize = (40, 20), constrained_layout = True)
        fig.suptitle('Pokemon statuses per generation', fontsize = 18, fontweight = 'bold')
        for i, ax in zip(range(len(status_list)), axs.flat):
            status_mask = pokemon_df[pokemon_df['status'] == status_list[i]]
            status_per_gen = status_mask['generation'].value_counts()
            donut_circle = plt.Circle( (0,0), 0.45, color = 'white')
            ax.pie(status_per_gen.values, labels = status_per_gen.index, autopct='%.2f%%', colors = sb.color_palette("Paired", len(status_per_gen.index)), labeldistance=None)
            ax.add_artist(donut_circle)
            handles, labels = ax.get_legend_handles_labels()
            handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key = lambda k: list(map(int, labels))[k])] )
            ax.legend(handles, labels, title = 'Generation:', bbox_to_anchor=(1, 1))
            ax.set_title(status_list[i], fontsize = 14, fontweight='bold')
        st.write(fig)
        st.write('It can be seen that in the seventh generation there are no Sub Legendary Pokemon.')
        st.write('The Normal status is the one who is disrtibuted more evenly throught the generations (beside the eight).')

        # I use a checkbox to show other plots
        if st.checkbox('Do you want to see other plots that could be used?'):
            st.markdown("""- A Donut chart shows clearly the wanted info, but a Barchart could also be efficient:""")
            fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 20), constrained_layout = True)
            for i, ax in zip(range(len(status_list)), axs.flat):
                status_mask = pokemon_df[pokemon_df['status'] == status_list[i]]
                status_per_gen = status_mask['generation'].value_counts()
                ax.bar(status_per_gen.index, status_per_gen.values, label = status_list[i], color = sb.color_palette("Paired", len(status_per_gen.index)))
                ax.tick_params(axis = 'both', labelsize = 15)
                ax.set_title(status_list[i], fontsize = 20, fontweight='bold')
            st.write(fig)
            st.write('The Barchart is not the best choice due to the fact that the values range too differently based on the status: if the same proportion was kept for all the subplots, only the Normal one would be readable. Nonetheless, by havng the proportions change for each subplots, the plot does not truly show the wanted info.')

            st.markdown("""- A Heatmap could otherwise be used:""")
            special_mask = pokemon_df[pokemon_df['status'] != 'Normal']
            status_per_gen = special_mask.groupby(['generation', 'status']).size().unstack(fill_value = 0)
            fig= plt.figure(figsize = (10, 8))
            sb.heatmap(status_per_gen, annot= True, cmap="Blues", fmt="d")
            plt.xlabel('Status', fontsize = 12)
            plt.ylabel('Generation', fontsize = 12)
            st.write(fig)
            st.write('The Normal status was removed since it spiked the results (there are way more "normal" Pokemon than "special" ones, so by keeping them in the heatmap would have been pratically all white besides for the Normal values.')

    if plot == 'Type 1 frequency':
        st.write('This Treemap shows how many times each Type 1 appears throught the dataframe:')
        type_1_frequency = pokemon_df['type_1'].value_counts()
        fig = plt.figure(figsize = (15, 10))
        plt.title('Type 1 frequencies', fontsize = 18, fontweight = 'bold')
        squarify.plot(sizes = type_1_frequency.values, label = type_1_frequency.index, value = type_1_frequency.values,
                alpha = 0.9, color=sb.color_palette('Paired'), pad = 1)
        plt.axis('off')
        st.write(fig)
        st.write('Water is the most frequent Type 1, followed by Normal and Grass.')
        st.write('The least frequent are Flying and Fairy.')

        if st.checkbox('Do you want to see other plots that could be used?'):
            st.write('Since I have quite a few values of Type 1, a Pie chart would have been quite confusing (but it still works):')
            fig = plt.figure(figsize=(10, 8))
            plt.title('Type 1 frequencies', fontsize = 18, fontweight = 'bold')
            plt.pie(type_1_frequency, labels=type_1_frequency.index, autopct='%.2f%%', startangle=90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },  labeldistance=None)
            plt.legend(bbox_to_anchor = (1.5, 1))
            plt.tight_layout()
            st.write(fig)

    type_2_present_mask = pokemon_df[pokemon_df['type_2'] != 'None']
    type_list = type_2_present_mask['type_1'].value_counts().index
    if plot == 'Type 2 frequency':
        st.write('This is a study about the frequency of Type 2 in relation to which Type 1 the Pokemon has.')
        st.markdown("""- Firstly, a Pie chart shows which Pokemon have a Type 2 and which do not (the next plots are applied to the formers).""")
        type_df_len = [(len(pokemon_df.index) - len(type_2_present_mask.index)), len(type_2_present_mask.index)]
        name = ['Pokemon without a Type 2', 'Pokemon with a Type 2']
        colors = ['#B7C3F3', '#8EB897']
        fig = plt.figure(figsize = (12, 8))
        plt.title('How many Pokemon have a Type 2?', fontsize = 18, fontweight = 'bold')
        plt.pie(type_df_len, labels = name, autopct='%.2f%%', startangle = 90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors = colors)
        st.write(fig)
        st.write('It is almost an even split between Pokemon who have a Type 2 and those who do not.')

        st.write("""- Now onto the frequency of the Types 2 wih a Treemap (as it was done for Type 1):""")
        type_2_frequency = type_2_present_mask['type_2'].value_counts()
        fig = plt.figure(figsize = (15, 10))
        plt.title("Type 2 frequencies",  )
        squarify.plot(sizes = type_2_frequency.values, label = type_2_frequency.index, value = type_2_frequency.values,
                alpha = 0.9, color=sb.color_palette("Paired"), pad=1)
        plt.axis('off')
        st.write(fig)
        st.write('Flying is the most frequent Type 1, followed by Psychic and Ground.')
        st.write('The least frequent are Flying and Fairy.')
        
        st.markdown("""- Finally a plot for the frequency of Types 2 in relation to its Type 1:""")
        fig, axs = plt.subplots(nrows = 18, ncols = 1, figsize = (15, 45), constrained_layout = True)
        for i, ax in zip(range(len(type_list)), axs.flat):
            type_1 = type_2_present_mask[type_2_present_mask['type_1'] == type_list[i]]
            type_2_frequency = type_1['type_2'].value_counts()
            ax.bar(type_2_frequency.index, type_2_frequency.values, label = type_list[i], color = sb.color_palette("Paired", len(type_2_frequency.index)))
            ax.set_xlabel('Type 2', fontsize = 10)
            ax.set_ylim([0, 25])
            ax.set_xticks(range(15)) 
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_title(type_list[i] + " Type 1", fontsize = 20, fontweight='bold')
        st.write(fig)
        st.write('Each subplots show how frequent are the Types 2 for each Type 1. For example, for the Type 1 Water (the most frequent), Ground and Flying are the most recurring Types 2.')
        st.write('Instead, the Types 1 Fairy and Flying do not present many Type 2.')

        if st.checkbox("Do you want to see other plots that could be used?"):
            st.write("For these alternative plots, only the Water Type 1 was studied to make it simpler.")
            st.markdown("""- With a Pie chart:""")
            water_type_1 = type_2_present_mask[type_2_present_mask['type_1'] == 'Water']
            type_1_water_frequency = water_type_1['type_2'].value_counts()
            fig = plt.figure(figsize=(10, 8))
            plt.pie(type_1_water_frequency.values, labels = type_1_water_frequency.index, autopct='%.2f%%', startangle=90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })
            plt.legend(bbox_to_anchor=(1.5, 1))
            st.write(fig)

            st.markdown("""- With a Heatmap:""")
            special_mask = pokemon_df[pokemon_df['type_2'] != 'None']
            status_per_gen = special_mask.groupby(['type_1', 'type_2']).size().unstack()
            fig = plt.figure(figsize = (10, 10))
            sb.heatmap(status_per_gen, annot=True, cmap="Blues")
            plt.xlabel('Type 2')
            plt.ylabel('Type 1')
            plt.xticks(rotation=35)
            st.write(fig)


    abil_1_frequency = pokemon_df['ability_1'].value_counts()
    abil_1_mask = abil_1_frequency[abil_1_frequency.values >= 10]
    if plot == 'Ability 1 frequency':
        st.markdown("""- The thought procees of the columns Type 1 and Type 2 is applied to Ability 1:""")
        fig = plt.figure(figsize = (15, 10))
        plt.title("Ability 1 frequencies", fontsize = 18, fontweight = 'bold')
        squarify.plot(sizes = abil_1_mask.values, label = abil_1_mask.index, value = abil_1_mask.values,
                alpha=0.9, color=sb.color_palette("hsv", len(abil_1_mask.index)), pad=1)
        plt.axis('off')
        st.write(fig)
        st.write('The most frequent Abilities are Levitate and Swift Swim (which makes sense since Water is the most frequent Type 1).')

        st.markdown("""- Now onto the frequency of the Abilities 1 associated with Types 1, using a Barchart:""")
        type_2_present_mask = pokemon_df[pokemon_df['type_2'] != 'None']
        type_list = type_2_present_mask['type_1'].value_counts().index
        abil_1_list = pokemon_df['ability_1'].value_counts().index
        fig, axs = plt.subplots(nrows = 8, ncols = 1, figsize = (15, 45), constrained_layout = True)
        for i, ax in zip(range(len(type_list)), axs.flat):
            type_1 = pokemon_df[pokemon_df['ability_1'] == abil_1_list[i]]
            abil_1_frequency = type_1['type_1'].value_counts()
            ax.bar(abil_1_frequency.index, abil_1_frequency.values, label = abil_1_list[i], color = sb.color_palette("Pastel1", len(abil_1_frequency.index)))
            ax.set_xlabel('Type 1', fontsize = 10)
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_ylim([0, 25])
            ax.set_xticks(range(12))
            ax.set_title(abil_1_list[i] + " ability", fontsize = 20, fontweight='bold')
        st.write(fig)
        st.write('For all the subplots the y axis was set to go from 0 to 25, so that they are proportional.')

    if plot == 'Ability 2 frequency':
        st.markdown("""- Firstly, a Pie chart shows which Pokemon have an Ability 2 and which do not (the next plots are applied to the formers).""")
        abil_2_present_mask = pokemon_df[pokemon_df['ability_2'] != 'None']
        type_df_len = [(len(pokemon_df.index) - len(abil_2_present_mask.index)), len(abil_2_present_mask.index)]
        name = ["Pokemon without an Ability 2", "Pokemon with an Ability 2"]
        colors = ['#B7C3F3', '#8EB897']
        fig = plt.figure(figsize = (12, 8))
        plt.title("How many Pokemon have an Ability 2?", fontsize = 18, fontweight = 'bold')
        plt.pie(type_df_len, labels = name, autopct='%.2f%%', startangle = 90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors = colors)
        st.write(fig)
        st.write('It is almost an even split also between Pokemon who have an Ability 2 and those who do not.')

        st.markdown("""- The plots done for Ability 1 are now applied to Ability 2, but focusing on the Abilities 2 held by at least 10 pokemon:""")
        abil_2_present_mask = pokemon_df[pokemon_df['ability_2'] != 'None']
        abil_2_frequency = abil_2_present_mask['ability_2'].value_counts()
        abil_2_mask = abil_2_frequency[abil_2_frequency.values >= 10]
        fig = plt.figure(figsize = (15, 10))
        plt.title("Ability 2 frequencies", fontsize = 18, fontweight = 'bold')
        squarify.plot(sizes = abil_2_mask.values, label = abil_2_mask.index, value = abil_2_mask.values,
                alpha=0.9, color=sb.color_palette("Set1", len(abil_1_mask.index)), pad=1)
        plt.axis('off')
        st.write(fig)
        st.write('The most frequent Abilities 2 are Frisk and  Sturdy.')

        st.markdown("""- Finally a plot for the frequency of Abilities 2 in relation to its Type 1:""")
        abil_list = abil_2_present_mask['ability_2'].value_counts().index
        fig, axs = plt.subplots(nrows = 8, ncols = 1, figsize = (15, 30), constrained_layout = True)
        for i, ax in zip(range(len(type_list)), axs.flat):
            type_1 = abil_2_present_mask[abil_2_present_mask['ability_2'] == abil_list[i]]
            abil_2_frequency = type_1['type_1'].value_counts()
            ax.bar(abil_2_frequency.index, abil_2_frequency.values, label = abil_list[i], color = sb.color_palette("Pastel1", len(abil_2_frequency.index)))
            ax.set_ylim([0, 12])
            ax.set_xticks(range(5))
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_title(abil_list[i] + " Ability 2", fontsize = 20, fontweight='bold')
        st.write(fig)
        st.write('For all the subplots the y axis was set to go from 0 to 12, so that they are proportional.')

    if plot == 'Strongest Pokemon':
        st.markdown("""- An analysis on the strenght of Pokemon based on their Total points.""")
        strongest_pokemon = pokemon_df['total_points'] == pokemon_df['total_points'].max()
        st.write('The strongest Pokemon are: ', pokemon_df[strongest_pokemon])
        weakest_pokemon = pokemon_df['total_points'] == pokemon_df['total_points'].min()
        st.write('The weakest Pokemon is: ', pokemon_df[weakest_pokemon])
        
        st.markdown("""- A histogram to show the distribution of Pokemon by their Total points:""")
        plt.figure(figsize = (15, 15))
        pokemon_df.hist(column = 'total_points', figsize=[12, 8], color = "green", bins=100, grid=False)
        plt.title('Pokemon by Total points', fontsize= 18, fontweight='bold')
        plt.xlabel('Total points')
        plt.ylabel('Number of Pokmeon')
        st.pyplot(fig=plt) 

        st.markdown("""- A histogram to show the distribution of Pokemon by their Total points, now also taking into account their Status:""")
        fig = plt.figure(figsize = (15, 15))
        for i in range(len(status_list)):
            plt.hist(pokemon_df.loc[pokemon_df['status'] == status_list[i], 'total_points'], alpha= 0.5, label= status_list[i], bins = 100)
            plt.title('Pokemon by Total points and by Status', fontsize= 18, fontweight='bold')
            plt.xlabel('Total points')
            plt.ylabel('Number of Pokemon')
            plt.legend(title='Status')
        st.write(fig)

        st.markdown("""- A Bar chart to show the 10 strongest Pokemon in the whole dataframe:""")
        strongest_10_Pokemon = pokemon_df.sort_values('total_points', ascending=False).head(10)
        fig =  plt.figure(figsize = (10, 8))
        plt.title('10 strongest Pokemon', fontsize= 18, fontweight='bold')
        plt.barh(strongest_10_Pokemon.name, strongest_10_Pokemon.total_points, color = sb.color_palette("Paired"))
        plt.xlabel('Total points')
        st.write(fig)
        st.write('As expected, the 10 strongest Pokemon are all Legendary.')

        if st.checkbox("Do you want to see other plots that could be used?"):
            st.markdown("""- A Scatter to show the Pokemon Total Points in correlation to their Pokedex number:""")
            fig = plt.figure(figsize=(10, 8))
            sb.lmplot(x = 'pokedex_number', y = 'total_points', data=pokemon_df, fit_reg=False, hue = 'status', legend = True, height=10, aspect=1)
            plt.title('Strongest Pokemon')
            plt.xlabel('Pokedex number')
            plt.ylabel('Total points')
            st.pyplot(fig=plt) 

            st.markdown("""- The same Scatter as above, but focusing on the "special" Pokemon:""")
            special_mask = pokemon_df[pokemon_df['status'] != 'Normal']
            sb.lmplot(x = 'pokedex_number', y = 'total_points', data = special_mask, fit_reg = False, hue = 'status', legend = True, height=10, aspect=1)
            plt.title('Strongest not Normal Pokemon')
            plt.xlabel('Pokedex number')
            plt.ylabel('Total points')
            st.pyplot(fig=plt)

    against = []
    for i in pokemon_df.columns:
        if 'against' in i:
            against.append(i) 
    against_df = pokemon_df[against]

    stats = []
    for i in pokemon_df.columns:
        if 'against' not in i:
            stats.append(i) 
    stats_df = pokemon_df[stats]
    stats_df.drop(['type_number', 'abilities_number'], axis=1, inplace=True)

    if plot == 'Correlation':
        st.write('Since the dataframe is made up of so many columns, it is plit it into 2: the columns with data about the strength againts others Pokemon and the rest.')
        st.markdown("""#### Against """)
        fig = plt.figure(figsize = (20, 20))
        sb.heatmap(against_df.corr(), annot = True, cmap="Blues").set_title('Correlatiton of against types', fontsize= 18, fontweight='bold')
        st.write(fig)
        st.write('Most of the correlation are as expected, like against_fire and against_grass.')

        st.markdown("""#### Stats """)
        fig = plt.figure(figsize = (20, 20))
        sb.heatmap(stats_df.corr(numeric_only = True), annot = True, cmap="Blues").set_title('Correlatiton of remaining stats', fontsize= 18, fontweight='bold')
        st.write(fig)
        st.write('Since it is a Correlation, only the numeric columns are taken into conisderation.')
        st.write('There are some strong correlation between certain columns.')

    if plot == 'Pokedex number and generation':
        st.write('Thanks to the Heatmap of the correlation, it was shown that there is (an obvious) correlation between Pokedex number and generation, here shown with a Lmplot:')
        sb.lmplot(x = 'pokedex_number', y = 'generation', data= pokemon_df, scatter_kws={'color': 'pink'}, fit_reg=False, legend = True, height=10, aspect=1).fig.set_size_inches(10, 10)
        plt.title('Pokedex number and generation')
        plt.xlabel("Pokedex number")
        plt.ylabel("Generation")
        st.pyplot(fig=plt)
        st.write('The correlation between Pokedex number and Generation is:', pokemon_df['pokedex_number'].corr(pokemon_df['generation']))

    if plot == 'Height and weight':
        st.markdown("""- This Lmplot is also due to the results of the correlation Heatmap.""")
        st.write('As it was already done, the Pokemon are split into groups based on their Status.')
        sb.lmplot(x = 'height_m', y = 'weight_kg', data=pokemon_df, fit_reg=False, hue = 'status', legend = True, height=10, aspect=1).fig.set_size_inches(10, 10)
        plt.title('Height and weight', fontsize= 18, fontweight='bold')
        plt.xlabel("Height")
        plt.ylabel("Weight")
        st.pyplot(fig=plt)

        st.markdown("""- Zooming in:""")
        height_and_weight_mask = pokemon_df[(pokemon_df['height_m'] <= 4) & (pokemon_df['weight_kg'] <= 150)]
        sb.lmplot(x = 'height_m', y = 'weight_kg', data= height_and_weight_mask, fit_reg= False, hue = 'status', legend = True, height= 10, aspect=1).fig.set_size_inches(10, 10)
        plt.title('Height and weight', fontsize= 18, fontweight='bold')
        plt.xlabel("Height")
        plt.ylabel("Weight")
        st.pyplot(fig=plt)
        st.write('The correlation between height and weight is:', pokemon_df['height_m'].corr(pokemon_df['weight_kg']))

    if plot == 'Top 10 correlations':
        st.write('Choose of which split of the dataframe you want to see the top 10 highets correlation:')
        if st.checkbox("Against datafrmae"):
            corr_matrix = against_df.corr()
            corr_mask = corr_matrix[corr_matrix < 1]
            top_5_corr_matrix = (corr_mask.stack().sort_values(ascending=False))[::2].head(5)
            top_5_neg_corr_matrix = (corr_mask.stack().sort_values())[::2].head(5)
            top_10_tot_corr_matrix = pd.concat([top_5_corr_matrix, top_5_neg_corr_matrix])

            st.write('Showing the 10 highest correlation in the Against part of the dataframe with a Lollipop plot:')
            fig = plt.figure(figsize = (10, 10))
            plt.title('"Against" correlation', fontsize= 18, fontweight='bold')
            plt.stem(top_10_tot_corr_matrix, use_line_collection = True)
            plt.ylabel('Correlation')
            x = list(range(len(top_10_tot_corr_matrix)))
            labels = []
            for i in range(len(top_10_tot_corr_matrix)):
                labels.append(top_10_tot_corr_matrix.index[i])
            plt.xticks(x, labels, rotation='vertical')
            st.write(fig)
            st.write('There are both negative and positive correlations (as expected).')

        if st.checkbox("Stats datafrmae"):
            corr_matrix = stats_df.corr(numeric_only = True)
            corr_mask = corr_matrix[corr_matrix < 1]
            top_5_corr_matrix = (corr_mask.stack().sort_values(ascending = False))[::2].head(5)
            top_5_neg_corr_matrix = (corr_mask.stack().sort_values())[::2].head(5)
            top_10_tot_corr_matrix = pd.concat([top_5_corr_matrix, top_5_neg_corr_matrix])

            st.write('Showing the 10 highest correlation in the Stats part of the dataframe with a Lollipop plot:')
            fig = plt.figure(figsize = (10, 10))
            plt.title('"Stats" correlation', fontsize= 18, fontweight='bold')
            plt.stem(top_10_tot_corr_matrix, use_line_collection = True)
            plt.ylabel('Correlation')
            x = list(range(len(top_10_tot_corr_matrix)))
            labels = []
            for i in range(len(top_10_tot_corr_matrix)):
                labels.append(top_10_tot_corr_matrix.index[i])
            plt.xticks(x, labels, rotation='vertical')
            st.write(fig)
            st.write('There are both negative and positive correlations (as expected).')

if st.sidebar.checkbox('Models'):
    st.title("MODELS")
    st.subheader('Prediction on which Type 1')
    st.write('This model tries to preditc the Type 1 of a Pokemon using a Random Forest.')
    pokemon_type = pokemon_df[['type_1','against_normal', 'against_fire', 'against_water', 'against_electric',
        'against_grass', 'against_ice', 'against_fight', 'against_poison',
        'against_ground', 'against_flying', 'against_psychic', 'against_bug',
        'against_rock', 'against_ghost', 'against_dragon', 'against_dark',
        'against_steel', 'against_fairy']].copy()
    pokemon_type['type_1'] = pd.Categorical(pokemon_type.type_1)
    
    x = pokemon_type.filter(regex='against')
    y = pokemon_type.type_1
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.5 , random_state = 42)
    model = RandomForestClassifier(random_state= 42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    st.write('Accuracy of the model: ', accuracy_score(y_test, y_pred))
    st.write('Classification report of the model: ')
    st.write(classification_report(y_test, y_pred, output_dict = True))

    st.subheader('Prediction on Status')    
    st.write('This model tries to preditc the Status of a Pokemon using a Random Forest.')
    pokemon_legendary = pokemon_df[['status', 'total_points']].copy()
    replace_dict = {
    'Normal': 0,
    'Sub Legendary': 1,
    'Legendary' : 1,
    'Mythical': 1}
    pokemon_legendary.status.replace(replace_dict, inplace = True)
    st.write('The correlation between Status and Total points is', pokemon_legendary['status'].corr(pokemon_legendary['total_points']))
    
    x = pokemon_df.total_points
    y = pokemon_legendary.status
    test_size_status = st.slider('Choose the test size: ', min_value = 0.1, max_value = 0.9, step = 0.1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= test_size_status, random_state = 22)
    model = RandomForestClassifier(random_state= 42)
    model.fit(x_train.to_numpy().reshape(-1, 1), y_train)
    y_pred = model.predict(x_test.to_numpy().reshape(-1, 1))
    st.write('Accuracy of the model: ', accuracy_score(y_pred, y_test))
    st.write("Precision of the model: ", precision_score(y_test, y_pred))
    st.write("Recall of the model:",  recall_score(y_test, y_pred))
    st.write('It is possible there is some Overfitting')