import pandas as pd
import numpy as np
import clean_dataset as cl # I import the Python file that cleans the dataframe
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sb
import squarify
from pywaffle import Waffle

# Create a dynamic page
st.header("Pokemon project")
st.subheader("To be like Professor Oak!")

pokemon_df = pd.read_csv('pokemon.csv')

# I use a sidebar to keep the Eda part of the project separate from the plots and models
# I split the EDA between before and after the cleaning
st.sidebar.write("Additional info")
if st.sidebar.checkbox("EDA before cleaning"):
    st.title("BEFORE CLEANING")
    st.write("Pokemon dataframe:")
    st.write(pokemon_df)
    st.write("Rows and columns:", pokemon_df.shape)
    
    st.write("Pokemon head and tail:")
    st.write(pokemon_df.head())
    st.write(pokemon_df.tail())

    st.write("Some numerical informations:")
    st.write(pokemon_df.describe())

# Now I work with the clean dataset -> no null values
pokemon_df = cl.return_clean_dataset()
if st.sidebar.checkbox("EDA after cleaning"):
    st.title("AFTER CLEANING")
    st.write("Pokemon dataframe:")
    st.write(pokemon_df)
    st.write("Rows and columns:", pokemon_df.shape)

    st.write("Pokemon head and tail:")
    st.write(pokemon_df.head())
    st.write(pokemon_df.tail())

    st.write("Some numerical informations:")
    st.write(pokemon_df.describe())

st.subheader("Plots")
# I use a selectbox to create a drop-down menu to be able to chose which plot to see
plot = st.selectbox('Which plot would you like to be see?',
    ('How many Pokemon per generation', 'Pokemon statuses per generation', 'Type 1 frequency', 'Type 2 frequency'))
#if st.checkbox('How many Pokemon per generation?'):
if plot == 'How many Pokemon per generation':
    st.write('As the title says, this is a graph to show how many Pokemon each generation has')
    # [8, 2] is to chose the dimension of the 2 columns
    col_1, col_2 = st.columns([8, 2])
    pokemon_per_generation = pokemon_df['generation'].value_counts()
    with col_1:
        fig = plt.figure(figsize = (10, 8))
        plt.title('How many Pokemon per generation?')
        plt.barh(pokemon_per_generation.index, pokemon_per_generation.values, color=['firebrick', 'turquoise', 'gold', 'limegreen', 'darkorange', 'palevioletred', 'darkviolet', 'cornflowerblue'])
        plt.xlabel('Number of Pokemon')
        plt.ylabel('Generation')
        st.write(fig)
    with col_2:
        st.write(pokemon_per_generation)

if plot == 'Pokemon statuses per generation':
    st.write('I study how in each generation the statuses of Pokemon are distributed')
    status_list = pokemon_df['status'].value_counts().index
    fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 20), constrained_layout = True)
    fig.suptitle('Pokemon statuses per generation', fontsize= 25, fontweight='bold')
    for i, ax in zip(range(len(status_list)), axs.flat):
        status_mask = pokemon_df[pokemon_df['status'] == status_list[i]]
        status_per_gen = status_mask['generation'].value_counts()
        donut_circle = plt.Circle( (0,0), 0.45, color = 'white')
        ax.pie(status_per_gen.values, labels = status_per_gen.index, autopct='%.2f%%', colors = sb.color_palette("Paired", len(status_per_gen.index)), labeldistance=None)
        ax.add_artist(donut_circle)
        handles, labels = ax.get_legend_handles_labels()
        handles, labels = zip(*[ (handles[i], labels[i]) for i in sorted(range(len(handles)), key = lambda k: list(map(int, labels))[k])] )
        ax.legend(handles, labels, title = 'Generation:', bbox_to_anchor=(1, 1))
        ax.set_title(status_list[i], fontsize = 20, fontweight='bold')
    st.write(fig)

    if st.checkbox("Do you want to see other plots that could be used?"):
        st.write("I chose to use a Donut chart because I think it shows clearly the wanted info, but I could have used a Barchart:")
        fig, axs = plt.subplots(nrows = 2, ncols = 2, figsize = (20, 20), constrained_layout = True)
        for i, ax in zip(range(len(status_list)), axs.flat):
            status_mask = pokemon_df[pokemon_df['status'] == status_list[i]]
            status_per_gen = status_mask['generation'].value_counts()
            ax.bar(status_per_gen.index, status_per_gen.values, label = status_list[i], color = sb.color_palette("Paired", len(status_per_gen.index)))
            ax.tick_params(axis = 'both', labelsize = 15)
            ax.set_title(status_list[i], fontsize = 20, fontweight='bold')
        st.write(fig)
        st.write("Or I could have used a Heatmap:")
        special_mask = pokemon_df[pokemon_df['status'] != 'Normal']
        status_per_gen = special_mask.groupby(['generation', 'status']).size().unstack(fill_value = 0)
        fig= plt.figure(figsize = (10, 8))
        sb.heatmap(status_per_gen, annot=True, cmap="Blues", fmt="d")
        st.write(fig)
        st.write('I removed the Normal status since it spiked my results')

if plot == 'Type 1 frequency':
    type_1_frequency = pokemon_df['type_1'].value_counts()
    fig = plt.figure(figsize = (15, 10))
    plt.title("Type 1 frequencies", fontdict = {'fontsize' : 20})
    squarify.plot(sizes = type_1_frequency.values, label = type_1_frequency.index, value = type_1_frequency.values,
              alpha=0.9, color=sb.color_palette("Paired"), pad=1)
    plt.axis('off')
    st.write(fig)

    if st.checkbox("Do you want to see other plots that could be used?"):
        st.write("Since I have many values of Type 1, a Pie chart would have been quite confusing (but still works):")
        fig = plt.figure(figsize=(10, 8))
        plt.title("Type 1 frequencies", fontdict = {'fontsize' : 20})
        plt.pie(type_1_frequency, labels=type_1_frequency.index, autopct='%.2f%%', startangle=90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' },  labeldistance=None)
        plt.legend(bbox_to_anchor=(1.5, 1))
        plt.tight_layout()
        st.write(fig)

if plot == 'Type 2 frequency':
    st.write('I study the frequency of Type 2 in relation to which Type 1 the pokemon has.')
    type_2_present_mask = pokemon_df[pokemon_df['type_2'] != 'None']

    st.write("I use a pie chart to show which Pokemon have a Type 2 and which don't (I'm gonna work on the former in the ext plot).")
    type_df_len = [(len(pokemon_df.index) - len(type_2_present_mask.index)), len(type_2_present_mask.index)]
    name = ["Pokemon without a Type 2", "Pokemon with a Type 2"]
    colors = ['#B7C3F3', '#8EB897']
    fig = plt.figure(figsize=(10, 8))
    plt.title("How many Pokemon have a Type 2?", fontdict = {'fontsize' : 20})
    plt.pie(type_df_len, labels = name, autopct='%.2f%%', startangle=90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' }, colors = colors)
    st.write(fig)

    st.write("Now I study the frequency of the Types 2 as I did for Type 1:")
    type_2_frequency = type_2_present_mask['type_2'].value_counts()
    fig = plt.figure(figsize = (15, 10))
    plt.title("Type 2 frequencies", fontdict = {'fontsize' : 20})
    squarify.plot(sizes = type_2_frequency.values, label = type_2_frequency.index, value = type_2_frequency.values,
              alpha = 0.9, color=sb.color_palette("Paired"), pad=1)
    plt.axis('off')
    st.write(fig)
    
    # DA SISTEAMRE
    st.write("Now I study the frequency of the Types 2 as I did for Type 1:")
    fig, axs = plt.subplots(nrows = 9, ncols = 2, figsize = (45, 45), constrained_layout = True)
    type_list = type_2_present_mask['type_1'].value_counts().index
    for i, ax in zip(range(len(type_list)), axs.flat):
        type_1 = type_2_present_mask[type_2_present_mask['type_1'] == type_list[i]]
        type_2_frequency = type_1['type_2'].value_counts()
        ax.bar(type_2_frequency.index, type_2_frequency.values, label = type_list[i], color = sb.color_palette("Paired", len(type_2_frequency.index)))
        ax.set_ylim([0, 25])
        ax.set_xticks(range(15)) 
        ax.tick_params(axis = 'both', labelsize = 15)
        ax.set_title(type_list[i] + " type 1", fontsize = 20, fontweight='bold')
    st.write(fig)

    if st.checkbox("Do you want to see other plots that could be used?"):
        st.write("For these alternative plots, I only studied the Water Type 1 to make it simpler.")
        st.write("With a Pie chart:")
        water_type_1 = type_2_present_mask[type_2_present_mask['type_1'] == 'Water']
        type_1_water_frequency = water_type_1['type_2'].value_counts()
        fig = plt.figure(figsize=(10, 8))
        plt.title("Type 2 frequencies for Water", fontdict = {'fontsize' : 20})
        plt.pie(type_1_water_frequency.values, labels = type_1_water_frequency.index, autopct='%.2f%%', startangle=90, wedgeprops = { 'linewidth' : 3, 'edgecolor' : 'white' })
        plt.legend(bbox_to_anchor=(1.5, 1))
        st.write(fig)

        st.write('Or a Heatmap:')
        special_mask = pokemon_df[pokemon_df['type_2'] != 'None']
        status_per_gen = special_mask.groupby(['type_1', 'type_2']).size().unstack()
        fig = plt.figure(figsize = (10, 10))
        sb.heatmap(status_per_gen, annot=True, cmap="Blues")
        plt.xlabel('Type 2')
        plt.ylabel('Type 1')
        plt.xticks(rotation=35)
        st.write(fig)
    
