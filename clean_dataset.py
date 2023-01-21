# All these updates of the datasets are studied with more details on the Google Colab file

import pandas as pd
pd.options.mode.chained_assignment = None

def return_clean_dataset():
  pokemon_df = pd.read_csv('pokemon.csv')

  pokemon_df.drop(['Unnamed: 0','german_name', 'japanese_name', 'growth_rate'], axis=1, inplace=True)
  pokemon_df.drop(['egg_type_number', 'egg_type_1', 'egg_type_2', 'percentage_male', 'egg_cycles'], axis=1, inplace=True)

  pokemon_df = pokemon_df[pokemon_df['ability_1'].notna()]
  pokemon_df[['type_2', 'ability_2', 'ability_hidden']] = pokemon_df[['type_2', 'ability_2', 'ability_hidden']].fillna("None")

  legendary_catch_rate_mean = round(pokemon_df[pokemon_df['status'] == 'Legendary'].catch_rate.mean(), 1)
  legendary_and_null_catch_mask =  (pokemon_df['status'] == 'Legendary') & (pokemon_df['catch_rate'] != pokemon_df['catch_rate'])
  pokemon_df.loc[legendary_and_null_catch_mask, 'catch_rate'] = pokemon_df.loc[legendary_and_null_catch_mask, 'catch_rate'].fillna(legendary_catch_rate_mean)
  pokemon_df = pokemon_df[pokemon_df['catch_rate'].notna()]

  null_status_list = ['Normal', 'Mythical', 'Sub Legendary', 'Legendary']
  null_column_name = ['base_friendship', 'base_experience']
  for j in null_column_name:
    for i in null_status_list:
      status_mask =  pokemon_df[pokemon_df['status'] == i]
      status_base_mean = round(status_mask[j].mean(), 1)
      status_and_null_mask =  (pokemon_df['status'] == i) & (pokemon_df[j] != pokemon_df[j])
      pokemon_df.loc[status_and_null_mask, j] = pokemon_df.loc[status_and_null_mask, j].fillna(status_base_mean)

  pokemon_df.reset_index(drop=True, inplace=True)
  
  return pokemon_df

if __name__ == '__mai__':
    print(return_clean_dataset)