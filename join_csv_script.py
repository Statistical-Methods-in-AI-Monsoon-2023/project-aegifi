# %%
import pandas as pd

# %%
# get the plot and genres csv files into dataframes
plot_df = pd.read_csv('./data/plots.csv')
genres_df = pd.read_csv('./data/genres.csv')

# %%
# perform join based on movie_id
merged_df = pd.merge(plot_df, genres_df, on='movie', how='inner')

# %%
merged_df

# %%
# save the merged dataframe to a csv file
merged_df.to_csv('./data/plots_and_summaries.csv', index=False)

# %%



