import pandas as pd

df = pd.read_csv('./data/plots_and_genres.csv')


genres_to_consider = [
    'Drama',
    'Comedy',
    'Thriller',
    'Romance',
    'Action',
    'Family',
    'Horror',
    'Crime',
    'Adventure',
    'Animation',
    'Fantasy',
    'Sci-Fi',
    'Mystery',
    'Biography',
    'Music',
    'History',
    'War',
    'Western',
    'Sport',
    'Musical',
]

# filter the dataframe to only include the genres we want to consider
def check(row):
    sublist = row['genres']
    # remove quotes from each item in the list
    sublist = [item.strip().strip("'") for item in sublist.strip('][').split(',')]

    # # if even one of the genres in the list is in the genres_to_consider list, return True
    # for genre in sublist:
    #     if genre in genres_to_consider:
    #         return True
    
    # check intersection of two lists
    if len(set(sublist).intersection(set(genres_to_consider))) > 0:
        # set genres to the intersection of the two lists
        row['genres'] = list(set(sublist).intersection(set(genres_to_consider)))
        return True
    
    return False

filtered_df = df[df.apply(check, axis=1)]

# count the number of movies in each genre
genre_counts = {}
for row in filtered_df['genres']:
    for genre in row:
        if genre in genre_counts:
            genre_counts[genre] += 1
        else:
            genre_counts[genre] = 1
genre_counts

# filter out plots that are more than 200 words long
filtered_df = filtered_df[filtered_df['plot'].apply(lambda x: len(x.split(' ')) < 200)]
filtered_df

filtered_df.to_csv('./data/filtered_plots_and_genres.csv', index=False)