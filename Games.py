# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Let us extract both datasets and their info.</span>

# %%
import pandas as pd
import re

base_df = pd.read_csv("Steamspy_50k.csv")
hltb_df = pd.read_csv("hltb_FINAL_results.csv")

# %%
base_df.head(5)

# %%
hltb_df.head(5)

# %%
print("--- Base (Steamspy) DataFrame Info ---")
base_df.info()


# %%
print("\n--- HLTB DataFrame Info ---")
hltb_df.info()

# %%
print("--- Base (Steamspy) Null Value Counts ---")
print(base_df.isnull().sum())

# %%
print("\n--- HLTB Null Value Counts ---")
print(hltb_df.isnull().sum())

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Let us now do some prelimenary cleaning. Let us drop the entries with NULL names so we can join the two datasets.</span>

# %%
base_df.dropna(subset=['name'], inplace=True)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Lets also standardize their names.</span>

# %%
def clean_name_safe(name):
    name_str = str(name).lower()
    name_str = name_str.replace('&', 'and')
    name_str = name_str.replace('™', '')
    name_str = name_str.replace('®', '')
    name_str = name_str.replace(' ix', ' 9')
    name_str = name_str.replace(' iv', ' 4')
    name_str = name_str.replace(' v', ' 5')
    name_str = name_str.replace(' vi', ' 6')
    name_str = name_str.replace(' vii', ' 7')
    name_str = name_str.replace(' viii', ' 8')
    name_str = name_str.replace(' x', ' 10')
    name_str = name_str.replace(' i', ' 1')
    name_str = re.sub(r'[^a-z0-9\s]', '', name_str)
    name_str = re.sub(r'\s+', ' ', name_str)
    return name_str.strip()

print("Applying 'clean_name_safe' to SteamSpy data...")

base_df['join_key'] = base_df['name'].apply(clean_name_safe)

# %%
print("\n--- Base (Steamspy) Cleaned Names ---")
base_df[['name', 'join_key']]

# %%
print("\n--- HLTB Cleaned Names ---")
hltb_df[['matched_name', 'search_key']]

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Remove noise</span>

# %%
noise_keywords = ['Soundtrack', 'DLC', 'Artbook', 'Demo', 'Beta', 'Prologue', 'Season Pass']
is_noise = base_df['name'].str.contains('|'.join(noise_keywords), case=False, na=False)

print(f"Original SteamSpy rows: {len(base_df)}")
base_df = base_df[~is_noise] 
print(f"Filtered SteamSpy rows (removed noise): {len(base_df)}")

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Now let us merge</span>

# %%
print("Merging dataframes...")
merged_df = pd.merge(
    base_df, 
    hltb_df, 
    left_on='join_key',    
    right_on='search_key', 
    how='left'
)

print(f"Merge complete. Total rows: {len(merged_df)}")

# %%
merged_df.head(5)

# %%
cols_to_drop = [
    'join_key', 
    'search_key', 
    'matched_name', 
    'score_rank',
    'positive',
    'negative', 
    'userscore',
    'appid'
]
merged_df.drop(columns=cols_to_drop, axis=1, inplace=True, errors='ignore')

# %%
merged_df.head(5)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Now let us drop the rows that did not merge properly for whatever reason</span>

# %%
hltb_specific_columns = ['Main Story (h)', 'Main + Sides (h)', 'Completionist (h)', 'Co-Op (h)', 'Vs (h)']
unmatched_mask = merged_df[hltb_specific_columns].isnull().all(axis=1)
unmatched_count = unmatched_mask.sum()

print(f"Total rows in merged_df: {len(merged_df)}")
print(f"Number of rows with no HLTB match (to be removed): {unmatched_count}")

# %%
hltb_specific_columns = [
    'Main Story (h)', 'Main + Sides (h)', 'Completionist (h)', 
    'Co-Op (h)', 'Vs (h)'
]

unmatched_mask = merged_df[hltb_specific_columns].isnull().all(axis=1)
unmatched_count = unmatched_mask.sum()
total_rows = len(merged_df)
matched_count = total_rows - unmatched_count

print(f"\n--- Merge Analysis Complete ---")
print(f"Total rows: {total_rows}")
print(f"Successfully Matched rows: {matched_count}")
print(f"Unmatched rows (no HLTB data): {unmatched_count}")

# %%
cleaned_df = merged_df.dropna(subset=hltb_specific_columns, how='all')
print(f"Original merged rows: {len(merged_df)}")
print(f"New 'cleaned' rows: {len(cleaned_df)}")

# %%
merged_df = cleaned_df
df = merged_df
df.head(5)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;"> Now let us clean up the vs column and fill it so that we can use it to split.</span>

# %%
df['Vs (h)'] = df['Vs (h)'].fillna(0)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">And remove the remaining useless columns </span>

# %%
df.drop(['price','initialprice','discount', 'average_2weeks', 'median_2weeks','ccu'],axis=1,inplace=True,errors='ignore')
df.head(5)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Let us also convert the avergae playtime to hours to standardize everything</span>

# %%
cols_to_convert = ['average_forever', 'median_forever']

def to_hours(x):
    return float(x/60)

df[cols_to_convert] = df[cols_to_convert].applymap(to_hours)
df.head(5)

# %%
df.shape

# %%
df.head(4)

# %%
df.sort_values(by='average_forever',ascending=False).head(10)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Let us also remove shovelware/games with owners too low to not mess up our analysis</span>

# %%
OWNER_THRESHOLD = 35000
AVERAGE_PLAYTIME_THRESHOLD = 2
MEDIAN_PLAYTIME_THRESHOLD = 1

print(f"Rows before filtering: {len(df)}")
df = df[df['owners_mid'] > OWNER_THRESHOLD]
df = df[df['median_forever'] >= MEDIAN_PLAYTIME_THRESHOLD]
df = df[df['average_forever'] >= AVERAGE_PLAYTIME_THRESHOLD]
print(f"Rows after filtering:  {len(df)}")

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Among Us </span>

# %%
df[df['name'] == 'Among Us']

# %% [markdown]
#  

# %% [markdown]
# -------------------------------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# 

# %% [markdown]
# <span style="color: pink; font-weight: bold;">
# 
# Hypothesis testing: top 75% of median playtime players will play full game + sides
# 
# Hypothesis testing: co-op games have higher average campaign beaten than non-co-op
# 
# </span>
# 
# t_statistic, p_value = stats.ttest_ind(
#     coop_games, 
#     no_coop_games,
#     alternative='greater',  # One-tailed: co-op > no co-op
#     equal_var=False  # Welch's t-test (doesn't assume equal variance)
# )

# %% [markdown]
# <span style="color: red; font-weight: bold;">Analysis Section </span>

# %%
df.head(2)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">Let us do a simple, blunt analysis. We will use the average lifetime hours divided by the mid of the owners to get the top played games </span>

# %%
df.sort_values(by='average_forever', ascending=False).head(10)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">And one with the MOST played games</span>

# %%
df['total_hours_played'] = df['average_forever']*df['owners_mid']
df.sort_values(by='total_hours_played',ascending=False)[['name','developer','publisher','total_hours_played','average_forever','median_forever']].head(10)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;">We notice that in the first table, the mean and median were TOO similar. Not good. let us see what the norm is:</span>

# %%
valid_rows = df[
    (df['median_forever'] > 0) &  
    (df['average_forever'] != df['median_forever'])  
].copy()

valid_rows['playtime_ratio'] = valid_rows['average_forever'] / valid_rows['median_forever']

average_playtime_ratio = valid_rows['playtime_ratio'].mean()

print(f"Number of valid rows for ratio calculation: {len(valid_rows)}")
print(f"Average Mean/Median Playtime Ratio: {average_playtime_ratio}")

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;"> Let us then define a metric for elimenating these entries </span>

# %%
print(f"Rows before reduction: {len(df)}")
ratio_is_sane = ((df['average_forever'] / df['median_forever']) > 1) & \
                  ((df['average_forever'] / df['median_forever']) < 15)
has_known_data = (df['Main Story (h)'] > 0.5) | (df['Vs (h)'] > 0.5)
df = df[ratio_is_sane & has_known_data].copy()
print(f"Rows after reduction: {len(df)}")

# %%
df.sort_values(by='average_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid']].head(10)

# %%
df.sort_values(by='median_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid']].head(10)

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;"> Much better! Now let us split these games by single and multiplayer. Let us define metrics for each and divide our dataframes into two seperate categories, while dropping rows who's time to beat is unknown.</span>

# %%
has_known_data = (df['Main Story (h)'] > 0) | (df['Vs (h)'] > 0)

df_filtered = df[has_known_data].copy()

is_single_player = df_filtered['Main Story (h)'] >= (2 * df_filtered['Vs (h)'])

single_player_df = df_filtered[is_single_player].copy()
multiplayer_df = df_filtered[~is_single_player].copy() 

print(f"Total games in original df: {len(df)}")
print(f"Games in single_player_df: {len(single_player_df)}")
print(f"Games in multiplayer_df: {len(multiplayer_df)}")


print("\nExporting datasets to CSV files...")

df_filtered.to_csv("all_filtered_games.csv", index=False)

single_player_df.to_csv("single_player_games.csv", index=False)

multiplayer_df.to_csv("multiplayer_games.csv", index=False)

print("Export complete: all_filtered_games.csv, single_player_games.csv, multiplayer_games.csv")

# %%


# %% [markdown]
# <span style="color: #445E80; font-weight: bold;"> Now let us repeat our averages and mids. </span>

# %%
single_player_df.sort_values(by='average_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid']].head(10)

# %%
single_player_df.sort_values(by='median_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid','Co-Op (h)']].head(10)

# %%
multiplayer_df.sort_values(by='average_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid','']].head(10)

# %%
multiplayer_df.sort_values(by='median_forever', ascending=False).head(10)

# %% [markdown]
# 

# %% [markdown]
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# %% [markdown]
# 

# %% [markdown]
# <span style="color: red"> Part 3: Single Player Analysis </span>

# %% [markdown]
# <span style="color: #445E80; font-weight: bold;"> Let us do one final analysis for the single player games. Find the games by their proportion of playtime played. We will exlude games shorter than 3 hours to make this analysis more robust. </span>

# %%
single_player_df.head(2)
single_player_df = single_player_df[single_player_df['Main Story (h)'] > 4]

# %%


# %%
single_player_df['average_campaign_beaten'] = single_player_df['average_forever']/single_player_df['Main Story (h)']
single_player_df['median_campaign_beaten'] = single_player_df['median_forever']/single_player_df['Main Story (h)']
single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)[['name','developer','publisher','average_campaign_beaten','Main Story (h)','average_forever','average_campaign_beaten','median_campaign_beaten']]

# %% [markdown]
# Also, let's do a final search for the games who's main campaign is not too shorter than the whole package.

# %%
campaign_single_player_df = single_player_df[single_player_df['Main Story (h)'] <= 2*single_player_df['Main + Sides (h)']].copy()
is_counter_strike = campaign_single_player_df['name'].str.contains('Counter-Strike', case=False, na=False)
campaign_single_player_df = campaign_single_player_df[~is_counter_strike]
campaign_single_player_df.sort_values(by='average_campaign_beaten',ascending=False)[['name','developer','publisher','average_campaign_beaten','Main Story (h)','average_forever','median_forever']].head(10)

# %%


# %%
short_campaign_single_player_df = single_player_df[single_player_df['average_campaign_beaten'] <= 6]
short_campaign_single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)[['name','developer','publisher','Main Story (h)','average_campaign_beaten']]

# %%
short_campaign_single_player_df = single_player_df[single_player_df['average_campaign_beaten'] <= 1]
short_campaign_single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)[['name','developer','publisher','Main Story (h)','average_campaign_beaten']]


