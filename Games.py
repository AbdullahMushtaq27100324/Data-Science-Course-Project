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

# %%
df.sort_values(by='average_forever',ascending=False)

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
single_player_df.sort_values(by='median_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid','Co-Op (h)','Vs (h)']].head(10)

# %%
multiplayer_df.sort_values(by='average_forever', ascending=False)[['name','developer','publisher','median_forever','average_forever','owners_mid']].head(10)

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
short_campaign_single_player_df = single_player_df[single_player_df['average_campaign_beaten'] <= 6]
short_campaign_single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)[['name','developer','publisher','Main Story (h)','average_campaign_beaten']]

# %%
short_campaign_single_player_df = single_player_df[single_player_df['average_campaign_beaten'] <= 1]
short_campaign_single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)[['name','developer','publisher','Main Story (h)','average_campaign_beaten']]

# %% [markdown]
# <span style='color: red'>For a final output:</span>

# %%
multiplayer_df.sort_values(by='average_forever', ascending=False).head(10)

# %%
multiplayer_df.sort_values(by='median_forever', ascending=False).head(10)

# %%
single_player_df.sort_values(by='average_campaign_beaten',ascending=False).head(10)

# %%
single_player_df.sort_values(by='median_campaign_beaten', ascending=False).head(10)

# %% [markdown]
# <span style="color: red"> Part 4: Hypothesis Testing </span>
# <br><br>
# Now that we have some good data, let us explore some interesting hypothesis questions. While we do not have enough entries to create proper predictions (e.g via a model), we nonetheless can try and get some semblance of an idea of the characteristics that more engaging games tend to posess.

# %% [markdown]
# <span style="color:pink; font-weight: bold;">
# Question 1: Do multiplayer games really have more engagement than single player games?
# </span> 
# <br><br>
# We can see this using the Mann-Whitney U Test:  

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np

# --- 1. Statistical Testing ---

# We check 'average_forever' first
sp_avg = single_player_df['average_forever']
mp_avg = multiplayer_df['average_forever']

# Mann-Whitney U Test (Non-parametric, good for skewed data)
# Alternative='less' implies we are testing if SP is LESS than MP (i.e., MP is greater)
stat, p_value = stats.mannwhitneyu(sp_avg, mp_avg, alternative='less')

print(f"--- Statistical Test Results (Average Playtime) ---")
print(f"Single Player Mean: {sp_avg.mean():.2f} hours")
print(f"Multiplayer Mean:   {mp_avg.mean():.2f} hours")
print(f"Mann-Whitney U statistic: {stat}")
print(f"P-value: {p_value:.5f}")

if p_value < 0.05:
    print("Result: Significant! Multiplayer games have statistically higher engagement.")
else:
    print("Result: Not significant. No statistical difference found.")

# --- 2. Visualization ---

# Combine data for plotting
# We add a 'Type' column to distinguish them in the plot
single_player_df['Game Type'] = 'Single Player'
multiplayer_df['Game Type'] = 'Multiplayer'

combined_plot_df = pd.concat([single_player_df, multiplayer_df])

# Create the plots
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Plot A: Box Plot of Average Playtime
sns.boxplot(x='Game Type', y='average_forever', data=combined_plot_df, ax=ax[0], palette="Set2")
ax[0].set_title('Distribution of Average Playtime')
ax[0].set_ylabel('Average Hours Played')
ax[0].set_yscale('log') # Log scale is CRITICAL here to see the data clearly
ax[0].grid(True, which="both", ls="-", alpha=0.2)

# Plot B: Box Plot of Median Playtime (Often a more "honest" metric)
sns.boxplot(x='Game Type', y='median_forever', data=combined_plot_df, ax=ax[1], palette="Set2")
ax[1].set_title('Distribution of Median Playtime')
ax[1].set_ylabel('Median Hours Played')
ax[1].set_yscale('log')
ax[1].grid(True, which="both", ls="-", alpha=0.2)

plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color: pink; font-weight: bold;">
# Qustion 2:Social Accountability: Do games with co-op mechanics result in higher campaign completion rates than pure single-player games?
# </span>
# <br><br>
# We can test this by first splitting games into co-op (greater than 0 co-op hours) and non-co-op (0 co-op hours) and testing:

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd

# --- 1. Data Prep ---
# (Assuming 'single_player_df' is already loaded from your previous steps)

# Filter for games with meaningful campaigns (> 4 hours)
df_analysis = single_player_df[single_player_df['Main Story (h)'] > 4].copy()

# Calculate the Ratio
df_analysis['completion_ratio'] = df_analysis['average_forever'] / df_analysis['Main Story (h)']

# Define Groups
df_analysis['has_coop'] = df_analysis['Co-Op (h)'].fillna(0) > 0
group_pure_sp = df_analysis[df_analysis['has_coop'] == False]['completion_ratio']
group_coop = df_analysis[df_analysis['has_coop'] == True]['completion_ratio']

# Create a Label column for the plot
df_analysis['Category'] = df_analysis['has_coop'].map({True: 'Has Co-Op', False: 'Pure Single Player'})

# --- 2. The Calculation (T-Test) ---
# We use "Welch's T-Test" (equal_var=False) because one group might be more variable than the other.
# 'alternative=greater' checks if Co-Op is GREATER than Pure SP.
t_stat, p_value = stats.ttest_ind(group_coop, group_pure_sp, equal_var=False, alternative='greater')

print(f"--- T-Test Results ---")
print(f"Pure Single Player Mean Ratio: {group_pure_sp.mean():.2f}")
print(f"Co-Op Enabled Mean Ratio:      {group_coop.mean():.2f}")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-value: {p_value:.5f}")

if p_value < 0.05:
    print("Result: Significant! Co-Op games have a higher mean completion ratio.")
else:
    print("Result: Not significant.")

# --- 3. The Visualization (Histogram) ---
plt.figure(figsize=(12, 6))

# We use stat='density' to normalize the bars. 
# Without this, the 'Pure SP' bars would be huge and 'Co-Op' tiny just because there are more SP games.
sns.histplot(
    data=df_analysis, 
    x='completion_ratio', 
    hue='Category', 
    element='step',     # 'step' makes it look like an outline/transparent overlap
    stat='density',     # CRITICAL: Normalizes so we compare shapes, not counts
    common_norm=False,  # Calculate density for each group separately
    palette='Set1',
    alpha=0.3
)

plt.title('Distribution of Campaign Completion Ratios', fontsize=14)
plt.xlabel('Completion Ratio (Avg Playtime / Campaign Length)', fontsize=12)
plt.xlim(0, 5) # Limit view to 0-5x ratio to hide extreme outliers
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# But this is not enough! We want to also test if increasing the number of hours in the co-op column; i.e, do games with a larger co-op component have greater levels of player engagment?

# %%
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

# --- 1. Filter for Co-Op Games ---
# We only want to analyze games that actually HAVE a co-op mode.
# We use the 'single_player_df' because we are looking at "Single Player games with Co-Op elements"
# (Make sure you have run the cell where you defined 'average_campaign_beaten' first!)

coop_analysis_df = single_player_df[
    (single_player_df['Co-Op (h)'] > 0) & 
    (single_player_df['average_campaign_beaten'] < 10) # Filter extreme outliers (>10x playtime) for cleaner plots
].copy()

# --- 2. The Statistical Test (Spearman Correlation) ---
# We use Spearman because playtimes are rarely linear/normal
corr_coef, p_value = stats.spearmanr(coop_analysis_df['Co-Op (h)'], coop_analysis_df['average_campaign_beaten'])

print(f"--- Correlation Analysis: Length of Co-Op vs. Engagement ---")
print(f"Spearman Correlation Coefficient: {corr_coef:.3f}")
print(f"P-value: {p_value:.5f}")

if p_value < 0.05:
    if corr_coef > 0:
        print("Result: Significant POSITIVE correlation. More Co-Op content links to higher engagement ratios.")
    else:
        print("Result: Significant NEGATIVE correlation.")
else:
    print("Result: No significant correlation found.")

# --- 3. Visualization (Regression Plot) ---
plt.figure(figsize=(10, 6))

# Regplot draws the scatter points AND a trend line (regression line)
sns.regplot(
    x='Co-Op (h)', 
    y='average_campaign_beaten', 
    data=coop_analysis_df, 
    scatter_kws={'alpha':0.5}, # Make dots transparent to see density
    line_kws={'color':'red'}   # Trend line color
)

plt.title(f'Does More Co-Op Content = Higher Engagement?\n(Spearman Corr: {corr_coef:.2f})', fontsize=14)
plt.xlabel('Hours of Co-Op Content (HLTB)', fontsize=12)
plt.ylabel('Engagement Ratio (Avg Playtime / Main Story)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# Therfore, we can funnily conclude that while having co-op features in a game does indeed increase player engagment, the *amount* of co-op features or their length may not have a special effect. Meaning players value the features themselves, rather than how 'meaty' they are.

# %% [markdown]
# <span style="color: pink; font-weight: bold;">
# The Distraction Effect: Does a high ratio of side content to main story length negatively correlate with main story completion rates?
# </span>

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np

# --- 1. Data Prep (Assumes single_player_df is loaded) ---
# Filter for meaningful campaigns
df_distraction = single_player_df[single_player_df['Main Story (h)'] > 4].copy()

# Calculate Variables
df_distraction['bloat_ratio'] = (df_distraction['Main + Sides (h)'] - df_distraction['Main Story (h)']) / df_distraction['Main Story (h)']
df_distraction['completion_ratio'] = df_distraction['average_forever'] / df_distraction['Main Story (h)']

# Clean up
df_distraction = df_distraction[
    (df_distraction['bloat_ratio'] >= 0) & 
    (df_distraction['completion_ratio'] < 6)
]

# Categorize
median_bloat = df_distraction['bloat_ratio'].median()
df_distraction['Bloat Category'] = np.where(
    df_distraction['bloat_ratio'] > median_bloat, 'High Bloat', 'Low Bloat'
)

group_high = df_distraction[df_distraction['Bloat Category'] == 'High Bloat']['completion_ratio']
group_low = df_distraction[df_distraction['Bloat Category'] == 'Low Bloat']['completion_ratio']

# --- 2. Advanced Statistics ---
# A. T-Test
t_stat, p_val = stats.ttest_ind(group_high, group_low, equal_var=False)

# B. Cohen's d (Effect Size)
# Formula: (Mean1 - Mean2) / Pooled Standard Deviation
n1, n2 = len(group_high), len(group_low)
var1, var2 = np.var(group_high, ddof=1), np.var(group_low, ddof=1)
pooled_se = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
cohens_d = (group_high.mean() - group_low.mean()) / pooled_se

# C. Confidence Interval (95%)
ci_low = group_low.mean() - 1.96 * (group_low.std() / np.sqrt(n2))
ci_high = group_low.mean() + 1.96 * (group_low.std() / np.sqrt(n2))

# --- 3. The "Fun" Report ---
print(f"--- 🎮 THE BLOAT REPORT 🎮 ---")
print(f"Total Games Analyzed: {len(df_distraction)}")
print(f"The 'Bloat Line' (Median): {median_bloat:.2f} (Games above this are 'Stuffed')")
print(f"\n--- The Scoreboard ---")
print(f"Focused Games Completion: {group_low.mean():.2f}x of campaign length")
print(f"Stuffed Games Completion: {group_high.mean():.2f}x of campaign length")
print(f"\n--- The Verdict ---")
print(f"T-Statistic: {t_stat:.2f}")
print(f"P-Value: {p_val:.2e}")
print(f"Cohen's d (Effect Size): {cohens_d:.2f}")

if cohens_d > 0.5:
    effect_text = "MEDIUM to LARGE effect! (Players definitely notice)"
elif cohens_d > 0.2:
    effect_text = "SMALL effect. (Noticeable but not huge)"
else:
    effect_text = "NEGLIGIBLE effect."

print(f"Analysis: {effect_text}")
if p_val < 0.05:
    print("Conclusion: MYTH BUSTED! 🚨 Bloated games actually keep players longer!")
else:
    print("Conclusion: PLAUSIBLE. Bloat might actually be boring.")

# --- 4. Annotated Visualization ---
plt.figure(figsize=(12, 7))

sns.histplot(
    data=df_distraction, 
    x='completion_ratio', 
    hue='Bloat Category', 
    element='step',     
    stat='density',     
    common_norm=False, 
    kde=True,
    palette='viridis',
    alpha=0.15
)

# Add Text Annotation directly on plot
stats_text = (
    f"T-Stat: {t_stat:.2f}\n"
    f"P-Value: {p_val:.1e}\n"
    f"Cohen's d: {cohens_d:.2f}\n"
    f"Effect: {effect_text.split('!')[0]}"
)
plt.text(
    x=3.5, y=0.4, 
    s=stats_text, 
    bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.5'),
    fontsize=11
)

plt.title('The Distraction Effect: Do "Bloated" Games Get Finished?\n(With Statistical Annotations)', fontsize=14)
plt.xlabel('Completion Ratio (Avg Playtime / Main Story)', fontsize=12)
plt.xlim(0, 5)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# <span style="color: pink; font-weight: bold;">
# Question 4: The Whale Indicator: Do multiplayer games exhibit a significantly larger skew between average and median playtime than single-player games?
# </span>

# %%
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import numpy as np

# --- 1. Data Prep ---
# (Assumes single_player_df and multiplayer_df are loaded)

# Calculate Skew Ratio: (Average - Median) / Median
single_player_df['whale_skew'] = (single_player_df['average_forever'] - single_player_df['median_forever']) / single_player_df['median_forever']
multiplayer_df['whale_skew'] = (multiplayer_df['average_forever'] - multiplayer_df['median_forever']) / multiplayer_df['median_forever']

# --- 2. Statistical Test ---
sp_skew = single_player_df['whale_skew']
mp_skew = multiplayer_df['whale_skew']

stat, p_val = stats.mannwhitneyu(mp_skew, sp_skew, alternative='greater')

print(f"--- 🐋 THE WHALE INDICATOR 🐋 ---")
print(f"Single Player Median Skew: {sp_skew.median():.2f}")
print(f"Multiplayer Median Skew:   {mp_skew.median():.2f}")
print(f"P-Value: {p_val:.2e}")

# --- 3. Visualization (Histogram) ---
# Combine for plotting
plot_df = pd.concat([
    pd.DataFrame({'Skew Ratio': sp_skew, 'Genre': 'Single Player'}),
    pd.DataFrame({'Skew Ratio': mp_skew, 'Genre': 'Multiplayer'})
])

# For the histogram, we focus on the "normal" range (0 to 5) so the plot isn't squashed.
# (There are outliers with skew > 20, but they make the graph unreadable)
plot_df_clean = plot_df[plot_df['Skew Ratio'] < 5]

plt.figure(figsize=(10, 6))

sns.histplot(
    data=plot_df_clean, 
    x='Skew Ratio', 
    hue='Genre', 
    element='step',     # Outline style (cleaner overlap)
    stat='density',     # Normalize bars to compare shapes
    common_norm=False, 
    kde=True,           # Add the smooth line
    palette='Set2',
    alpha=0.3
)

plt.title('The Whale Indicator: Playtime Skew Distribution', fontsize=14)
plt.xlabel('Skew Ratio ((Avg - Med) / Med)', fontsize=12)
plt.ylabel('Density of Games', fontsize=12)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


