Partisanship Estimation using Personalized PageRank
This repository contains the Python source code for estimating the partisanship of users, news articles, and media agencies from a comment network. The methodology is based on applying the Personalized PageRank (PPR) algorithm to a bipartite graph of users and the articles they comment on.

This project was developed as part of a data analysis series detailed in our blog post: Mapping Media Bias: How We Used AI and a 'Biased Voter' Algorithm to Chart Partisanship on Yahoo! News Japan.

Overview
The core of this project is the PartisanshipEstimator class, which takes a dataset of user comments and a small set of "seed" articles with known political leanings (left or right) to infer the partisanship of all other nodes in the network.

The process works by:

Constructing a Network: Building a bipartite graph where users and articles are nodes, and comments represent the links between them.

AI-Powered Seeding: Using a pre-labeled set of seed articles (e.g., labeled via Large Language Models) to establish political "anchors" in the network.

Personalized PageRank: Running the PPR algorithm twice—once personalized for left-leaning seeds and once for right-leaning seeds.

Calculating Scores: Using the output of the two PPR runs to calculate a final, normalized partisanship score for every user and article, ranging from -1 (most left-leaning) to +1 (most right-leaning).

Features
Efficiently handles large datasets using sparse matrix computations with scipy.sparse.

Calculates partisanship scores for users, articles, and can be aggregated to score media agencies.

Encapsulated in a clean, reusable Python class.

Includes helper functions for visualizing the score distributions.

Requirements
The necessary Python libraries are listed in the requirements.txt file.

Installation
Clone the repository:

git clone https://github.com/CSS-Laboratory/Partisanship-Estimation.git
cd your-repo-name

Install the required packages using pip:
'''python
pip install -r requirements.txt
'''
Usage
To use the estimator, you need to prepare two main pieces of data:

comments_df: A pandas DataFrame containing the comment data. It must have at least two columns: one for the user ID (e.g., publisher_id) and one for the article ID (news_id).

seed_scores: A Python dictionary where keys are the news_id of the seed articles and values are their partisanship score (-1 for left-leaning, 1 for right-leaning).

Example
Here is a basic example of how to use the PartisanshipEstimator class.

'''python
import pandas as pd
import pickle
from ppr_methods import PartisanshipEstimator # Assuming the class is in this file

# 1. Load your data
# This is an example; you will load your actual data here.
# comments_df should have columns like ['publisher_id', 'news_id']
comments_df = pd.read_pickle("path/to/your/comment_data.pkl")

# seed_scores is a dict like {'news_id_1': -1, 'news_id_2': 1, ...}
with open("path/to/your/seed_scores.pkl", "rb") as f:
    seed_scores = pickle.load(f)

# 2. Initialize and run the estimator
estimator = PartisanshipEstimator(comments_df, seed_scores)
final_scores_df = estimator.estimate(alpha=0.85)

# 3. Analyze the results
# The output is a DataFrame with IDs, types (user/article), and scores.
print("Partisanship scores calculated successfully!")
print(final_scores_df.head())

# Example: Get the top 10 most right-leaning users
right_leaning_users = final_scores_df[final_scores_df['type'] == 'user'].sort_values(
    by='partisanship_score', ascending=False
).head(10)

print("\nTop 10 most right-leaning users:")
print(right_leaning_users)
'''

Code Structure
ppr_methods.py: Contains the main PartisanshipEstimator class and its methods.

__init__(self, comments_df, seed_scores): Initializes the estimator with data.

estimate(self, alpha=0.85): The main public method that runs the full pipeline and returns the final scores.

_prepare_data(): Internal method for data preprocessing and creating node mappings.

_build_transition_matrix(): Builds the normalized sparse transition matrix.

_run_pagerank(...): The core power iteration loop for the PPR algorithm.

Helper Functions: The script also includes functions like kde_plot for creating kernel density estimate plots of the score distributions.

Citation
If you use this methodology or code in your research, please consider citing our original blog post:

Kunhao Yang. (2025). "Mapping Media Bias: How We Used AI and a 'Biased Voter' Algorithm to Chart Partisanship on Yahoo! News Japan." CSS Lab. [Link]
