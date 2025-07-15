# -*- coding: utf-8 -*-
import os

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.stats import ttest_ind

from time import time
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import seaborn as sns

import pickle
import tldextract

from matplotlib.colors import LinearSegmentedColormap
import plotly.graph_objects as go


class PartisanshipEstimator:
    """
    Estimates partisanship of users and articles using a bipartite network
    and personalized PageRank.
    """
    def __init__(self, comments_df, seed_scores):
        """
        Initializes the estimator.

        Args:
            comments_df (pd.DataFrame): DataFrame with 'publisher_id' and 'news_id' columns.
            seed_scores (dict): Dictionary with news_id as keys and partisanship (-1, 1, 0) as values.
        """
        self.comments_df = comments_df
        self.seed_scores = seed_scores
        self.user_map = {}
        self.article_map = {}
        self.nodes = []
        self.T = None # Transition matrix

        self.preprocessed =False

    def _prepare_data(self, user_col = 'publisher_id', item_col = 'news_id'):
        """
        Prepares data by creating node mappings and calculating edge weights.
        This step converts symbolic IDs into numerical indices for matrix construction.
        """
        print("Step 1: Preparing data and creating node indices...")

        # Aggregate to get edge weights (number of comments per user-article pair)
        edge_weights = self.comments_df.groupby([user_col, item_col]).size().reset_index(name='weight')

        # Get unique users and articles
        unique_users = edge_weights[user_col].unique()
        unique_articles = edge_weights[item_col].unique()

        # Create mappings from ID to integer index
        self.user_map = {uid: i for i, uid in enumerate(unique_users)}
        self.article_map = {aid: i + len(unique_users) for i, aid in enumerate(unique_articles)}

        self.n_users = len(unique_users)
        self.n_articles = len(unique_articles)
        self.n_total_nodes = self.n_users + self.n_articles

        # Create reverse mapping to get original IDs back
        self.nodes = list(unique_users) + list(unique_articles)

        # Map IDs in the edge list to their new integer indices
        self.rows = edge_weights[user_col].map(self.user_map).values
        self.cols = edge_weights[item_col].map(self.article_map).values
        self.weights = edge_weights['weight'].values

        self.preprocessed = True

    def _build_transition_matrix(self):
        """
        Builds the sparse transition matrix for the random walk.
        The matrix represents the entire bipartite graph.
        """
        print("Step 2: Building the sparse transition matrix...")

        # Construct the bipartite adjacency matrix
        # It's a square matrix of size (n_users + n_articles)
        # The top-right block is user-article connections, bottom-left is article-user (transpose)
        adj_matrix = csr_matrix((self.weights, (self.rows, self.cols)), shape=(self.n_total_nodes, self.n_total_nodes))
        adj_matrix = adj_matrix + adj_matrix.T

        # Normalize the adjacency matrix to create the transition matrix T
        # T_ij = probability of transitioning from node j to node i
        print("Step 3: Normalizing the matrix...")

        # Calculate out-degree (sum of weights for each column)
        out_degree = np.array(adj_matrix.sum(axis=0)).flatten()

        # Avoid division by zero for nodes with no outgoing edges
        out_degree[out_degree == 0] = 1

        # Create a diagonal matrix of the inverse of the out-degrees
        inv_out_degree_matrix = diags(1.0 / out_degree)

        # T = A * D^-1
        self.T = adj_matrix.dot(inv_out_degree_matrix)

        self.preprocessed = True

    def _run_pagerank(self, p, alpha=0.85, max_iter=100, tol=1e-6):
        """
        Executes the personalized PageRank algorithm using power iteration.

        Args:
            p (np.array): The personalization vector.
            alpha (float): The damping factor (teleportation probability).
            max_iter (int): Maximum number of iterations.
            tol (float): Convergence tolerance.

        Returns:
            np.array: The resulting PageRank vector.
        """
        # Start with a uniform rank distribution
        r = np.full(self.n_total_nodes, 1.0 / self.n_total_nodes)

        for i in range(max_iter):
            r_last = r.copy()
            # The core PageRank equation for sparse matrices
            r = alpha * self.T.dot(r) + (1 - alpha) * p

            # Check for convergence
            if np.linalg.norm(r - r_last, 1) < tol:
                print(f"  Converged after {i + 1} iterations.")
                return r

        print(f"  Did not converge after {max_iter} iterations.")
        return r

    def estimate(self, alpha=0.85):
        """
        The main method to run the entire estimation pipeline.

        Args:
            alpha (float): The damping factor for PageRank.

        Returns:
            pd.DataFrame: A DataFrame with node IDs, types, and their partisanship scores.
        """
        if not self.preprocessed: self._prepare_data()
        self._build_transition_matrix()

        print("Step 4: Setting up personalization vectors...")
        # Create personalization vectors for left and right-leaning seeds
        p_left = np.zeros(self.n_total_nodes)
        p_right = np.zeros(self.n_total_nodes)

        left_seeds = [self.article_map[aid] for aid, score in self.seed_scores.items() if score == -1 and aid in self.article_map]
        right_seeds = [self.article_map[aid] for aid, score in self.seed_scores.items() if score == 1 and aid in self.article_map]

        if not left_seeds or not right_seeds:
            raise ValueError("Seed scores must contain both left (-1) and right (1) leaning articles.")

        p_left[left_seeds] = 1.0 / len(left_seeds)
        p_right[right_seeds] = 1.0 / len(right_seeds)

        print("Step 5: Running Personalized PageRank for RIGHT-leaning seeds...")
        pagerank_right = self._run_pagerank(p_right, alpha)

        print("Step 5: Running Personalized PageRank for LEFT-leaning seeds...")
        pagerank_left = self._run_pagerank(p_left, alpha)

        print("Step 6: Calculating final partisanship scores...")
        # To avoid division by zero, add a small epsilon where the sum is zero
        denominator = pagerank_right + pagerank_left
        denominator[denominator == 0] = 1e-9 # Avoid division by zero

        partisanship = (pagerank_right - pagerank_left)/ denominator

        # Create the final results DataFrame
        results_df = pd.DataFrame({
            'id': self.nodes,
            'type': ['user'] * self.n_users + ['article'] * self.n_articles,
            'partisanship_score': partisanship,
            'pagerank_left': pagerank_left,
            'pagerank_right': pagerank_right
        })

        return results_df