#!/usr/bin/env python
# coding: utf-8

# In[8]:


# Simple Recommendation System using Collaborative Filtering
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error

# Load data (or create synthetic data)
def create_synthetic_data(n_users=100, n_items=50, n_ratings=5000):
    np.random.seed(42)
    user_ids = np.random.randint(1, n_users + 1, n_ratings)
    item_ids = np.random.randint(1, n_items + 1, n_ratings)
    ratings_values = np.random.choice([1, 2, 3, 4, 5], n_ratings, p=[0.05, 0.10, 0.25, 0.35, 0.25])
    
    ratings = pd.DataFrame({
        'user_id': user_ids,
        'item_id': item_ids,
        'rating': ratings_values
    })
    
    return ratings.drop_duplicates(subset=['user_id', 'item_id'])

# Create data and split into train/test sets
ratings = create_synthetic_data()
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# Create user-item matrix
user_item_matrix = ratings.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print(f"User-Item Matrix Shape: {user_item_matrix.shape}")

# Memory-Based Collaborative Filtering: User-Based
def user_based_recommendations(user_id, user_item_matrix, k=10, n_recommendations=5):
    """Generate recommendations for a user using user-based collaborative filtering"""
    # Calculate user similarity
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, 
                                     index=user_item_matrix.index, 
                                     columns=user_item_matrix.index)
    
    # Find most similar users
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)[1:k+1]
    
    # Items the user has already rated
    user_rated_items = user_item_matrix.loc[user_id].replace(0, np.nan).dropna().index
    
    # Generate predictions for unrated items
    recommendations = {}
    for item in user_item_matrix.columns:
        if item not in user_rated_items:
            # Get ratings from similar users for this item
            item_ratings = user_item_matrix.loc[similar_users.index, item]
            # Remove zeros (unrated items)
            item_ratings = item_ratings[item_ratings > 0]
            
            if not item_ratings.empty:
                # Calculate weighted average
                weights = similar_users.loc[item_ratings.index]
                predicted_rating = np.sum(item_ratings * weights) / np.sum(weights)
                recommendations[item] = predicted_rating
    
    # Return top n recommendations
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

# Memory-Based Collaborative Filtering: Item-Based
def item_based_recommendations(user_id, user_item_matrix, k=10, n_recommendations=5):
    """Generate recommendations for a user using item-based collaborative filtering"""
    # Calculate item similarity
    item_similarity = cosine_similarity(user_item_matrix.T)
    item_similarity_df = pd.DataFrame(item_similarity, 
                                     index=user_item_matrix.columns, 
                                     columns=user_item_matrix.columns)
    
    # Items the user has rated
    user_ratings = user_item_matrix.loc[user_id]
    rated_items = user_ratings[user_ratings > 0].index
    
    # Generate predictions for unrated items
    recommendations = {}
    for item in user_item_matrix.columns:
        if item not in rated_items:
            # Find k most similar items that the user has rated
            similar_items = item_similarity_df[item].sort_values(ascending=False)
            similar_rated_items = similar_items[similar_items.index.isin(rated_items)][:k]
            
            if not similar_rated_items.empty:
                # Get user's ratings for similar items
                user_ratings_for_similar = user_ratings[similar_rated_items.index]
                # Calculate weighted average
                predicted_rating = np.sum(user_ratings_for_similar * similar_rated_items) / np.sum(similar_rated_items)
                recommendations[item] = predicted_rating
    
    # Return top n recommendations
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:n_recommendations]

# Matrix Factorization from scratch
class MatrixFactorization:
    def __init__(self, n_factors=20, learning_rate=0.01, regularization=0.1, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        
    def fit(self, user_item_matrix):
        self.user_item_matrix = user_item_matrix
        self.n_users, self.n_items = user_item_matrix.shape
        
        # Initialize user and item factors with small random values
        np.random.seed(42)
        self.user_factors = np.random.normal(scale=0.1, size=(self.n_users, self.n_factors))
        self.item_factors = np.random.normal(scale=0.1, size=(self.n_items, self.n_factors))
        
        # Create matrices of indices of non-zero values
        nonzero_indices = np.where(user_item_matrix.values > 0)
        self.user_indices = nonzero_indices[0]
        self.item_indices = nonzero_indices[1]
        self.ratings = user_item_matrix.values[nonzero_indices]
        
        # Train the model
        self._train()
        
        return self
    
    def _train(self):
        for epoch in range(self.n_epochs):
            # Calculate predicted ratings
            predicted = self._predict_all()
            
            # Calculate error
            error = self.ratings - predicted
            
            # Update factors
            for i in range(len(self.ratings)):
                user_idx = self.user_indices[i]
                item_idx = self.item_indices[i]
                err = error[i]
                
                # Update user and item factors
                user_factor = self.user_factors[user_idx].copy()
                item_factor = self.item_factors[item_idx].copy()
                
                self.user_factors[user_idx] += self.learning_rate * (err * item_factor - self.regularization * user_factor)
                self.item_factors[item_idx] += self.learning_rate * (err * user_factor - self.regularization * item_factor)
            
            # Calculate RMSE for this epoch
            if epoch % 10 == 0:
                rmse = np.sqrt(np.mean(error**2))
                print(f"Epoch {epoch}: RMSE = {rmse:.4f}")
    
    def _predict_all(self):
        """Predict all ratings in the training set"""
        return np.sum(self.user_factors[self.user_indices] * self.item_factors[self.item_indices], axis=1)
    
    def predict(self, user_idx, item_idx):
        """Predict rating for a specific user-item pair"""
        return np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
    
    def recommend_items(self, user_id, n=5):
        """Generate top-N recommendations for a user"""
        # Map user_id to matrix index
        user_idx = list(self.user_item_matrix.index).index(user_id)
        
        # Get items the user has already rated
        rated_items = self.user_item_matrix.loc[user_id].replace(0, np.nan).dropna().index
        rated_item_indices = [list(self.user_item_matrix.columns).index(item) for item in rated_items]
        
        # Predict ratings for all items
        predictions = []
        for item_idx in range(self.n_items):
            if item_idx not in rated_item_indices:
                item_id = self.user_item_matrix.columns[item_idx]
                predicted_rating = self.predict(user_idx, item_idx)
                predictions.append((item_id, predicted_rating))
        
        # Return top N recommendations
        return sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

# Evaluation function
def evaluate_recommendations(predictions, test_data):
    """Calculate RMSE for predictions"""
    true_ratings = []
    pred_ratings = []
    
    for _, row in test_data.iterrows():
        user_id = row['user_id']
        item_id = row['item_id']
        
        # Find prediction for this user-item pair
        for pred_user_id, pred_items in predictions.items():
            if pred_user_id == user_id:
                for pred_item_id, pred_rating in pred_items:
                    if pred_item_id == item_id:
                        true_ratings.append(row['rating'])
                        pred_ratings.append(pred_rating)
                        break
    
    if true_ratings:
        rmse = np.sqrt(mean_squared_error(true_ratings, pred_ratings))
        return rmse
    else:
        return None

# Main execution
if __name__ == "__main__":
    print("Sample data:")
    print(ratings.head())
    
    # Test user-based CF
    print("\nUser-based Collaborative Filtering:")
    sample_user = user_item_matrix.index[0]
    user_cf_recs = user_based_recommendations(sample_user, user_item_matrix)
    print(f"Top 5 recommendations for user {sample_user}:")
    for item_id, rating in user_cf_recs:
        print(f"Item {item_id}: Predicted rating = {rating:.2f}")
    
    # Test item-based CF
    print("\nItem-based Collaborative Filtering:")
    item_cf_recs = item_based_recommendations(sample_user, user_item_matrix)
    print(f"Top 5 recommendations for user {sample_user}:")
    for item_id, rating in item_cf_recs:
        print(f"Item {item_id}: Predicted rating = {rating:.2f}")
    
    # Test Matrix Factorization
    print("\nMatrix Factorization:")
    mf = MatrixFactorization(n_factors=10, n_epochs=50)
    mf.fit(user_item_matrix)
    
    mf_recs = mf.recommend_items(sample_user)
    print(f"Top 5 recommendations for user {sample_user}:")
    for item_id, rating in mf_recs:
        print(f"Item {item_id}: Predicted rating = {rating:.2f}")
    
    print("\nRecommendation system training complete!")


# In[ ]:




