# Pranitha
Smart Shopping: Data and AI for Personalized E-Commerce
from surprise import Dataset, Reader
from surprise import KNNBasic
from surprise import accuracy
import pandas as pd

# Example: Simulated data of users and their ratings for products
# You can replace this with your actual user-item interaction data
data = {
    'user_id': ['user1', 'user2', 'user3', 'user4', 'user5', 'user6'],
    'item_id': ['product1', 'product2', 'product3', 'product1', 'product2', 'product3'],
    'rating': [5, 4, 4, 5, 3, 2]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Define the format for Surprise
reader = Reader(rating_scale=(1, 5))

# Load data into Surprise format
data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = data.split(n_folds=3)

# Use K-Nearest Neighbors (KNN) collaborative filtering
algo = KNNBasic()

# Train the algorithm
algo.fit(trainset)

# Test the algorithm on the testset
predictions = algo.test(testset)

# Calculate RMSE (Root Mean Squared Error)
rmse = accuracy.rmse(predictions)
print(f"RMSE: {rmse}")

# Making a prediction for a specific user-item pair
user = 'user1'
item = 'product3'
prediction = algo.predict(user, item)
print(f"Predicted rating for {user} on {item}: {prediction.est}")

# Get top-n recommendations for a user
def get_top_n(predictions, n=3):
    top_n = {}
    
    for uid, iid, true_r, est, _ in predictions:
        if uid not in top_n:
            top_n[uid] = []
        top_n[uid].append((iid, est))
    
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    
    return top_n

# Get top 3 recommendations for each user
top_n = get_top_n(predictions, n=3)
print("\nTop 3 recommendations for each user:")
for uid, user_ratings in top_n.items():
    print(f"{uid}: {user_ratings}")
