import pandas as pd
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# Load your data
# Here we assume the data is in a CSV file with 'userId', 'itemId', 'rating' columns
data = pd.read_csv('user_item_ratings.csv')

# Create a Reader object to parse the data
reader = Reader(rating_scale=(1, 5))

# Load the dataset from the pandas dataframe
dataset = Dataset.load_from_df(data[['userId', 'itemId', 'rating']], reader)

# Split the dataset into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.2)

# Use Singular Value Decomposition (SVD) algorithm
model = SVD()

# Train the model
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Compute and print RMSE
rmse = accuracy.rmse(predictions)

# Function to get top N recommendations for a given user
def get_top_n_recommendations(predictions, n=10):
    # First map the predictions to each user
    top_n = {}
    for uid, iid, true_r, est, _ in predictions:
        if not top_n.get(uid):
            top_n[uid] = []
        top_n[uid].append((iid, est))

    # Sort the predictions for each user and retrieve the top N
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

# Get top 10 recommendations for each user
top_n_recommendations = get_top_n_recommendations(predictions, n=10)

# Print the recommendations
for user_id, recommendations in top_n_recommendations.items():
    print(f"User {user_id} recommendations:")
    for item_id, rating in recommendations:
        print(f"    Item {item_id} - Predicted Rating: {rating:.2f}")
