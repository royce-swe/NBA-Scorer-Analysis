import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import numpy as np

import warnings
warnings.simplefilter("ignore")

# Load data
df = pd.read_csv("all_seasons.csv")

# Removes players who did not attend college
df = df.dropna(subset=["college"])

# Prevents duplicate players
df_unique_players = df.groupby("player_name").agg({"college": "first", "pts": "mean"}).reset_index()

# Group by college and calculate the average PPG per player
college_avg_ppg = df_unique_players.groupby("college")["pts"].mean().sort_values(ascending=False)


# Finds the top 50 highest player point averages in NBA History
topScorers = df.nlargest(50, "pts")[["player_name", "pts", "college"]]

# Count how many times each college appears
college_counts = topScorers["college"].value_counts()


### MACHINE LEARNING MODEL ###

# Define top scorers (players with >20 PPG in the NBA)
df["top_scorer"] = (df["pts"] > 20).astype(int)

# Encode "college" into numbers
label_encoder = LabelEncoder()
df["college_encoded"] = label_encoder.fit_transform(df["college"])

# Select features (Player Height, Weight, College, Age)
features = ["player_height", "player_weight", "college_encoded", "age"]
X = df[features]
y = df["top_scorer"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[features])

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create subplots to show both graphs side by side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adding another subplot for the confusion matrix

# First plot: Colleges that produced the most top 50 scorers
college_counts.head(10).plot(kind="bar", color="royalblue", ax=axes[0])
axes[0].set_xlabel("College")
axes[0].set_ylabel("Number of Top 50 Scorers")
axes[0].set_title("Colleges That Produced the Most Top 50 NBA Scorers")
axes[0].tick_params(axis='x', rotation=45)

# Second plot: Top colleges based on average PPG
college_avg_ppg.head(10).plot(kind="bar", color="darkorange", ax=axes[1])
axes[1].set_xlabel("College")
axes[1].set_ylabel("Average Career PPG")
axes[1].set_title("Top Colleges Producing the Best NBA Scorers")
axes[1].tick_params(axis='x', rotation=45)

# Third plot: Confusion matrix of top scorer predictions
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Top Scorer", "Top Scorer"],
            yticklabels=["Not Top Scorer", "Top Scorer"], ax=axes[2])
axes[2].set_xlabel("Predicted Label")
axes[2].set_ylabel("True Label")
axes[2].set_title("Confusion Matrix of Top Scorer Predictions")

# Adjust layout for clarity
plt.tight_layout()
plt.show()


# Make a prediction for a player (example: Height=6'8", Weight=250 lbs, College='Duke', Age=22)
def predict_top_scorer(height, weight, college, age):
    college_encoded = label_encoder.transform([college])[0]
    player_data = [[height, weight, college_encoded, age]]
    prediction = model.predict(player_data)

    if prediction == 1:
        return "Top Scorer!"
    else:
        return "Not a Top Scorer."


# Example prediction
prediction1 = predict_top_scorer(195.58, 99.79024, "Arizona State", 26.0)
print(f"Prediction for a 195.58\' cm, 99.79 kg, 26-year-old Arizona State player: {prediction1}")
print()


# Function to find a combination likely to be a top scorer
def find_top_scorer_combination(model, label_encoder, scaler):
    heights = np.linspace(180, 220, 10)  # Heights in cm (roughly 5'11" to 7'3")
    weights = np.linspace(80, 130, 10)  # Weights in kg (176 to 286 lbs)
    ages = np.linspace(18, 35, 10)  # Ages (18 to 35 years)

    colleges = df["college"].unique()  # Get list of unique colleges

    for college in colleges:
        for h in heights:
            for w in weights:
                for a in ages:
                    # Encode the college
                    college_encoded = label_encoder.transform([college])[0]

                    # Scale input values
                    input_features = np.array([[h, w, college_encoded, a]])
                    input_features_scaled = scaler.transform(input_features)

                    # Predict if this is a top scorer
                    prediction = model.predict(input_features_scaled)[0]

                    if prediction == 1:
                        print(f"Found a potential top scorer!")
                        print(f"Height: {h:.2f} cm, Weight: {w:.2f} kg, Age: {a:.2f}, College: {college}")
                        return h, w, college, a  # Return first found top scorer

    print("No top scorer combination found. Model might be biased.")
    return None


# Call the function to find a top scorer combination
best_scorer = find_top_scorer_combination(model, label_encoder, scaler)

print()
print("Check for bias by anaylzing non top scorer count vs top scorer count:")
print(df["top_scorer"].value_counts())

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print()
print(f"Model Accuracy: {accuracy:.2f}")
