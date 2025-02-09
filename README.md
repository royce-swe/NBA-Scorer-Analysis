# NBA-Scorer-Analysis  
**NBA Top Scorer Prediction & College Analysis** 🏀📊  

This project analyzes NBA player data to determine which colleges produce the best scorers and uses machine learning to predict whether a player is likely to be a top scorer (20+ PPG).  

## Key Features:  

### 📊 Data Analysis:  
- Identifies colleges that have produced the most top NBA scorers and colleges that have produced the most star players (20+ PPG).  
- Calculates the average PPG of players from each college.  
- Visualizes results with bar charts.  

### 🤖 Machine Learning Model:  
- Trains a **Random Forest Classifier** to predict if a player will be a top scorer.  
- Uses **height, weight, age, and college** as features.  
- Handles categorical college data with **Label Encoding**.  
- Standardizes numerical features with **StandardScaler**.  

### 📈 Visualization & Prediction:  
- **Confusion Matrix** to evaluate model accuracy.  
- Function to predict a player's **top scorer potential** based on input stats.  
- Automated search for feature combinations likely to predict a **top scorer**.  

### 🔍 Results & Findings:  
- Displays the **model's accuracy**.  
- Checks for **bias in classification**.  
