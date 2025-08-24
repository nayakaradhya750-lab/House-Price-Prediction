# House-Price-Prediction
Predict house prices using California Housing datasetHouse Price Prediction

## Project Overview
This project predicts house prices using the **California Housing dataset**.  
It is a classic **regression problem**, where the model learns from numerical features such as:

- Median Income
- House Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Latitude
- Longitude  

and predicts the **median house value** in that area.

---

## Features Used
- `MedInc` : Median income in block  
- `HouseAge` : Median house age in block  
- `AveRooms` : Average rooms per household  
- `AveBedrms` : Average bedrooms per household  
- `Population` : Population in block  
- `AveOccup` : Average occupants per household  
- `Latitude` : Block latitude  
- `Longitude` : Block longitude  

---

## Steps Performed
1. **Data Loading**: Loaded California Housing dataset using `scikit-learn`.  
2. **Exploratory Data Analysis (EDA)**: Visualized data distributions and correlations.  
3. **Preprocessing**: 
   - Split data into training and testing sets  
   - Standardized numerical features  
4. **Modeling**: Trained a **Linear Regression** model.  
5. **Evaluation**: 
   - Mean Squared Error (MSE)  
   - R-squared score  
6. **Visualization**: Plotted predicted vs actual house prices.

---

## How to Run
1. Clone this repository:

```bash
git clone https://github.com/your-username/House-Price-Prediction.git

