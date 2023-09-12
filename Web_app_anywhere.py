from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
from io import BytesIO
import base64

app = Flask(__name__)

# Load and preprocess your data
data_pt1 = pd.read_csv('Team_Analytics_Data_pt1.csv')
data_pt1 = data_pt1.dropna()
data_pt2 = pd.read_csv('Team_Analytics_Data_pt2.csv')
data_pt2 = data_pt2.dropna()
merged_df = pd.merge(data_pt1, data_pt2, on='Game_Key', how='inner')
columns_to_keep = ['NPIE_A', 'PM_A', 'NDOFF_A', 'NSFF_A', 'SEASON_YEAR_A_x']
data = merged_df[columns_to_keep]

validation = data[data['SEASON_YEAR_A_x'] == '2022-23']
modelData = data[data['SEASON_YEAR_A_x'] != '2022-23'].sample(frac=1)
X = modelData.drop(columns=['PM_A', 'SEASON_YEAR_A_x'], axis=1)
y = modelData['PM_A']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33)

# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(X_train)
scaled_data_train = scaler.transform(X_train)

scaler.fit(X_test)
scaled_data_test = scaler.transform(X_test)
# Standard Scaling Prediction Variables
scaler = preprocessing.StandardScaler()
scaler.fit(validation.drop(['PM_A','SEASON_YEAR_A_x'],axis=1))
scaled_val_data = scaler.transform(validation.drop(['PM_A','SEASON_YEAR_A_x'],axis=1))
y_valid = validation['PM_A']

# Create your machine learning models and preprocessors here
random.seed(420)
rfr = RandomForestRegressor()
rfr.fit(scaled_data_train,y_train)
lr = LinearRegression()
lr.fit(scaled_data_train,y_train)
knnNorm = KNeighborsRegressor(n_neighbors= 10)
knnNorm.fit(scaled_data_train, y_train)
gbr = GradientBoostingRegressor()
gbr.fit(scaled_data_train,y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/simulate', methods=['GET'])
def simulate():
    return render_template('simulate.html')
    
    

@app.route('/simulation_result', methods=['POST'])
def simulation_result():
    if request.method == 'POST':
        user_numbers = []
        for i in range(6):
            user_input = float(request.form[f'number_{i+1}'])
            user_numbers.append(user_input)
        
        mean_1, std_dev_1, mean_2, std_dev_2, mean_3, std_dev_3 = user_numbers
        
        # Simulate using the user-provided values
        num_samples = 10000
        samples_1 = np.random.normal(mean_1, std_dev_1, num_samples)
        samples_2 = np.random.normal(mean_2, std_dev_2, num_samples)
        samples_3 = np.random.normal(mean_3, std_dev_3, num_samples)

        # Create a pandas DataFrame using the simulated samples
        data_simulated = {
            'NPIE_A': samples_1,
            'NDOFF_A': samples_2,
            'NSFF_A': samples_3
        }

        df_simulated = pd.DataFrame(data_simulated, columns=['NPIE_A', 'NDOFF_A', 'NSFF_A'])

        # Apply any necessary preprocessing on df_simulated here
        scaled_MyGameSim = scaler.transform(df_simulated)
        # ...

        # Use your machine learning models for prediction
        rfr_predicted_sim = rfr.predict(scaled_MyGameSim)
        lr_predicted_sim = lr.predict(scaled_MyGameSim)
        knnNorm_predicted_sim = knnNorm.predict(scaled_MyGameSim)
        gbr_predicted_sim = gbr.predict(scaled_MyGameSim)
        average_predictions = (rfr_predicted_sim + lr_predicted_sim + knnNorm_predicted_sim + gbr_predicted_sim) / 4

        #Histogram creation output:
        num_samples = 100

        # Calculate mean of the average predictions
        mean_average = np.mean(average_predictions)
        std_average = np.std(average_predictions)

        # Create an array to store bootstrapped sample means
        bootstrapped_means = np.zeros(num_samples)

        # Perform bootstrapping
        for i in range(num_samples):
            # Randomly sample indices with replacement
            bootstrap_indices = np.random.choice(len(average_predictions), size=len(average_predictions), replace=True)

            # Use the sampled indices to create a bootstrap sample
            bootstrap_sample = average_predictions[bootstrap_indices]

            # Calculate the mean of the bootstrap sample
            bootstrapped_means[i] = np.mean(bootstrap_sample)

        # Calculate the 95% confidence interval
        confidence_interval = np.percentile(bootstrapped_means, [2.5, 97.5])

        # Print the mean and confidence interval
        print(f"Mean Average: {mean_average}")
        print(f"Standard Deviation: {std_average}")
        print(f"95% Confidence Interval: {confidence_interval}")


        # Create a figure and axis with a larger figure size
        plt.figure(figsize=(10, 6))

        # Create a histogram of the average predictions with enhanced visuals
        plt.hist(bootstrapped_means, bins=20, color='skyblue', edgecolor='gray', alpha=0.7)

        # Add vertical lines for mean and confidence intervals with labels
        plt.axvline(x=mean_average, color='red', linestyle='dashed', linewidth=2, label='Mean')
        
        # Add labels and legend
        plt.xlabel('Average Plus Minus Predictions')
        plt.ylabel('Frequency')
        plt.title('Histogram of Plus Minus Bootstrap')
        plt.legend()

        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout to prevent label cutoff and overlap
        plt.tight_layout()

        # Save the plot as a BytesIO object and encode it to base64
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        img_base64_1 = base64.b64encode(img_stream.read()).decode('utf-8')
        plt.close()
        
        # Create the second histogram with enhanced visuals (replace this with your code)
        plt.figure(figsize=(10, 6))
        
        # Calculate mean and 25% confidence interval of the average predictions
        mean_average = np.mean(average_predictions)
        std_average = np.std(average_predictions)
        confidence_interval = np.percentile(average_predictions, [37.5, 62.5])  # 25% confidence interval

        # Print the values of the 25% confidence interval
        print(f"Mean Average: {mean_average}")
        print(f"Standard Deviation: {std_average}")
        print(f"Lower Bound of 25% Confidence Interval: {confidence_interval[0]}")
        print(f"Upper Bound of 25% Confidence Interval: {confidence_interval[1]}")

        # Create a histogram of the average predictions
        plt.hist(average_predictions, bins=20, color='blue', alpha=0.7)

        # Add mean and 70% confidence interval lines to the histogram
        plt.axvline(x=mean_average, color='red', linestyle='dashed', linewidth=2, label='Mean')
        plt.axvline(x=confidence_interval[0], color='green', linestyle='dashed', linewidth=2, label='25% Confidence Interval')
        plt.axvline(x=confidence_interval[1], color='green', linestyle='dashed', linewidth=2)

        # Add labels and legend
        plt.xlabel('Plus Minus')
        plt.ylabel('Frequency')
        plt.title('Histogram of Plus Minus')
        plt.legend()
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout to prevent label cutoff and overlap
        plt.tight_layout()

        # Save the plot as a BytesIO object and encode it to base64
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        img_base64_2 = base64.b64encode(img_stream.read()).decode('utf-8')
        plt.close()
        
        # Create histogram 3 with enhanced visuals (replace this with your code)
        plt.figure(figsize=(10, 6))
        
        # Calculate the proportion of values greater than 1
        boolean_array = average_predictions > 1
        proportion_greater_than_1 = np.mean(boolean_array)
        # Convert boolean array to numbers (0s and 1s)
        boolean_array = boolean_array.astype(int)

        # Create a figure and axis with a larger figure size
        plt.figure(figsize=(10, 6))

        # Create a histogram of the proportion values with enhanced visuals
        plt.hist(boolean_array,  bins=[ -0.3, 0.3, 0.7, 1.3], color='blue', alpha=0.7,edgecolor='black')

        # Add mean line to the histogram
        plt.axvline(x=proportion_greater_than_1, color='red', linestyle='dashed', linewidth=2, label='Mean')
        plt.xticks([0, 0.5, 1], ['0', '0.5', '1'])
        # Add labels and legend
        plt.xlabel('Win Loss')
        plt.ylabel('Frequency')
        plt.title('Win Loss Histogram')
        plt.legend()
    
        # Print the mean and confidence interval values
        print(f"Mean Proportion: {proportion_greater_than_1}")
        
        # Add a grid for better readability
        plt.grid(True, linestyle='--', alpha=0.5)

        # Adjust layout to prevent label cutoff and overlap
        plt.tight_layout()

        # Save the plot as a BytesIO object and encode it to base64
        img_stream = BytesIO()
        plt.savefig(img_stream, format='png')
        img_stream.seek(0)
        img_base64_3 = base64.b64encode(img_stream.read()).decode('utf-8')
        plt.close()

        # Return a response to the user
        return render_template('simulation_result.html', result=np.mean(average_predictions), win_chance=np.mean(average_predictions>1),img_data_1=img_base64_1, img_data_2=img_base64_2, img_data_3=img_base64_3,
                           mean_average=mean_average, std_average=std_average,
                           confidence_interval=confidence_interval,proportion_greater_than_1=proportion_greater_than_1 
                           )

    return "Invalid request"

    


