import pandas as pd

# Load the data from the provided CSV files
potato_data = pd.read_csv(r"potato.csv")
states_data = pd.read_csv(r"states.csv")
rainfall_data = pd.read_csv(r"rainfall_new.csv")

# Cleaning up the states data by removing unnecessary columns
states_data = states_data[['State', 'District']]

# Converting Centre_Name to lowercase for easier matching later
potato_data['Centre_Name'] = potato_data['Centre_Name'].str.lower()
states_data['District'] = states_data['District'].str.lower()

# Merge potato data with states data to map Centre_Name to State using District
merged_data = pd.merge(potato_data, states_data, left_on='Centre_Name', right_on='District', how='left')

# Extract month and year from the Date column
merged_data['Date'] = pd.to_datetime(merged_data['Date'], format='%d-%m-%Y')
merged_data['Month'] = merged_data['Date'].dt.strftime('%b').str.upper()
merged_data['Year'] = merged_data['Date'].dt.year

# Correcting the 'State' column name to 'States/UTs' for the rainfall data
rainfall_data['States/UTs'] = rainfall_data['States/UTs'].str.lower()
merged_data['State'] = merged_data['State'].str.lower()

# Merge the merged_data with rainfall data on State and Year
final_data = pd.merge(merged_data, rainfall_data, left_on=['State', 'Year'], right_on=['States/UTs', 'YEAR'], how='left')

# Extract the rainfall data for the corresponding month
final_data['Rainfall'] = final_data.apply(lambda row: row[row['Month']], axis=1)

# Select the required columns for the final output
final_output = final_data[['State', 'Date', 'Rainfall', 'Price']]

# Save the final output to a CSV file
final_output.to_csv('final_potato_rainfall_data.csv', index=False)

print("Data processing complete. Output saved to 'final_potato_rainfall_data.csv'.")
