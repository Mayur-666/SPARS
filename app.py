from flask import Flask, render_template, request
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.ioff()

app = Flask(__name__)

# Load the dataset (update path as necessary)
file_path = 'moong_file.csv'
df = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%b-%Y')

# Optionally, extract year and month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month

@app.route('/')
def index():
    # Display a dropdown of commodities
    commodities = df['Commodity'].unique()
    return render_template('index.html', commodities=commodities)

@app.route('/forecast', methods=['POST'])
def forecast():
    selected_commodity = request.form['commodity']
    
    # Filter data for the selected commodity
    commodity_data = df[df['Commodity'] == selected_commodity]
    prices = commodity_data['Price']

    # Apply ARIMA model for forecasting
    model = ARIMA(prices, order=(5, 1, 0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=10)

    # Plot the forecasted prices
    plt.figure()
    plt.plot(forecast, label='Forecasted Prices')
    plt.legend()
    plt.title(f'Price Forecast for {selected_commodity}')
    plt.savefig('static/forecast.png')

    return render_template('forecast.html', commodity=selected_commodity)

if __name__ == '__main__':
    app.run(debug=True)
