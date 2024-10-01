import yfinance as yf  # Yahoo finances lib
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import datetime
import pandas as pd
import numpy as np
import base64
import io


from sklearn.linear_model import LinearRegression

# Stock Layout
app = dash.Dash()
app.title = "Stock and Sales Visualization"

app.layout = html.Div(children=[
    html.H1('Stock and Sales Visualization Dashboard'),
    
    html.H4('Please enter the stock name (e.g., AAPL, MSFT)'),
    dcc.Input(id="input-stock", value='AAPL', type='text'),  # Default input for testing
    html.Div(id="output-graph-stock"),
    
    html.H4('Upload a CSV file for Sales Visualization'),
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload File'),
        multiple=False
    ),
    html.Div(id='output-graph-sales')
])

# Callback for stock data visualization
@app.callback(
    Output(component_id="output-graph-stock", component_property='children'),
    [Input(component_id="input-stock", component_property="value")]
)
def update_stock_value(stock_name):
    start = datetime.datetime(2010, 1, 1)
    end = datetime.datetime.now()
    
    try:
        df = yf.download(stock_name, start=start, end=end)
        if df.empty:
            return html.Div(f"No data found for the stock symbol: {stock_name}")
        
        # Prepare data for linear regression
        df['Date'] = df.index.map(datetime.datetime.toordinal)
        X = df['Date'].values.reshape(-1, 1)
        y = df['Close'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict future prices
        future_dates = pd.date_range(start=df.index[-1] + datetime.timedelta(days=1), periods=30)
        future_dates_ordinal = future_dates.map(datetime.datetime.toordinal).values.reshape(-1, 1)
        predicted_prices = model.predict(future_dates_ordinal)
        
        return dcc.Graph(
            id="stock-graph",
            figure={
                'data': [
                    {'x': df.index, 'y': df['Close'], 'type': 'line', 'name': stock_name},
                    {'x': future_dates, 'y': predicted_prices, 'type': 'line', 'name': 'Predicted Prices', 'line': {'dash': 'dash'}}
                ],
                'layout': {'title': f'Stock Prices for {stock_name}', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Price (USD)'}}
            }
        )
    except Exception as e:
        return html.Div(f"Error fetching data: {e}")

# Callback for sales data visualization
@app.callback(
    Output('output-graph-sales', 'children'),
    [Input('upload-data', 'contents')]
)
def update_sales_graph(contents):
    if contents is None:
        return html.Div("Please upload a CSV file.")
    
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        # Load the CSV into a DataFrame
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        
        # Ensure the 'Date' column is properly parsed
        df['Date'] = pd.to_datetime(df['Date'], format='%m-%d-%y')
        
        # Aggregate the data by Date, summing up the sales Amount
        df_grouped = df.groupby('Date')['Amount'].sum().reset_index()
        
        return dcc.Graph(
            id="sales-graph",
            figure={
                'data': [{'x': df_grouped['Date'], 'y': df_grouped['Amount'], 'type': 'bar', 'name': 'Sales Amount'}],
                'layout': {'title': 'Sales Data', 'xaxis': {'title': 'Date'}, 'yaxis': {'title': 'Sales Amount'}}
            }
        )
    except Exception as e:
        return html.Div(f"Error processing the file: {e}")

if __name__ == "__main__":
    app.run_server(debug=True)
