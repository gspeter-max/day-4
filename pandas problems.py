'''
Problem 1: Advanced Data Aggregation and Transformation
You are working with a large e-commerce dataset. The dataset contains the following columns:

order_id (integer): Unique identifier for each order.
product_id (integer): Unique identifier for each product.
product_category (string): The category of the product.
order_date (datetime): The date the order was placed.
quantity_sold (integer): Quantity of the product sold.
total_sales (float): Total sales amount for the product sold.
customer_id (integer): Unique identifier for the customer.
Task:

Clean the data by handling missing values and outliers.
For each customer, calculate:
The total amount spent (sum of total_sales).
The total quantity bought (quantity_sold).
The average order value (sum of total_sales divided by number of unique order_ids).
For each product category, calculate:
The total sales across all orders.
The average sales per order.
The category with the highest total sales, and its most sold product.
Extract the month and year from order_date and calculate:
The month with the highest total sales for each customer.
The month-wise trends of sales across different product categories.
Use both Pandas for aggregations, and NumPy for efficient calculations where appropriate (e.g., when handling large numerical transformations or calculating statistics).
''' 

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)


n_orders = 10000
n_customers = 1000
n_products = 50



order_ids = np.arange(1, n_orders + 1)
product_ids = np.random.choice(np.arange(1, n_products + 1), size=n_orders)
product_categories = np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Toys', 'Books'], size=n_orders)
order_dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_orders)]

quantities = np.random.randint(1, 5, size=n_orders)
total_sales = np.round(np.random.uniform(10, 500, size=n_orders), 2)
customer_ids = np.random.choice(np.arange(1, n_customers + 1), size=n_orders)




df = pd.DataFrame({
    'order_id': order_ids,
    'product_id': product_ids,
    'product_category': product_categories,

    'order_date': order_dates,
    'quantity_sold': quantities,
    'total_sales': total_sales,
    'customer_id': customer_ids
})
    
''' have no missing values ''' 
'''outliers ''' 

from scipy.stats import zscore 

df['z_score'] = abs(zscore(df['total_sales']))
df['is_outliers'] = df['z_score'] > 3 

outliers_free = df[df['is_outliers'] == False]

''' each customer '''

outliers_free['total_amount_spent'] = outliers_free.groupby('customer_id')['total_sales'].transform('sum')

outliers_free['total_quantity_bought'] = outliers_free.groupby('customer_id')['quantity_sold'].transform('sum')

average_order_values = outliers_free.groupby('customer_id').apply(lambda x : (x['total_amount_spent'] / x['order_id'].unique().sum()).round(2))




outliers_free['total_sales_products'] = outliers_free.groupby(
    'product_category'
)['total_sales'].transform('sum')

outliers_free['avg_sales_per_order'] = outliers_free.groupby(
    ['product_category','order_id']
)['total_sales'].transform('mean')


highest_category = outliers_free.groupby('product_category')['total_sales'].agg("sum").reset_index().sort_values(
    by = 'total_sales', 
    ascending = False
).loc[0,'product_category']


most_sold_df = outliers_free[outliers_free['product_category'] == highest_category]
most_sold_product = most_sold_df.groupby('product_id')['quantity_sold'].transform('sum').sort_values(
        ascending = False
    ).reset_index()
most_sold_product.columns = ['product_id','quantity_sold']
most_sold_product = most_sold_product.loc[0,'product_id']

print(f"most hightest category : '{highest_category}' and  the product of that : '{most_sold_product}'")



outliers_free['order_date'] = pd.DataFrame(outliers_free['order_date'])
outliers_free['month'] = outliers_free['order_date'].dt.month 
outliers_free['year'] = outliers_free['order_date'].dt.year

month_df = outliers_free.groupby(['month','customer_id'])['total_sales'].agg('sum').sort_values(
	ascending = False
).reset_index().loc[0,'month']
print(f' The month with the highest total sales for each customer : {month_df}')

moth_wise_trend = outliers_free.groupby(['month','product_category'])['total_sales'].agg('mean')
print(moth_wise_trend)

''' Problem 2: Multi-level Indexing and Advanced Window Operations
You are working with a dataset of daily stock prices for multiple companies. The dataset contains the following columns:

company_id (integer): Unique identifier for each company.
date (datetime): The date for the stock price.
stock_open (float): The stock's opening price.
stock_close (float): The stock's closing price.
stock_high (float): The stock's highest price for the day.
stock_low (float): The stock's lowest price for the day.
volume (integer): The number of shares traded.
Task:

Convert the date column to a datetime format and set it as the index.
Create a multi-level index using company_id and date.
Calculate the moving average of the stock_close price for each company, using a rolling window of 7 days.
Calculate the percentage change in stock price (stock_close) for each company, and calculate the cumulative return for each company.
Find the company with the highest average trading volume across all dates, and determine the range (difference between max and min) of closing prices for that company.
Implement resampling of data to monthly frequency and compute the monthly average stock price for each company.
Find the correlation between daily stock closing prices for the top 5 companies with the highest total trading volume in the last 3 months.  '''






import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

n_companies = 20
n_days = 500

company_ids = np.arange(1, n_companies + 1)
dates = [datetime(2020, 1, 1) + timedelta(days=i) for i in range(n_days)]

data = []
for company_id in company_ids:
    for date in dates:
        stock_open = np.round(np.random.uniform(100, 1000), 2)
        stock_close = stock_open + np.random.uniform(-5, 5)
        stock_high = max(stock_open, stock_close) + np.random.uniform(0, 5)
        stock_low = min(stock_open, stock_close) - np.random.uniform(0, 5)
        volume = np.random.randint(1000, 100000)
        data.append([company_id, date, stock_open, stock_close, stock_high, stock_low, volume])

df = pd.DataFrame(data, columns=['company_id', 'date', 'stock_open', 'stock_close', 'stock_high', 'stock_low', 'volume'])

df['date'] = pd.to_datetime(df['date'])
df = df.set_index(['company_id', 'date'])

moving_average = df.groupby(level='company_id')['stock_close'].rolling(window=7).mean()

percentage_change = df.groupby('company_id')['stock_close'].pct_change()

cumulative_return = df.groupby('company_id').apply(lambda x: (x['stock_open'] - x['stock_close']) / x['stock_open'])

high_trading_volumes = df.groupby(['company_id', 'date'])['volume'].mean().idxmax()
company_id, date = high_trading_volumes
df_temp = df.xs(key=company_id, level='company_id')

ranges = (df_temp['stock_close'].max() - df_temp['stock_close'].min())

df = df.reset_index()
df = df.set_index('date')
resample_monthly = df.resample(rule='m').apply(lambda x: x.groupby('company_id')['stock_close'].agg('mean'))

df = df.reset_index()

dates = (df['date'].max() - timedelta(days=90))
df_3_month = df[df['date'] >= dates]

total_volumes = df_3_month.groupby('company_id')['volume'].sum().reset_index()
top_5 = total_volumes.nlargest(columns='volume', n=5)['company_id'].tolist()

top_5_data = df[df['company_id'].isin(top_5)]

top_5_pivot = top_5_data.pivot(index='date', columns='company_id', values='stock_close')

correlation = top_5_pivot.corr()

print(correlation)

