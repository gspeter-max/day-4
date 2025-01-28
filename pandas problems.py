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
