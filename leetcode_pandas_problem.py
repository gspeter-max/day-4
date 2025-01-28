''' 1327. List the Products Ordered in a Period ''' 

import pandas as pd

def list_products(products: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
     
    merge = pd.merge(
     	orders, 
     	products,
     	on = 'product_id', 
     	how = 'left'
    )
    merge = merge[(merge['order_date'] >= '2020-02-01') & (merge['order_date'] <= '2020-02-29')]
    merge['unit'] = merge.groupby('product_id')['unit'].transform('sum')
    
    return merge[merge['unit'] >= 100][['product_name','unit']].drop_duplicates()

''' 
1341. Movie Rating  '''

import pandas as pd

def movie_rating(movies: pd.DataFrame, users: pd.DataFrame, movie_rating: pd.DataFrame) -> pd.DataFrame:

    merge_temp = pd.merge(movie_rating, users, on='user_id', how='left')
    merge = pd.merge(merge_temp, movies, on='movie_id', how='left')
    
   
    merge['movies_rated'] = merge.groupby('user_id')['movie_id'].transform('count')
    top_user = merge.sort_values(by=['movies_rated', 'name'], ascending=[False, True]).iloc[0]['name']
    
   
    feb_ratings = merge[(merge['created_at'] >= '2020-02-01') & (merge['created_at'] <= '2020-02-29')]
    if not feb_ratings.empty:
       
        feb_ratings['avg_rating'] = feb_ratings.groupby('movie_id')['rating'].transform('mean')
        top_movie = feb_ratings.sort_values(by=['avg_rating', 'title'], ascending=[False, True]).iloc[0]['title']
    else:
        top_movie = None  
    results = pd.DataFrame({'results': [top_user, top_movie]})
    return results



'''  1378. Replace Employee ID With The Unique Identifier '''


import pandas as pd

def replace_employee_id(employees: pd.DataFrame, employee_uni: pd.DataFrame) -> pd.DataFrame:
    merge = pd.merge(
    	employees, 
    	employee_uni, 
    	on = 'id', 
    	how = 'left'
   )
    return merge[['unique_id','name']]
