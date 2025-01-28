''' 1327. List the Products Ordered in a Period ''' 
)
with temp_temp as (
	
	select  product_id, sum( unit) as unit 
	from orders 
	where order_date between '2020-02-01' and '2020-02-29'
	group by product_id 


		)
select e.product_name , t.unit 
from temp_temp t 
left join products e on e.product_id = t.product_id 
where t.unit >= 100 ; 



''' 
1341. Movie Rating  '''
# Write your MySQL query statement below


with main_table as (
	select 
        m.movie_id, 
        u.user_id, 
        u.name, 
        mo.title, 
        m.rating, 
        m.created_at
	from movierating m 
	left join users u on m.user_id = u.user_id
	left join movies mo on mo.movie_id = m.movie_id
	) ,


temp_temp  as (
	select name ,count( distinct movie_id) as  movie_count 
	from main_table 
	group by  name                                           
	order by movie_count desc,
	name asc 	
	limit 1 	
), 


rating_movies as (
	select title, 
	avg(rating) as avg_rating
	from main_table 
	where created_at between  '2020-02-01' and '2020-02-29'
	group by title 
	order by avg_rating desc,
	title asc 
	limit 1 
) 


select name as results 
from temp_temp 

union all 

select title as results 
from rating_movies ;  


'''  1378. Replace Employee ID With The Unique Identifier '''


select e2.unique_id , e.name
from Employees  e 
left join EmployeeUNI  e2  on e.id = e2.id ; 


