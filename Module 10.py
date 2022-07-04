import numpy as np
a=([[1,2,3],[4,5,6]])
print(a)
type(a)
b=np.array(a)
print(b)
print(b.shape)
print(b.ndim)
print(b.itemsize)
print(b.size)
print(type(b))
print(b.dtype)
#Exercise 1
import numpy as np
x=np.array([1,2,3,4,5,6,7,8,9,10])
print(x)
a1=x.astype(int)
print(a1)
print(a1.dtype)#type(a1)
a2=x.astype(float)
print(a2)
print(a2.dtype)

#Exercise 2
import numpy as np
x=(range(4),range(10,14))
arr=np.array(x)
print(arr)
print(arr.shape)
print(arr.size)
print(np.amax(arr))
print(np.amin(arr))

#Exercise 3:
import numpy as n
x=([1,2,3],[4,5,6],[7,8,9])
arr=np.array(x)
print(arr)
print(arr.shape)
print(arr.size)
print(np.amax(arr))
print(np.amin(arr))
print(arr.ndim)
print(arr.itemsize)
print(arr.dtype)
s=np.ones((3,3))
print(s)
y=np.zeros((2,3,4))
print(y)
v=np.ones((2,3,4))
print(v)
f=np.arange(0,999,3)
print(f)
g=np.linspace(0,2,9)
print(g)


#Element Wise OPeration
#Sum
a=n.array([20,30,40,50])
b=n.arange(4)
c=a+b
print(c)

#Difference
a=n.array([20,30,40,50])
b=n.arange(4)
c=a-b
print(c)

#Square of elements

b=n.arange(4)
c=b**2
print(c)
#INcrement Operators
a=n.array([20,30,40,50])
b=n.arange(4)
a+=b
print(a)
a*=b
print(a)

#SAMPLE:
import numpy as n
#v=n.ones((3,3))
v=n.array([1,2,3])
v2=n.array([1,1,1])
C=v*v2#Not matrix multiplication
print(C)
m=v.dot(v2)#Matrix multiplication
print(m)

#Comparison Operator
a=n.array([20,30,40,50])
b=n.arange(4)
c=n.array([0,1,2,4])
print(a>b)
print(b==c)

#SAMPLE 2
import numpy as n
v=n.array([1,0,1])
v2=n.array([1,1,1])
print(n.logical_or(v,v2))

#Exercise 4:
import numpy as np
a=np.array([0,0,0,0,1])
b=np.array([1,1,1,1,0])
print(np.logical_and(a,b))
print(np.logical_not(a==0))
print(np.logical_not(b==0))


#Broadcasting
#Broadcasting Different Rank Different shapes
a=np.array([1,2,3])
np.rank(a)
a.shape
b=np.array([[3],[4],[5]])
np.rank(b)
b.shape
c=a+b
print(c)

#Sum by rows and by columns

x=np.array([[1,1],[2,5]])
print(x)
x.sum(axis=0)#By columns
x[:,0].sum()#All rows and 1st coloumn
x[:,1].sum()#All rows and 2nd column
x.sum(axis=1)#By rows
x[0,:].sum()#All columns and 1st row
x[1,:].sum()#All columns and 2nd row


#Extrema
x=np.array([1,3,2])
x.min()
x.max()
x.argmin()
x.argmax()

#Statistics
x=np.array([1,2,3,1])
y=np.array([[1,2,3],[5,6,1]])
x.mean()
np.median(x)
np.median(y)
np.median(y,axis=0)
np.median(y,axis=1)
x.std()

# Logical Operaions
np.all([True,True,False])
np.any([True,True,False])

#Indexing with Boolean Arrays
a=np.arange(12).reshape(3,4)
print(a)
b1=np.array([False,True,True])
b2=np.array([True,False,True,False])
z=a[b1,:]
z1=a[:,b2]
print(a[b1,b2])

#UNiversal Functions
a=np.zeros((100,100))
print(a)
np.any(a!=0)
np.all(a==a)
a=np.array([1,2,3,2])
b=np.array([2,2,3,2])
c=np.array([6,4,4,5])
a<=b
b<=c
((a<=b)&(b<=c)).all()




#Exercise 5:
import numpy as np
arr=np.array([0,1,2,3,4,5,6,7,8,9])
x=arr<3
print(x)#prints the boolean values in the array that satisy the condition
y=arr[arr<3]
print(y)#prints the values in the array that satisy the condition
z=((arr < 3) | (arr > 8))
print(z)
print(arr[z])
result = np.where(((arr < 3) | (arr > 8)),arr*5,arr*-5)
print(result)


#Exercise 6:
import numpy as np
a=np.array([range(4),range(10,14)])
b=np.array([2,-1,1,0])
c=a*b
print(c)
b1=b*100
b2=b*100.0
print(b1)
print(b2)
print(b1==b2)
print(b1.dtype)
print(b2.dtype)

#One Dimensional Slicing
data=np.array([11,22,33,44,55])
print(data[:])
print(data[0:1])
print(data[-2:])

a=np.arange(10)
print(a)
slicea=a[2:9:3]
print(slicea)

x=np.arange(10)
x
x[:5]
x[5:]
x[4:7]  
x[::2]
x[1::2]
x[::-1]
x[5::-2]



#MultiDimensional Slicing
x2=np.random.randint(10,size=(3,4))
print(x2)
x2[:2,:3]#prints 2 rows and 3 columns
x2[:3,::2]
x2[::-1,::-1]#Reverse the order of rows


#Exercise6
a=np.array([2,3.2,5.5,-6.4,-2.2,2.4])
a[1]
a[1:4]
a1=np.array([[2,3.2,5.5,-6.4,-2.2,2.4],[1,22,4,0.1,5.3,-9],[3,1,2.1,21,1.1,-2]])
a1
a1[:,3]
a1[1:4,0:4]
a1[1:,2]

#Exercise 8
a=np.random.randint(low=10,high=60,size=(100,100))#make all elements bigger than 20 and smaller than 50 equal to -1
res1=np.where(((a>20)&(a<50)),-1,a)
print(res1)
a1=a[1:4,1:5]#Exract a sub matrix
print(a1)

#Stacking together
a=np.array((1,2,3))
b=np.array((2,3,4))
c=np.hstack([a,b])
print(a)
b
c
a=np.array((1,2,3))
b=np.array((2,3,4))
c=np.vstack((a,b))
print(a)
b
c

#Split into smaller ones
x=np.array([[1,2,3,4],
            [4,5,6,7],
            [2,3,4,5],
            [3,5,6,6]])
y=np.hsplit(x,2)
y
z=np.vsplit(x,2)
z
a6=np.array([[1.0,2.0],[3.0,4.0]])
a6
print(np.linalg.inv(a6))
a7=np.array([[5.],[7.]])
print(np.linalg.solve(a6,a7))
u=np.eye(3)
u



#Exercise 9
arr=np.arange(8)
print(arr)
arr1=arr.reshape(2,2,2)
print(arr1)
print(arr1.T)
print(arr1.ravel())
print(arr1.astype(float))



#Shallow Copy(no new objects are created)
import numpy as np
a=np.arange(12)
print('Before')
print(a)
b=a
b[0]=500
print('After')
print(a)
print(b)


#Deep Copy(new object is created)
import numpy as np
a=np.arange(12)
print('Before')
print(a)
b=a.copy()
b[0]=500
print('After')
print(a)
print(b)



#PANDAS
#Create Series from the list
import pandas as pd
print(pd.Series([1,3,5,6]))
print(pd.Series([1,3,5,6],index=['A1','A2','A3','A4']))
#Create series from np as array
a=np.random.randn(100)*5+100
date=pd.date_range('20170101',periods=100)
s=pd.Series(a,index=date)
print(s)
#Create Series from Dictionary
a2={'A1':5,'A2':3,'A3':6,'A4':2}
s1=pd.Series(a2)
print(s1)

#Series Arithmetic
a3=pd.Series([1,2,3,4],index=['a','b','c','d'])
a4=pd.Series([4,3,2,1],index=['d','c','b','a'])
print(a3+a4)
print(a3-a4)
print(a3*a4)
print(a3/a4)


#Series Attributes
print(a3.index)
print(a3.values)
print(len(a3))



#viewing Series Data
print(s.head())#return first 5 rows
print(s.head(10))# return first 10 rows
print(s.tail())#return last 5 rows
print(s.tail(3))#return last 3 rows




#Selecting Data
print(a3['b'])
print(a3[2])
print(a3[['b','d']])#multiple select by label
print(a3[[1,3]])#multiple select by index integer

#Slicing Series Data
print(a3[1:4])#Slice items 1to 4
print(a3[:4])
print(a3[2:])







#Exercise 9
date=pd.date_range('20170101',periods=20)
s=pd.Series(np.random.randn(20),index=date)
print(s)
print(s['2017-01-05':'2017-01-10'])

# Create Data Frame from the list
d=[[1,2],[3,4]]
df=pd.DataFrame(d,index=[1,2],columns=['e','f'])
print(df)
#Create DF from Numpy Array
d=np.arange(24).reshape(6,4)
df=pd.DataFrame(d,index=np.arange(1,7),columns=list('ABCD'))
print(df)
#Create DF from Dictionary
print(pd.DataFrame({'name':['Ally','Jane','Belinda'],'height':[160,155,163]},columns=['name','height'],index=['A1','A2','A3']))
#Create DataFrame from np as array
date=pd.date_range('20170101',periods=6)
s=pd.Series(np.random.randn(6),index=date)
s1=pd.Series(np.random.randn(6),index=date)
df=pd.DataFrame({'Asia':s,'Europe':s1})
print(df)



date=pd.date_range(start='2017-01-01',end='2019-01-01',freq='M')#Print the date within the range month frequency
date


#Exercise
v=pd.DataFrame({'name':['Ally','Jane','Belinda'],'height':[160,155,163],'age':[40,35,42]},columns=['name','height','age'],index=['A1','A2','A3'])
print(v)
print(v.shape)
print(v.columns)
print(v.index)
print(v.values)





#Append a new column
v['weight']=[45,56,44]
print(v)
#or
v.insert(0,'id',[1,2,3])
print(v)

#Dataframe Arithmetic
a3=[[3,4],[5,6]]
a4=[[6,5],[4,3]]
b3=pd.DataFrame(a3,index=[1,2],columns=['d','b'])
b4=pd.DataFrame(a4,index=[3,2],columns=['c','b'])
print(b3)
print(b4)
print(b3+b4)

#Viewing Data
d=pd.DataFrame(np.random.randn(20,5))
d
print(d.head())
print(d.head(8))
print(d.tail())
print(d.tail(8))

#Selecting Data
v=pd.DataFrame({'name':['Ally','Jane','Belinda'],'height':[160,155,163],'age':[40,35,42]},columns=['name','height','age'],index=['A1','A2','A3'])
v
#Selecting by column names
v.name
v[['name','height']]
#Selecting by row
print(v.iloc[2])#determine by position
print(v.loc['A1'])#determine by key label
#Slicing Seriees Data
print(v[0:1])#Slicing here only 0th item will be shown
print(v[:2])#here values until previous position to 2 will be displayed
print(v[2:])#values including 2 until last position-1 will be displayed
print(v.loc['A1']['height'])#Extract only with label A1 and column value height
print(v.iloc[2]['height'])#Extract value in position 2 at the column height

# 

#CSV

r=pd.read_csv("C:/Users/arun/Desktop/iris.csv")
print(r)
print(r.head())
print(r.columns)
print(r.sepal_length[1:21])#Print only sepal length for first 20 values

#Excel
import pandas as pd
df = pd.read_excel("C:/Users/arun/Desktop/itrain python/iris.xls")
print(df.head(10))


#Filtering series from np as array
a=np.random.randn(10)*4+5
date=pd.date_range('20170101',periods=10)
s=pd.Series(a,index=date)
s
print(s>5)#Print the boolean results
print(s[s>5])#print the value results


# Filtering Dataframe
r=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
r.columns = attributes
print(r.head())
print(r.columns)
print(r.sepal_length[(r.species=='Iris-virginica')& (r.sepal_length >5.0)])#Prints the sepal length values for the species Iris Virginica and of sepal length >5.0

#HANDLING MISSING VALUES
#Discover missing values
missing=np.nan
s=pd.Series([3,4,missing,6,missing,8])
s
s.isnull()
 
mydata=pd.read_csv("C:/Users/arun/Desktop/itrain python/python/Titanic.csv")
print(mydata.head(5))
print(mydata.info())
# To check  Any missing values?
print(mydata.isnull().values.any())
#Counting missing data
mydata.isnull().sum()
#Filling the missing values with 0
z=mydata.fillna(0)
z.isnull().sum()#Counting missing data
#Fill the missing values with the previous value
print(mydata.head(5))
y=mydata.fillna(method='ffill')
print(y.head(5))
m=mydata.fillna(method='bfill')#Fill the missing values with the next value
print(m.head(5))
#Remove na rows 
s=mydata.dropna()#Remove only the rows
print(s.head(5))
#Remove  columns having NA values 
f=mydata.dropna(axis=1)
print(f.head(5))
mydata.dropna(how='all')#Identify the rows that conatin all missing values 

# Duplicates
#Check for duplicates
print(mydata.duplicated())
#If duplicates present we can drop the duplicates by
c=mydata.drop_duplicates()
print(c)


#CONCAT Series
s1=pd.Series(['a','b'])
s2=pd.Series(['c','d'])
s3=pd.concat([s1,s2])
s3=pd.concat([s1,s2],keys=['s1','s2'])
print(s3)

#CONCAT DATAFRAME
data1 = {
        'id': ['1', '2', '3', '4', '5'],
        'Feature1': ['A', 'C', 'E', 'G', 'I'],
        'Feature2': ['B', 'D', 'F', 'H', 'J']}
dataf1 = pd.DataFrame(data1, columns = ['id', 'Feature1', 'Feature2'])
print(dataf1)
data2 = {
        'id': ['1', '2', '6', '7', '8'],
        'Feature1': ['K', 'M', 'O', 'Q', 'S'],
        'Feature2': ['L', 'N', 'P', 'R', 'T']}
dataf2 = pd.DataFrame(data2, columns = ['id', 'Feature1', 'Feature2'])
print(dataf2)

dataconcat=pd.concat([dataf1,dataf2])
print(dataconcat)#just concats as it is 0,1,2,3,4,0,1,2,3,4
dataconcat=pd.concat([dataf1,dataf2],ignore_index=True)
print(dataconcat)#concats by ignoring the index 0,1,2,3,4,5,6,7,8,9
dataconcat=pd.concat([dataf1,dataf2],keys=['x','y'])#concatthe data frame with unique keys for each dataframe
print(dataconcat)

print(dataconcat.loc['y'])
dconcolumn=pd.concat([dataf1,dataf2],axis=1)#concatenate DataFrames along column
print(dconcolumn)



#join
dajoin1=pd.merge(dataf1,dataf2,on='id',how='inner')
print(dajoin1)#prints only the rows that has common id in both data frames
dajoin2=pd.merge(dataf1,dataf2,on='id',how='outer')
print(dajoin2)#just merge two dataframes replacing Nan for the id that is not in the dataframe


#Sorting in dataframe
import numpy as np
d=np.random.randn(24).reshape(12,2)
print(d)
df=pd.DataFrame(d,columns=['a','b'])
print(df)
df1=df.sort_values(by=['a'])
print(df1)

#GROUP BY
ipl_data = {'Team': ['Riders', 'Riders', 'Devils', 'Royals', 'Kings'],
   'Rank': [1, 2, 2, 3, 3],
   'Year': [2014,2015,2016,2015,2017],
   'Points':[876,789,863,673,741]}
df = pd.DataFrame(ipl_data)
print(df)
print(df.describe())
df1=df.groupby('Team')
print(df1.groups)
print(df1.max())
df2=df.groupby('Year')
print(df2['Points'].agg(np.mean))

#Covariance is a measure of relationship between 2 variables. 
#It measures the degree of change in the variables, i.e. when one variable changes, will there be the same/a similar change in the other variable. 
s=df[['Points','Rank']].cov()
print(s)
#Correlation (-1 to 1)
v=df[['Points','Rank']].corr()
print(v)