import numpy as np
a=([1,2,3],[4,5,6])
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
print(a1.dtype)
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
arr=n.array(x)
print(arr)
print(arr.shape)
print(arr.size)
print(n.amax(arr))
print(n.amin(arr))
print(arr.ndim)
print(arr.itemsize)
print(arr.dtype)
s=n.ones((3,3))
print(s)
y=n.zeros((2,3,4))
print(y)
v=n.ones((2,3,4))
print(v)
f=n.arange(0,999,3)
print(f)


#SAMPLE:
import numpy as n
#v=n.ones((3,3))
v=n.array([1,2,3])
v2=n.array([1,1,1])
C=v*v2#Not matrix multiplication
print(C)
m=v.dot(v2)#Matrix multiplication
print(m)

#SAMPLE 2
import numpy as n
v=n.array([1,0,1])
v2=n.array([1,1,1])
print(n.logical_or(v,v2))

#Exercise 4:
import numpy as n
a=n.array([0,0,0,0,1])
b=n.array([1,1,1,1,0])
print(n.logical_and(a,b))
print(n.logical_not(a==0))
print(n.logical_not(b==0))

a=np.arange(12).reshape(3,4)
print(a)
b1=np.array([False,True,True])
b2=np.array([True,False,True,False])
z=a[b1,:]
z1=a[:,b2]
print(a[b1,b2])

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

from numpy import array
var = np.array( [1, 3, 5] )
print('var = ' , var)
s = var.sum()
print('Sum result = ' , s)

x = np.array([[1, 1], [2, 2]])
x.sum(axis=0)   # columns (first dimension)
x[:, 0].sum(), x[:, 1].sum()
x.sum(axis=1)   # rows (second dimension)
x[0, :].sum(), x[1, :].sum()


x = np.array([1, 3, 2])
x.min()
x.max()
x.argmin()  
x.argmax() 


x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
x.mean()

np.median(x)

np.median(y, axis=-1)       # last axis

x.std()          
#Exercise 8
import numpy as np
#a=np.random.randint(size=(100,100))
#a1=np.array(a)
#print(a1)

a=np.random.randint(low=20,high=50,size=(100,100))#make all elements bigger than 20 and smaller than 50 equal to -1
print(a)
a1=a[1:4,1:5]#Exract a sub matrix
print(a1)


#Exercise 9
arr=np.arange(8)
print(arr)
arr1=arr.reshape(2,2,2)
print(arr1)
print(arr1.T)
print(arr1.ravel())
print(arr1.astype(float))
#Deep Copy
import numpy as np
a=np.arange(12)
print('Before')
print(a)
b=a
b[0]=500
print('After')
print(a)
print(b)


#Shallow Copy
import numpy as np
a=np.arange(12)
print('Before')
print(a)
b=a.copy()
b[0]=500
print('After')
print(a)
print(b)



