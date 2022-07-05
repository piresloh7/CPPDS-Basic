
#CHART PROPERTIES
import numpy as np 
import matplotlib.pyplot as plt 
x = np.arange(0,10) 
print(x)
y = x**2 
print(y)
#Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
# Formatting the line colors
#Line plot is mainly used to visualize a trend in data over intervals of time 
plt.plot(x,y,'r')
plt.plot(x,y,'bs')
plt.plot(x,y,'g^')
# Formatting the line type  
plt.plot(x,y,'>')
plt.plot(x,y,linewidth="5.0")
#Sample
plt.figure(figsize=(15,5))#specify the size of the figure using method figure() with the values as the length of rows and columns 
plt.plot([1,2,3],[4,5,1])


#Multiple plots 
t=np.arange(0.,5.,0.2)
print(t)
plt.plot(t,t,'r--')
plt.plot(t,t,'r--',t,t**2,'bs',t,t**3,'g^')
plt.show()

#Multiple plots in one fiqure
x=np.arange(1,5)
print(x)
y=x**3
print(y)
plt.subplot(2,1,1)
plt.plot(x,y,'r^')
plt.title('1stSubplot')

plt.subplot(2,1,2)
plt.plot([1,2,3,4],[1,4,9,16],'go')
plt.title('2ndSubplot')
plt.suptitle("My Sub plot")
plt.show()


#Adding legends
a = b = np.arange(0, 3, .2)
a
c = np.exp(a)
c
d = c[::-1]#Reverse of array c
d
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, c + d, 'k', label='Total message length')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('C0')
plt.grid(True,color='r')
plt.show()

#DIFFERENT GRAPHS
#BAR GRAPH
#The x-axis represents the categories and are spaced evenly. The y-axis represents the quantity for each category 
import numpy as np
import matplotlib.pyplot as plt 
# Our data
labels = ["JavaScript", "Java", 
"Python", "C#"]
usage = [69.8, 45.3, 38.8, 34.4]
# Creating our bar plot
plt.bar(labels, usage,color="red")
plt.xlabel("Programming language")
plt.ylabel("Usage (%)")
plt.title("Programming language usage")
plt.show()


#histogram
#Histogram plot is generally used to summarize the distribution of a data sample.
#The x-axis represents discrete bins or intervals for the observations.
import pandas as pd
r=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
r.columns = attributes
r.hist(bins=12)
plt.show()




#SCATTER PLOT
#A scatter plot (or ‘scatterplot’) is generally used to summarize the relationship between two paired data samples.
#Paired data samples means that two measures were recorded for a given observation, such as the weight and height of a person.
import numpy as np
import matplotlib.pyplot as plt 
temp = [30, 32, 33, 28.5, 35, 29, 29]
ice_creams_count = [100, 115, 115, 75, 125, 79, 89]
#plt.plot(temp, ice_creams_count)
plt.scatter(temp, ice_creams_count)
plt.title("Temperature vs. Sold ice creams")
plt.xlabel("Temperature")
plt.ylabel("Sold ice creams count")
plt.show()


#Stack plot
x = [1, 2, 3, 4, 5]
y1 = [1, 1, 2, 3, 5]
y2 = [0, 4, 2, 6, 8]
y3 = [1, 3, 5, 7, 9]
labels = ["Fibonacci ", "Evens", "Odds"]
fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3, labels=labels)
ax.legend(loc='upper left')
plt.show()


#ARea Chart
x=range(1,15)
y=[1,4,6,8,4,5,3,2,4,1,5,6,8,7]
# Change the color and its transparency
plt.fill_between( x, y, color="skyblue", alpha=0.4)
plt.show()
# Same, but add a stronger line on top (edge)
plt.fill_between( x, y, color="skyblue", alpha=0.2)
plt.plot(x, y, color="Slateblue", alpha=0.6)


#Pie Chart
import numpy as np
import matplotlib.pyplot as plt 
sizes = [25, 20, 45, 10]
labels = ["Cats", "Dogs", "Tigers", "Goats"]
plt.pie(sizes, labels = labels, autopct = "%.2f")
plt.show()


#BASIC ANNOTATION
import numpy as np
x = np.arange(0,10) 
y = x ^ 2 
z = x ^ 3
t = x ^ 4 
# Labeling the Axes and Title
plt.title("Graph Drawing") 
plt.xlabel("Time") 
plt.ylabel("Distance") 
plt.plot(x,y)
#Annotate
plt.annotate(xy=[2,1], s='Second Entry') 
plt.annotate(xy=[4,6], s='Third Entry') 

#SEABORN
#Violin PLot
import pandas as pd
import seaborn as sns
sns.set(palette='Reds',style='whitegrid',font='sans-serif', font_scale=1)#Deafult setting of seaborn
r=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
r.columns = attributes
sns.violinplot(x='species',y='petal_length', data=r)
plt.show()

#HEATMAPS
# Create a dataset 
df = pd.DataFrame(np.random.random((5,5)), columns=["a","b","c","d","e"])
print(df)
# Default heatmap: just a visualization of this square matrix
p1 = sns.heatmap(df)


#BOXPLOT
sns.boxplot( y=r["sepal_length"] )
plt.show()
sns.boxplot( x=r["species"], y=r["sepal_length"] )
plt.show()

#Swarm plot
sns.swarmplot(x='species',y='petal_length', data=r)
plt.show()

#Distribution plot
x = np.random.randn(100)
ax = sns.distplot(x)


#Joint Plot
import seaborn as sns
g = sns.jointplot("sepal_width", "petal_length", data=r, color="g")
 
#Pair plot
sns.pairplot(r)


#countplot
sns.countplot(x="species",data=r)


#Strip plot
sns.stripplot(x='species',y='petal_length', data=r,jitter=0.5)
plt.show()






 
