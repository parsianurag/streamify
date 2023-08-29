#!/usr/bin/env python
# coding: utf-8

# # Numpy
# 

# In[1]:


import numpy as np


# In[2]:


l=[1,2,3,4,5]


# In[6]:


#convert to array
arr=np.array(l)


# In[7]:


type(arr)


# In[8]:


np.asarray(l)


# In[12]:


arr1=np.array([[1,2,3],[2,3,4]])


# In[13]:


arr.ndim


# In[14]:


arr1.ndim


# In[16]:


mat=np.matrix(l)


# In[17]:


mat


# In[18]:


a=arr


# In[19]:


a


# In[20]:


arr[0]


# In[21]:


arr[0]=100


# In[22]:


a


# In[23]:


b=np.copy(arr)


# In[24]:


b


# In[25]:


b[0]=234


# In[26]:


b


# In[27]:


arr


# In[29]:


list(i*i for i in  range(5)) 


# In[31]:


np.fromstring('23 56 76',sep=' ')


# In[32]:


arr.size


# In[33]:


arr1.size


# In[34]:


arr.shape


# In[35]:


import numpy as np


# In[36]:


list(range(5))


# In[37]:


np.arange(.4,10.4,0.2)


# In[38]:


np.zeros((3,4))


# In[39]:


np.zeros((3,4,2)) #dimention,row,column


# In[40]:


np.ones(5)


# In[41]:


np.ones((3,4))


# In[42]:


np.random.rand(2,3)


# In[44]:


arr2=np.random.randint(1,5,(3,4))


# In[45]:


arr2


# In[46]:


arr2.size


# In[48]:


arr2.reshape(4,3)


# In[49]:


arr2>2


# In[51]:


arr1[0]


# In[52]:


arr2


# In[53]:


arr2[2:4,[2,3]]


# In[54]:


arr1@arr2 #matrix multiplication


# In[58]:


arr3=np.random.randint(1,10,(4,4))


# In[59]:


arr3


# In[60]:


arr3.T


# In[61]:


np.repeat(data)


# In[70]:


data=np.random.randint(3,6,(1,4))


# In[71]:


data


# In[72]:


np.repeat(data,4) #it repeat 4 times


# In[73]:


np.diag(np.array([1,2,3,4])) #diagonal matrix


# In[74]:


arr1=np.random.randint(1,10,(3,4))
arr2=np.random.randint(1,10,(3,4))


# In[75]:


arr1


# In[76]:


arr2


# In[77]:


arr1>arr2


# # string operion in array

# In[78]:


arr=np.array(['sudh','kumar'])


# In[79]:


arr


# In[80]:


#convert to upper case
np.char.upper(arr)


# In[81]:


np.char.capitalize(arr)


# In[82]:


arr1


# In[83]:


np.sin(arr1)


# In[84]:


np.tan(arr1)


# In[85]:


np.exp(arr1) #exponentional


# In[86]:


np.mean(arr1)


# In[87]:


np.median(arr1)


# In[89]:


np.std(arr1) #standard division


# In[90]:


np.var(arr1)


# In[91]:


np.max(arr1)


# In[92]:


np.min(arr1)


# In[93]:


np.multiply(arr1,arr2)


# In[95]:


np.subtract(arr1,arr2)


# In[97]:


np.mod(arr1,arr2)


# In[108]:


arr=np.array([2,3,0,9,5,0,6])


# In[109]:


np.sort(arr)


# In[110]:


np.count_nonzero(arr)


# In[111]:


np.where(arr>0)


# In[112]:


np.extract(arr>2,arr1)


# # matplotlib

# In[141]:


import matplotlib.pyplot as plt


# In[142]:


import numpy as np


# In[115]:


x=np.random.rand(50)
y=np.random.rand(50)


# In[116]:


x


# In[117]:


y


# In[118]:


plt.scatter(x,y) #scatter map


# In[125]:


plt.figure(figsize=(6,4))
plt.scatter(x,y,c='r')
plt.xlabel("this is x axis")
plt.ylabel("this is y axis")
plt.title("this is x vs y")
plt.grid()


# In[126]:


plt.plot(x,y)


# In[128]:


x=np.linspace(1,10,100)
y=np.sin(x)


# In[129]:


x


# In[130]:


y


# In[131]:


plt.plot(x,y)


# In[132]:


x=['a','b','c','d','e']
y=np.random.rand(5)


# In[133]:


x


# In[134]:


y


# In[136]:


plt.bar(x,y)
plt.xlabel("representing my catagorical")
plt.ylabel("representing my num vakues")
plt.title("bar plot")
plt.figure(figsize=(5,3))


# In[137]:


data=[1,1,1,5,5,8,7,8,8,9,9,0,4,2,3]


# In[138]:


plt.hist(data) #it shows the frequency 1 repeated 3times


# In[145]:


x=np.random.rand(50)
y=np.random.rand(50)
z=np.random.rand(50)
fig=plt.figure()
ax=fig.add_subplot(projection='3d')
ax.scatter(x,y,z)
plt.show()


# # seaborn
# 

# In[1]:


import seaborn as sns


# In[2]:


iris=sns.load_dataset('iris')


# In[3]:


iris


# In[5]:


sns.scatterplot(x=iris.sepal_length,y=iris.sepal_width)


# In[7]:


tips=sns.load_dataset('tips')


# In[8]:


tips


# In[8]:


sns.scatterplot(x=tips.total_bill,y=tips.tip)


# In[9]:


tips.head()


# In[10]:


tips['smoker'].value_counts()


# In[13]:


sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,hue="smoker")


# In[10]:


#how are smoker and not smoker
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,style="smoker")


# In[11]:


#
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,style="size")


# In[12]:


#rhe person came for lunch or dinner
sns.relplot(x=tips.total_bill,y=tips.tip,data=tips,style="size",hue='time')


# In[13]:


#how many people are coming to restarent daily
sns.catplot(x='day',y='total_bill',data=tips)


# In[14]:


sns.jointplot(x=tips.total_bill,y=tips.tip)


# In[15]:


sns.pairplot(tips)


# # Measure Of Central Tendency

# In[1]:


age=[12,13,14,15,21,24]


# In[2]:


(12+13+14+15+21+24)/6


# In[3]:


import numpy as np


# In[4]:


#mean
np.mean(age)


# In[5]:


#median
np.median(age)


# In[8]:


#mode
from scipy import stats


# In[9]:


stats.mode(age)


# 
# # Measure of Dispersion

# In[10]:


ages_lst=[23,24,34,34,23,25,65,75,32]


# In[12]:


import numpy as np


# In[14]:


mean=np.mean(ages_lst)


# In[15]:


mean


# In[16]:


var=np.var(ages_lst)


# In[17]:


var


# In[18]:


std=np.std(ages_lst)


# In[19]:


std


# In[20]:


data=[[10,12,13],[34,23,65],[32,33,21]]


# In[21]:


data


# In[23]:


import pandas as pd


# In[26]:


df=pd.DataFrame(data,columns=["A","B","C"])


# In[27]:


df


# In[29]:


#Row wise
df.var(axis=1)


# In[30]:


#column wise
df.var(axis=0)


# In[31]:


import seaborn as sns


# In[34]:


df=sns.load_dataset('healthexp')
df.head()


# In[35]:


import numpy as np


# In[36]:


df.cov() #covarience


# In[37]:


#correlation
df.corr(method='spearman')


# In[38]:


df=sns.load_dataset('penguins')
df.head()


# In[39]:


df.corr()


# # Check Normal Distubution using QQ plot-Quantile-Quantile plot
# 

# In[40]:


import scipy.stats as stat
import pylab
import numpy as np


# In[41]:


import seaborn as sns


# In[42]:


df=sns.load_dataset("iris")
df.head()


# In[47]:


import matplotlib.pyplot as plt


# In[53]:


def plot_data(df,feature):
    plt.figure(figsize=(10,6))
    plt.subplot(1,2,1)
    sns.histplot(df[feature],kde=True)
    plt.subplot(1,2,2)
    stat.probplot(df[feature],dist='norm',plot=pylab)
    plt.show()


# In[54]:


plot_data(df,'sepal_length')


# In[55]:


plot_data(df,'sepal_width')


# In[56]:


plot_data(df,'petal_length')


# In[ ]:


Chi square test


# In[1]:


import scipy.stats as stat
import numpy as np


# In[2]:


#No of hours student study daily in a weekliy basis
#mond,tus,wend,thus,friday,saturday,sunday
expected_data=[8,6,7,9,6,9,7]
observed_data=[7,8,6,9,9,6,7]


# In[4]:


sum(expected_data),sum(observed_data)


# In[5]:


#chi square goodness of fit
chisquare_test_statistics,p_value=stat.chisquare(observed_data,expected_data)


# In[6]:


print(chisquare_test_statistics),print(p_value)


# In[8]:


#find the critical value
significance=0.05
dof=len(observed_data)-1
critical_value=stat.chi2.ppf(1-significance,dof)


# In[9]:


critical_value


# In[10]:


if chisquare_test_statistics > critical_value:
    print("reject the null hypothesis")
else:
    print("accept the null hypothesis")


# # Missing values
# 

# In[1]:


import seaborn as sns


# In[2]:


df=sns.load_dataset('titanic')


# In[3]:


df.head()


# In[4]:


#check missing values
df.isnull()


# In[5]:


df.isnull().sum() #in which column how many null values are there


# In[6]:


#delete the rows or data points to handle missing values
df.shape


# In[ ]:





# In[7]:


df.dropna().shape #it drop all nan values


# In[8]:


##columns wise delete
df.dropna(axis=1)


# # imputation missing values
# 1-Mean Value Imputation

# In[9]:


sns.histplot(df['age'],kde=True)


# In[10]:


df["age_mean"]=df['age'].fillna(df['age'].mean()) #it replace all the NSAN values with mean of the age


# In[ ]:


df["age_mean"]=df['age'].fillna(df['age'].mean())


# In[11]:


df['age_mean']=df['age'].fillna(df['age'].mean())


# In[13]:


df['age_median']=df['age'].fillna(df['age'].median())


# In[15]:


df["age_median"]


# In[12]:


df[["age_mean",'age']]


# # 2. Median value Imputation.If we have outliers in the dataset

# In[18]:


df['age_median']=df['age'].fillna(df['age'].median())


# In[ ]:


df['age_median']=df['age'].fillna(df['age'].median())


# In[ ]:





# In[20]:


df[['age_median','age']]


# ### Mode  Imputation Technique--Categorical values

# In[21]:


df[df['embarked'].isnull()]


# In[22]:


df['embarked'].unique()


# In[ ]:


mode_values=df[df['embarked'].notna()]['embarked'].mode()[0]


# In[ ]:


df['median_age']=df['age'].fillna(df['age'].median())


# In[16]:


df['embarked'].unique()


# In[ ]:


df[df['embarked'].notna()]['embarked'].mode()[0]


# In[28]:


mode_value=df[df['embarked'].notna()]['embarked'].mode()[0]


# In[ ]:


df[df['embarked'].notna()]#it don't gives NAN values so we can do mofe
['embarked'].mode()[0] #this is for doing mode 


# In[29]:


df["embarked_mode"]=df['embarked'].fillna(mode_value)


# In[30]:


df[['embarked_mode','embarked']]


# In[31]:


df['embarked_mode'].isnull().sum()


# ### Handling Imbalanced Dataset
# ##1.Up Sampling
# ##2.Down Sampling
# 

# In[34]:


import numpy as np
import pandas as pd
#set the random seed for reproducinility
np.random.seed(123)
#create a datafram with two classes
n_samples=1000
class_0_ratio=0.9
n_class_0=int(n_samples * class_0_ratio)
n_class_1=n_samples - n_class_0


# In[35]:


n_class_0,n_class_1


# # linear interpolation
# 

# In[1]:


import numpy as np
x=np.array([1,2,3,4,5,])
y=np.array([2,4,6,8,10])


# In[2]:


import matplotlib.pyplot as plt
plt.scatter(x,y)


# In[3]:


#interpollate the data using liner interpolation
x_new=np.linspace(1,5,10) #create new x values 1 to 5 ,in between 10 numbers
y_interp=np.interp(x_new,x,y)


# In[4]:


plt.scatter(x_new,y_interp)


# # 2.Cubic Interpolation with Scipy

# In[2]:


import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([1,8,27,64,125])


# In[3]:


from scipy.interpolate import interp1d


# In[4]:


#create a cubic interpolation function
f=interp1d(x,y,kind='cubic')


# In[5]:


#interpolate the data
x_new=np.linspace(1,5,10)
y_interp=f(x_new)


# In[6]:


plt.scatter(x,y)


# In[12]:


plt.scatter(x_new,y_interp)


# # 3.Polynomial Interpolation

# In[13]:


#create some ample data
x=np.array([1,2,3,4,5])
y=np.array([1,4,9,16,25])


# In[14]:


#interpolate the data using polynomial interpolation
p=np.polyfit(x,y,2)


# In[15]:


x_new=np.linspace(1,5,10) #create new x values
y_interp=np.polyval(p,x_new) #interpolate y valies


# In[16]:


plt.scatter(x_new,y_interp)


# # Covariance and coorrelation With Python

# In[1]:


import seaborn as sns


# In[3]:


df=sns.load_dataset('healthexp')
df.head()


# In[4]:


#covariance 
import numpy as np


# In[5]:


df.cov()


# In[6]:


##Correlation
df.corr(method='spearman')


# In[7]:


#pearrson Correlation
df.corr(method='pearson')


# # Exploratary data analysis

# In[8]:


import pandas as pd
df=pd.read_csv('winequality-red.csv')
df.head()


# In[9]:


#summary of the dataset
df.info()


# In[10]:


#ddescriptive summary of the dataset
df.describe()


# In[12]:


df.shape


# In[13]:


#list down all the column names
df.columns


# In[14]:


df['quality'].unique()


# In[15]:


#missing values in the dataset
df.isnull().sum()


# In[17]:


#duplicate records
df[df.duplicated()]


# In[18]:


##remove the duplicates
df.drop_duplicates(inplace=True) #inplace


# In[20]:


df.shape


# #  5 Number Summary And Box Plot

# In[1]:


## Minimum,Maximum,Median,Q1,Q3,IQR


# In[2]:


import numpy as np


# In[8]:


lst_marks=[42,32,56,75,89,54,32,89,90,87,67,54,45,98,99,67,74]
minimum,Q1,median,Q3,maximum=np.quantile(lst_marks,[0,0.25,0.50,0.75,1.0])


# In[9]:


minimum,Q1,median,Q3,maximum


# In[10]:


IQR=Q3-Q1
print(IQR)


# In[11]:


lower_fence=Q1-1.5*(IQR)
higher_fence=Q3+1.5*(IQR)


# In[12]:


lower_fence


# In[13]:


higher_fence


# In[14]:


import seaborn as sns


# In[15]:


sns.boxplot(lst_marks)


# In[7]:


import seaborn as sns


# In[8]:


df=sns.load_dataset('tips')


# In[18]:


df.head()


# In[19]:


import numpy as np
mean=np.mean(df['total_bill'])
std=np.std(df['total_bill'])
print(mean,std)


# In[20]:


normalized_data=[]
for i in list(df['total_bill']):
    z_score=(i-mean)/std
    normalized_data.append(z_score)


# In[21]:


normalized_data


# In[22]:


sns.histplot(df['total_bill'])


#  # Feature scaling (standardization)
# from sklearn.preprocessing import StandardScaler Z-score formula 
# 

# In[48]:


from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()


# In[3]:


scaler


# In[9]:


scaler.fit(df[['total_bill']])#it compute the mean and standard devasion


# In[12]:


scaler.transform(df[['total_bill']])#scaling down the z-score formula


# In[15]:


df.head()


# In[17]:


import pandas as pd
pd.DataFrame(scaler.fit_transform(df[['total_bill','tip']]),columns=['total_bill','tip'])#in one single line


# In[18]:


scaler.transform([[13,4]])


# # Normaliztion--Min MAx Scaler
# xi-xmin/xmax-xmin

# In[29]:


df=sns.load_dataset('taxis')


# In[30]:


df.head()


# In[31]:


from sklearn.preprocessing import MinMaxScaler


# In[34]:


min_max=MinMaxScaler() #standarzation


# In[36]:


min_max.fit_transform(df[['distance','fare','tip']]) #see it will be in between 0,1


# In[37]:


min_max.fit(df[['distance','fare','tip']])


# In[39]:


min_max.transform([[1.6,7.0,2.15]])


# # unit vector
# under root of asq2+bsq2

# In[40]:


from sklearn.preprocessing import normalize


# In[46]:


import pandas as pd
unit_vector=pd.DataFrame(normalize(df[['distance','fare','tip']])) #under root of a square + b square


# In[47]:


unit_vector


# # Encoding
# Norminal/OHE Encoding

# In[49]:


import pandas as pd


# In[50]:


from sklearn.preprocessing import OneHotEncoder


# In[52]:


#Create a simple dataframe
df=pd.DataFrame({'color':['red','blue','green','green','red','blue']})


# In[53]:


df.head()


# In[54]:


#create an instance of onehotencoder
encoder=OneHotEncoder()


# In[57]:


encoded=encoder.fit_transform(df[['color']]).toarray() #this gives binary numbers


# In[58]:


import pandas as pd
encoder_df=pd.DataFrame(encoded,columns=encoder.get_feature_names_out())


# In[59]:


encoder_df


# In[60]:


df.head()


# # Label Encoder
# It assigns unique values to each other

# In[61]:


from sklearn.preprocessing import LabelEncoder


# In[62]:


lbl_encoder=LabelEncoder() #instance


# In[63]:


lbl_encoder.fit_transform(df[['color']]) #this gives unique values


# In[64]:


lbl_encoder.transform([['red']]) #afterwards if data may come so use just this to encode a new 


# # Ordinal Encoding
# It having instrinsic order or ranking.In this techique,each category is assigned a numerical value based on its position in the order. if e have catergorical variarbles 
# 1.High school:1
# 2.College:2
# 3.Graduate:3
# 4.Post-gradute:4
# 

# In[65]:


#ordinal encoding
from sklearn.preprocessing import OrdinalEncoder


# In[66]:


df=pd.DataFrame({'size':['small','medium','large','medium','small','large']})


# In[67]:


df


# In[68]:


#create as instance of ordinalencoder and then fit_transform
encoder=OrdinalEncoder(categories=[['small','medium','large']])


# In[69]:


encoder.fit_transform(df[['size']])


# In[70]:


encoder.transform([['small']])


# # Traget Guided Ordinal Encoding
# It is technique used to encode based on the relationship with the target variable.This encoding technique is useful when have a categorical variable with a large number of unique categories
# We can replace category in the categorical variable witha numerical value based on the mean or median of the target variable for that category

# In[72]:


#create a simple dataset
import pandas as pd
df=pd.DataFrame({'city':['New York','London','Paris','Tokyo','New York','Paris'],
                'price':[200,150,300,250,180,320]})


# In[73]:


df


# In[75]:


mean_price=df.groupby('city')['price'].mean().to_dict()


# mean_price

# In[76]:


mean_price


# In[77]:


df['city_encoded']=df['city'].map(mean_price)


# In[78]:


df


# In[79]:


import seaborn as sns


# In[81]:


df=sns.load_dataset('tips')


# In[82]:


df.head()


# In[92]:


mean_totalbill=df.groupby('time')['total_bill'].mean().to_dict()


# In[93]:


mean_totalbill


# In[94]:


df['encoded_time']=df['time'].map(mean_totalbill)


# In[95]:


df


# In[109]:


import pandas as pd
df=pd.read_csv("winequality-red.csv")
df.head()


# In[97]:


#Summary of the dataset
df.info()


# In[98]:


#descriptive summary of the dataset
df.describe()


# In[99]:


df.shape


# In[100]:


df.columns


# In[102]:


df['quality'].unique()


# In[103]:


#missing values in he dataset
df.isnull().sum()


# In[105]:


df.quality.value_counts().plot(kind='bar')
plt.xlabel('Wine Quality')
plt.ylabel("Count")


# In[106]:


df.head()


# In[107]:


for column in df.columns:
    sns.histplot(df[column],kde=True)


# In[108]:


sns.histplot(df['alcohol'])


# In[ ]:


#univariate,bivariate,multivariate analysis


# In[110]:


sns.pairplot(df)


# In[111]:


sns.catplot(x='quality',y='alcohol',data=df,kind='box')


# In[113]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[114]:


df=pd.read_csv('stud.csv')
df.head()


# # Data Check to perform
# * Check Missing values
# * Check Duplicates
# * Check data type
# *Check the number of unique values of each column
# *Check statistics of dataset
# *Check various categories present in the different categorical column

# In[115]:


##Checking missing values
df.isnull().sum()


# In[116]:


df.isna().sum()


# In[118]:


##Check Duplicates
df.duplicated().sum()


# In[119]:


#check datatypes
df.info()


# In[121]:


##3.1 Checking the number of uniques,values of each columns
df.nunique()


# In[122]:


#check the statistices of the datset
df.describe()


# In[123]:


#Explore more info about the data
df.head()


# In[127]:


#segrregate numerical and categorical features
[feature for feature in df.columns]# it gives all columns


# In[17]:


numerical_feature=[feature for feature in df.columns if df[feature].dtype!='O']#it gives all int columns
categorical_feature=[feature for feature in df.columns if df[feature].dtype=='O']


# In[131]:


numerical_feature


# In[132]:


categorical_feature


# In[133]:


#Aggregate the total score with mean
df['total_score']=(df['math_score']+df['reading_score']+df['writing_score'])
df['average']=df['total_score']/3
df.head()


# In[136]:


#Explore More Visualization
fig,axis=plt.subplots(1,2,figsize=(15,7))
plt.subplot(121)
sns.histplot(data=df,x='average',bins=30,kde=True,color='g')
plt.subplot(122)
sns.histplot(data=df,x='average',bins=30,kde=True,hue='gender')
plt.show()


# # Zomato Dataset Exploratory Data Analysis

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[147]:


import sys  

reload(sys)  
sys.setdefaultencoding('utf8')


# In[17]:


df=pd.read_csv('zomato.csv',encoding='latin-1')


# In[18]:


df.head()


# In[19]:


df.columns


# In[20]:


df.info()


# In[21]:


df.describe()


# # In data Analysis
# * Missing Values
# *Explore about the Numerical Variables
# *Explore ABout categorical Variable
# * Finding Relationship between Feature

# In[70]:


df.isnull().sum()


# In[10]:


[features for features in df.columns if df[features].isnull().sum()>0]


# In[ ]:





# In[14]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[69]:


df_country=pd.read_excel('Country-Code.xlsx')
df_country.head()


# In[24]:


df.columns


# In[28]:


#combine the dataset
final_df=pd.merge(df,df_country,on='Country Code',how='left')


# In[29]:


final_df


# In[30]:


##to check Data Types
final_df.dtypes


# In[32]:


country_names=final_df.Country.value_counts().index


# In[35]:


country_val=final_df.Country.value_counts().values


# In[41]:


##Pie Chart-top 3 countries that usese zomato
plt.pie(country_val[0:3],labels=country_names[0:3],autopct="%1.2f%%")


# In[42]:


final_df.columns


# In[48]:


ratings=final_df.groupby(['Aggregate rating','Rating color','Rating text']).size().reset_index().rename(columns={0:'Rating Count'})


# In[49]:


ratings


# In[51]:


ratings.head()


# In[60]:


import matplotlib
matplotlib.rcParams['figure.figsize']=(12,6)
sns.barplot(x='Aggregate rating',y='Rating Count',hue='Rating color',data=ratings,palette=['blue','red','orange','yellow','green','green'])


# In[62]:


##Count plot
sns.countplot(x='Rating color',data=ratings,palette=['blue','red','orange','yellow','green','green'])


# In[65]:


##Find the counties name that has given 0 rating
final_df.groupby(['Aggregate rating','Country']).size().reset_index().head(5)


# In[66]:


##find out which currency is used by which country
final_df[['Country','Currency']].groupby(['Country','Currency']).size().reset_index()


# In[67]:


##Which countries do have online deleviesrs option
final_df[final_df['Has Online delivery']=="Yes"].Country.value_counts()


# In[68]:


##drop city category Feature
df.drop('City_Category',axis=1,inplace=True)


# In[71]:


df.columns


# In[73]:


df_train=pd.read_csv('train.csv')


# In[74]:


df_train


# In[99]:


##import  the test data
df_test=pd.read_csv('test.csv')


# In[100]:


df_test


# In[101]:


#merge both train and test data
df=df_train.append(df_test)


# In[102]:


df


# In[103]:


df.info()


# In[104]:


df.describe()


# In[86]:


df.drop(['User_ID'],axis=1,inplace=True)


# In[87]:


df.head()


# In[88]:


#Convert Gender catagorical to numerical 
df['Gender']=df['Gender'].map({'F':0,'M':1})


# In[89]:


df


# In[136]:


#HAndel age catageriocal feature to  numnerical
df['Age'].unique()


# In[137]:


df['Age']=df['Age'].map({'0-17':1,'18-25':2,'26-35':3,'36-45':4,'46-50':5,'51-55':6,'55+':7})


# In[138]:


df


# In[95]:


#Second Techique
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['Age']=label_encoder.fit_transform(df['Age'])
df['Age'].unique()


# In[96]:


df


# In[98]:


#fixing categorical City_categort
df_city=pd.get_dummies(df['City_Category'],drop_first=True)


# In[105]:


df_city


# In[106]:


df=pd.concat([df,df_city],axis=1)
df.head()


# In[108]:


#drop City category
df.drop('City_Category',axis=1,inplace=True)


# In[109]:


df.head()


# In[110]:


#Check missing values
df.isnull().sum()


# In[111]:


#Focus on replacing missing values
df['Product_Category_2'].unique()


# In[112]:


df['Product_Category_2'].value_counts()


# In[113]:


##Replace the missing values with mode
df['Product_Category_2']=df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0])


# In[114]:


df['Product_Category_2'].isnull().sum()


# In[115]:


df['Product_Category_3']=df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0])


# In[116]:


df['Product_Category_3'].isnull().sum()


# In[117]:


df['Stay_In_Current_City_Years'].unique()


# In[127]:


#replace 4+ with 4
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].str.replace('+', '')


# In[128]:


df.head()


# In[129]:


#Convert object into intergers
df['Stay_In_Current_City_Years']=df['Stay_In_Current_City_Years'].astype(int)


# In[130]:


df.info()


# In[132]:


df['B']=df['B'].astype(int)
df['C']=df['C'].astype(int)


# In[133]:


df.info()


# In[139]:


#Visulisation
sns.barplot('Age','Purchase',hue="Gender",data=df)


# In[9]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[10]:


df=pd.read_csv('stud (2).csv')
df.head()


# In[11]:


df.shape


# In[12]:


#missing values
df.isnull().sum()


# In[13]:


#check duplicated
df.duplicated().sum()


# In[14]:


#Ckeck datatypes
df.info()


# In[15]:


#checking number of uniques values of each columns
df.nunique()


# In[16]:


#check the statistics of the dataset
df.describe()


# In[17]:


df.head()


# In[18]:


numerical_feature=[feature for feature in df.columns if df[feature].dtype!='O']
categorical_feature=[feature for feature in df.columns if df[feature].dtype=='O']


# In[19]:


numerical_feature


# In[20]:


categorical_feature


# In[ ]:


#explore more visualization
fig,axis=plt.subplot(1,2,figsize=(15,7))
plt.subplot(121)
sns.histplot(data=df,x=)


# # Flight Price 

# In[21]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[22]:


df=pd.read_excel('flight_price.xlsx')


# In[23]:


df.head()


# In[24]:


df.tail()


# In[25]:


#get the basic info of data
df.info()


# In[26]:


df.describe()


# In[27]:


df.head()


# In[34]:


#feature engineering
#data_of_journey seperate date month and year
df['Date']=df['Date_of_Journey'].str.split('/').str[0]
df['Month']=df['Date_of_Journey'].str.split('/').str[1]
df['Year']=df['Date_of_Journey'].str.split('/').str[2]


# In[35]:


df.head()


# In[36]:


df.info() #see all the date month and year are still objects soo we should convert to numerical


# In[37]:


df['Date']=df['Date'].astype(int)#it convert object to interger
df['Month']=df['Month'].astype(int)
df['Year']=df['Year'].astype(int)


# In[38]:


df.info()


# In[40]:


##Drop data of journey
df.drop('Date_of_Journey',axis=1,inplace=True)


# In[47]:


df['Arrival_Time']=df['Arrival_Time'].apply(lambda x:x.split(' ')[0])


# In[48]:


##now time
df['Arrival_hour']=df['Arrival_Time'].str.split(':').str[0]
##now time
df['Arrival_min']=df['Arrival_Time'].str.split(':').str[1]


# In[49]:


df.head()


# In[50]:


df.head(2)


# In[51]:


df['Arrival_hour']=df['Arrival_hour'].astype(int)
df['Arrival_min']=df['Arrival_min'].astype(int)


# In[52]:


df.drop('Arrival_Time',axis=1,inplace=True)


# In[53]:


df.head()


# In[54]:


#dep_time
df['Depature_hour']=df['Dep_Time'].str.split(':').str[0]
df['Depature_min']=df['Dep_Time'].str.split(':').str[1]


# In[55]:


df.head(2)


# In[56]:


df['Depature_hour']=df['Depature_hour'].astype(int)
df['Depature_min']=df['Depature_min'].astype(int)


# In[57]:


df.info()


# In[58]:


df.drop('Dep_Time',axis=1,inplace=True)


# In[59]:


df.head(2)


# In[60]:


df['Total_Stops'].unique()


# In[62]:


df[df['Total_Stops'].isnull()]


# In[63]:


#total stops that are converted to numerical
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0,'1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4,np.nan:1})


# In[64]:


df[df['Total_Stops'].isnull()]


# In[65]:


df.head()


# In[66]:


df['Duration'].str.split(' ').str[0].str.split('h').str[0]


# In[71]:


from sklearn.preprocessing import OneHotEncoder


# In[73]:


encoder=OneHotEncoder()


# In[76]:


encoder.fit_transform(df[['Airline','Source','Destination']]).toarray()


# In[79]:


pd.DataFrame(encoder.fit_transform(df[['Airline','Source','Destination']]).toarray(),columns=encoder.get_feature_names_out())


# # Steps we are going to follow 
# * Data Cleaning
# * Exploratory Data Analysis
# * Feature engineering

# In[80]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[83]:


df=pd.read_csv('https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv')


# In[84]:


df.head()


# In[85]:


df.shape


# In[86]:


df.info()


# In[87]:


##Summary of the dataset
df.describe()


# In[88]:


df.isnull().sum()


# In[89]:


df.head(2)


# In[90]:


#Check if all the values 
df['Reviews'].unique()


# In[92]:


df['Reviews'].str.isnumeric().sum()


# In[94]:


df[~df['Reviews'].str.isnumeric()]


# In[95]:


df_copy=df.copy()


# In[96]:


df_copy=df_copy.drop(df_copy.index[10472])


# In[97]:


##Convert Reviews Datatype to int
df_copy['Review']=df_copy['Review'].astype()


# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df=pd.read_csv('height-weight.csv')


# In[3]:


df.head()


# In[4]:


plt.scatter(df['Weight'],df['Height'])
plt.xlabel("Weight")
plt.ylabel("Height")


# In[22]:


#divide our dataset into independent and dependent edatures
X=df[['Weight']] ##independent feature
y=df['Height'] ##dependent feature


# In[23]:


##Train test palte
from sklearn.model_selection import train_test_split


# In[24]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)


# In[25]:


X.shape


# In[26]:


X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[17]:


##Standardize the dataset Train independent data
from sklearn.preprocessing import StandardScaler


# In[18]:


scaler=StandardScaler()


# In[29]:


X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)


# In[30]:


plt.scatter(X_train,y_train)


# In[31]:


##Train model simple linear regression model
from sklearn.linear_model import LinearRegression


# In[32]:


regressor=LinearRegression()


# In[33]:


regressor.fit(X_train,y_train)


# In[35]:


print("The slope or coefficient of weight is ",regressor.coef_)
print("Intercept",regressor.intercept_)


# In[37]:


plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train),'r')


# In[9]:


from sklearn.preprocessing import 


# In[23]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[32]:


dataset=pd.read_csv("Algerian_forest_fires_dataset_UPDATE.csv")


# In[33]:


dataset.head()


# In[34]:


dataset.info()


# In[35]:


##missing values
dataset[dataset.isnull().any(axis=1)]


# In[27]:


dataset.loc[:122,"Region"]=0
dataset.loc[122:,"Region"]=1
df=dataset


# In[28]:


df.info


# In[29]:


df[['Reigin']]=df[['Region']].astype(int)


# In[30]:


df.head()


# In[31]:


df.isnull().sum()


# In[ ]:





# In[16]:


df.iloc[[122]]


# In[36]:


df=pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')


# In[37]:


df.head()


# In[38]:


df.columns


# In[40]:


#drop month,day,year
df.drop(['day','month','year'],axis=1,inplace=True)


# In[41]:


df.head()


# In[42]:


df['Classes'].value_counts()


# In[43]:


#Encoding
df["Classes"]=np.where(df['Classes'].str.contains("not fire"),0,1)


# In[44]:


df.tail()


# In[45]:


X_train.corr()


# In[46]:


df['Classes'].value_counts()


# In[47]:


#Independent and dependent features
X=df.drop('FWI',axis=1)
y=df['FWI']


# In[48]:


X.head()


# In[49]:


y


# In[50]:


#Tain Test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42)


# In[51]:


X_train.shape,X_test.shape


# In[52]:


##Feature Selection based on correlation
X_train.corr()


# In[53]:


#Check for multicolinearity
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr,annot=True)


# In[54]:


def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j]) > threshold:
                colname=corr_martix.columns[i]
                col_corr.add(colname)
    return col_corr


# In[ ]:


##Threshold=Domain


# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


df=pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')


# In[24]:


df.head()


# In[25]:


df.columns


# In[26]:


#drop month,day and year
df.drop(['day','month','year'],axis=1,inplace=True)


# In[27]:


df.head()


# In[28]:


df['Classes'].value_counts()


# # Simple Linear Regression

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#Read a dataset
df=pd.read_csv('height-weight.csv')


# In[3]:


df.head(2)


# In[4]:


plt.scatter(df['Weight'],df['Height'])
plt.xlabel('Weight')
plt.ylabel('Height')


# In[16]:


##Divide our dataset into independent and dependent  feature
X=df[['Weight']] #Independent Feature
y=df['Height']#Dependent Feature


# In[17]:


#Train test split
from sklearn.model_selection import train_test_split


# In[18]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42) #20 percent of dataset is used for test the data


# In[19]:


X.shape


# In[20]:


X_train.shape,X_test.shape #see in the test data 20 percent of data is there


# In[22]:


#standardize the dataset Train independent data
from sklearn.preprocessing import StandardScaler # means using Z-score 


# In[23]:


scaler=StandardScaler()


# In[25]:


X_train.head()


# In[26]:


X_train=scaler.fit_transform(X_train) #fit is used for mean and standardizion form
X_test=scaler.transform(X_test)


# In[27]:


plt.scatter(X_train,y_train)


# In[29]:


##rain the simple linear regression model
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()


# In[30]:


regressor.fit(X_train,y_train)


# In[34]:


print('The slope or coefficient of weight is',regressor.coef_) #slope
print('Intercept',regressor.intercept_)


# In[36]:


plt.scatter(X_train,y_train)
plt.plot(X_train,regressor.predict(X_train))


# In[37]:


y_pred_test=regressor.predict(X_test)


# In[38]:


y_pred_test,y_test


# In[40]:


plt.scatter(X_test,y_test)
plt.plot(X_test,regressor.predict(X_test))


# In[41]:


#Perfomance Matrix

##MSE,MAE,RMSC


# # R square and adjusted R square

# In[43]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[44]:


mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
print(mse)
print(mae)
print(rmse)


# In[45]:


from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred_test)


# In[46]:


score


# In[47]:


residuals=y_test-y_pred_test
residuals


# In[48]:


import seaborn as sns
sns.distplot(residuals,kde=True)


# # Multiple Linear Regression

# In[1]:


from sklearn.datasets import fetch_california_housing


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


california=fetch_california_housing()


# In[4]:


type(california)


# In[5]:


california.keys()


# In[6]:


print(california.DESCR)


# In[7]:


california.target_names


# In[8]:


print(california.data)


# In[9]:


print(california.target)


# In[10]:


california.feature_names


# In[11]:


##lets preprare the dataframe
dataset=pd.DataFrame(california.data,columns=california.feature_names)
dataset.head()


# In[12]:


dataset['Price']=california.target


# In[13]:


dataset.head()


# In[14]:


dataset.info()


# In[15]:


dataset.isnull().sum()


# In[16]:


import seaborn as sns
sns.pairplot(dataset)


# In[23]:


dataset.corr()


# In[24]:


sns.heatmap(dataset.corr(),annot=True)


# In[22]:


dataset.head()


# In[28]:


#independent and dependent feature
X=dataset.iloc[:,:,-1] #independent features
y=dataset.iloc[:,-1] #dependent features


# In[ ]:





# In[ ]:





# In[ ]:




