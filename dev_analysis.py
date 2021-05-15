#!/usr/bin/env python
# coding: utf-8

# # Project Title - Global Life Expectancy
# 
# TODO - Write some introduction about your project here: describe the dataset, where you got it from, what you're trying to do with it, and which tools & techniques you're using. You can also mention about the course [Data Analysis with Python: Zero to Pandas](zerotopandas.com), and what you've learned from it.

# ### How to run the code
# 
# This is an executable [*Jupyter notebook*](https://jupyter.org) hosted on [Jovian.ml](https://www.jovian.ml), a platform for sharing data science projects. You can run and experiment with the code in a couple of ways: *using free online resources* (recommended) or *on your own computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing this notebook is to click the "Run" button at the top of this page, and select "Run on Binder". This will run the notebook on [mybinder.org](https://mybinder.org), a free online service for running Jupyter notebooks. You can also select "Run on Colab" or "Run on Kaggle".
# 
# 
# #### Option 2: Running on your computer locally
# 
# 1. Install Conda by [following these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). Add Conda binaries to your system `PATH`, so you can use the `conda` command on your terminal.
# 
# 2. Create a Conda environment and install the required libraries by running these commands on the terminal:
# 
# ```
# conda create -n zerotopandas -y python=3.8 
# conda activate zerotopandas
# pip install jovian jupyter numpy pandas matplotlib seaborn opendatasets --upgrade
# ```
# 
# 3. Press the "Clone" button above to copy the command for downloading the notebook, and run it on the terminal. This will create a new directory and download the notebook. The command will look something like this:
# 
# ```
# jovian clone notebook-owner/notebook-id
# ```
# 
# 
# 
# 4. Enter the newly created directory using `cd directory-name` and start the Jupyter notebook.
# 
# ```
# jupyter notebook
# ```
# 
# You can now access Jupyter's web interface by clicking the link that shows up on the terminal or by visiting http://localhost:8888 on your browser. Click on the notebook file (it has a `.ipynb` extension) to open it.
# 

# ## Downloading the Dataset
# 
# **TODO** - add some explanation here

# > Instructions for downloading the dataset (delete this cell)
# >
# > - Find an interesting dataset on this page: https://www.kaggle.com/datasets?fileType=csv
# > - The data should be in CSV format, and should contain at least 3 columns and 150 rows
# > - Download the dataset using the [`opendatasets` Python library](https://github.com/JovianML/opendatasets#opendatasets)

# In[1]:


get_ipython().system('pip install jovian opendatasets --upgrade --quiet')


# Let's begin by downloading the data, and listing the files within the dataset.

# In[2]:


# Change this
dataset_url = 'https://www.kaggle.com/augustus0498/life-expectancy-who' 


# In[3]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[3]:


# Change this
data_dir = './life-expectancy-who'


# In[4]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[5]:


project_name = "global-life-expectancy" # change this (use lowercase letters and hyphens only)


# In[6]:


get_ipython().system('pip install jovian --upgrade -q')


# In[53]:


import jovian


# In[54]:


jovian.commit(project=project_name)


# ## Data Preparation and Cleaning
# 
# We have raw dataset. To make it available for analysis we have to remove unwanted data, make the data into more readable format, search for any duplicates in the data, etc. All of these steps 
# 
# 

# > Instructions (delete this cell):
# >
# > - Load the dataset into a data frame using Pandas
# > - Explore the number of rows & columns, ranges of values etc.
# > - Handle missing, incorrect and invalid data
# > - Perform any additional steps (parsing dates, creating additional columns, merging multiple dataset etc.)

# In[1]:


import pandas as pd
import numpy as np


# In[203]:


led_df_raw = pd.read_csv('./life-expectancy-who/led.csv')
led_df_raw


# In[204]:


#Step 1 : Renaming few columns in the dataset to make them easier to read

led_df_raw.rename(columns={'thinness1-19years':'Thinness_Aged_10-19_yrs'}, inplace=True)
led_df_raw.rename(columns={'thinness5-9years':'Thinness_Aged_5-9_yrs'}, inplace=True)
led_df_raw.rename(columns={'Lifeexpectancy':'Life_Expectancy'}, inplace=True)
led_df_raw.rename(columns={'AdultMortality':'Adult_Mortality'}, inplace=True)
led_df_raw.rename(columns={'infantdeaths':'Infant_Deaths'}, inplace=True)
led_df_raw.rename(columns={'percentageexpenditure':'Percentage_Expenditure'}, inplace=True)
led_df_raw.rename(columns={'under-fivedeaths':'Under_Five_Deaths'}, inplace=True)
led_df_raw.rename(columns={'Totalexpenditure':'Total_Expenditure'}, inplace=True)
led_df_raw.rename(columns={'Incomecompositionofresources':'Income_Composition'}, inplace=True)


# In[205]:


led_df_raw.info()


# In[206]:


#Step 2 : To find the number of missing values if present in the dataset
count_nan_in_df  = led_df_raw.isnull().sum()
print ('Count of NaN: ' + str(count_nan_in_df))


# In[207]:


#Replacing NaN (Null) values with the mean value for that particular column
led_df_raw.fillna(led_df_raw.mean(axis=0,skipna=True), inplace=True)
led_df_raw


# In[208]:


count_nan_in_df  = led_df_raw.isnull().sum().sum()
print ('Count of NaN: ' + str(count_nan_in_df))


# In[209]:


#Step 3 : To find outliers.
#Use subplot to plot multiple boxplots and the points that do not fall in the range indicates outliers for that particular column.
import matplotlib.pyplot as plt
cont_vars = list(led_df_raw.columns)[3:] 
def outliers_visual(data):
    plt.figure(figsize=(15, 40))
    i = 0
    for col in cont_vars:
        i += 1
        plt.subplot(8, 4, i)
        plt.boxplot(data[col])
        plt.title('{} boxplot'.format(col),fontsize=10)
    plt.show()
outliers_visual(led_df_raw)


# The above boxplots indicates that there are outliers present in all the columns. Let's indentify outliers using the IQR(Interquartile Range) method, anything outside the 1.5 times the IQR can be considered as an outlier.

# In[210]:


def outlier_count(col, data=led_df_raw):
    print(15*'-' + col+15*'-')
    q75=np.percentile(data[col],75)
    q25=np.percentile(data[col],25)
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    outlier_count = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    outlier_percent = round(outlier_count/len(data[col])*100, 2)
    print('Number of outliers: {}'.format(outlier_count))
    print('Percent of data that is outlier: {}%'.format(outlier_percent))
    
for col in cont_vars:
    outlier_count(col)


# In[211]:


#We will winsorise the data to handle outliers
# For Eg.
from scipy.stats.mstats import winsorize
data={'number':[ -100,-120,1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 , 14 , 15 ,100,120,150]}
#20 elements
# .1 10% 2 elements 15*20/100
df=pd.DataFrame(data=data)
plt.subplot(121)
plt.boxplot(df)
wins_data = winsorize(data['number'], limits=(0.1,0.15))
plt.subplot(122)
plt.boxplot(wins_data)


# We have considered a dataframe having only one column number which consists of 20 numbers. Here if we plot this dataframe, we can see that -100,-120,100,120,150 are outliers as majority of the numbers are within the limits 1 to 15. In order to handle these outliers, we will winsorize (limiting) the values for number on its own until no outliers remain. 10% of lower limit i.e elements -100, -120 will be replaced by the 1 (the minimum number in the range) and 15% of upper limit i.e elements 100,120,150 will be replaced by 15 (maximum number in the range). 

# In[212]:


from scipy.stats.mstats import winsorize
def winsorization_data(col, lower_limit, upper_limit, show_plot=True):
    winsorized_data = winsorize(led_df_raw[col], limits=(lower_limit, upper_limit))
    winsorized_dict[col] = winsorized_data
    if show_plot == True:
        plt.figure(figsize=(15,5))
        plt.subplot(121)
        plt.boxplot(led_df_raw[col])
        plt.title('original {}'.format(col))
        plt.subplot(122)
        plt.boxplot(winsorized_data)
        plt.title('wins=({},{}) {}'.format(lower_limit, upper_limit, col))
        plt.show()


# In[213]:


winsorized_dict = {}
winsorization_data(cont_vars[0],0.008,0,show_plot=False)
winsorization_data(cont_vars[1],0,0.03,show_plot=False)
winsorization_data(cont_vars[2],0,0.11,show_plot=True)
winsorization_data(cont_vars[3],0,0.002,show_plot=False)
winsorization_data(cont_vars[4],0,0.14,show_plot=False)
winsorization_data(cont_vars[5],0.11,0,show_plot=False)
winsorization_data(cont_vars[6],0,0.19,show_plot=False)
winsorization_data(cont_vars[7],0,0,show_plot=False)
winsorization_data(cont_vars[8],0,0.14,show_plot=False)
winsorization_data(cont_vars[9],0.1,0,show_plot=False)
winsorization_data(cont_vars[10],0,0.02,show_plot=False)
winsorization_data(cont_vars[11],0.11,0,show_plot=False)
winsorization_data(cont_vars[12],0,0.19,show_plot=False)
winsorization_data(cont_vars[13],0,0.11,show_plot=False)
winsorization_data(cont_vars[14],0,0,show_plot=False)
winsorization_data(cont_vars[15],0,0.04,show_plot=False)
winsorization_data(cont_vars[16],0,0.04,show_plot=False)
winsorization_data(cont_vars[17],0.05,0,show_plot=False)
winsorization_data(cont_vars[18],0.03,0.01,show_plot=False)


# In[214]:


new_led_df_raw = led_df_raw.iloc[:, 0:3]
for col in cont_vars:
    new_led_df_raw[col] = winsorized_dict[col]


# In[215]:


new_led_df_raw_df_cols = list(new_led_df_raw.columns)[3:] 
def outliers_handle(data):
    plt.figure(figsize=(15, 40))
    i = 0
    for col in new_led_df_raw_df_cols:
        i += 1
        plt.subplot(8, 4, i)
        plt.boxplot(data[col])
        plt.title('{} boxplot'.format(col),fontsize=10)
    plt.show()
outliers_handle(new_led_df_raw)


# Now if we visualize all the columns, we can see we have no outliers.

# In[216]:


new_led_df_raw


# In[217]:


#Step 4 : To check if any duplicate records are present
duplicateRowsDF = new_led_df_raw[new_led_df_raw.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# In[218]:


#Step 5 : Deleting all the Country records that have data only for a single year 
count_country = new_led_df_raw.groupby('Country')['Country'].count()
count_country.sort_values(ascending=True).head(10)


# In[219]:


led_df_raw_index=new_led_df_raw.set_index("Country")
led_df_raw_country=led_df_raw_index.drop(["Dominica","Palau","Nauru","Tuvalu","CookIslands","MarshallIslands","Monaco","Niue","SaintKittsandNevis","SanMarino"])
led_df_raw_final=led_df_raw_country.reset_index()


# In[220]:


#Extract a copy of the data into a new data frame led_df so that We can continue to modify further without affecting the original data frame.
led_df=led_df_raw_final.copy()
led_df


# In[19]:


import jovian


# In[152]:


jovian.commit()


# ## Exploratory Analysis and Visualization
# 
# **TODO** - write some explanation here.
# 
# 

# > Instructions (delete this cell)
# > 
# > - Compute the mean, sum, range and other interesting statistics for numeric columns
# > - Explore distributions of numeric columns using histograms etc.
# > - Explore relationship between columns using scatter plots, bar charts etc.
# > - Make a note of interesting insights from the exploratory analysis

# Let's begin by importing`matplotlib.pyplot` and `seaborn`.

# In[160]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# 1. Let us first explore all the variables on its own by plotting histograms for each column

# In[161]:


led_df_column = list(led_df.columns)[3:] 
def all_columns(data):
    plt.figure(figsize=(30,30))
    i = 0
    for column in led_df_column:
        i += 1
        plt.subplot(8, 4, i)
        plt.hist(data[column])
        plt.title(format(column),fontsize=12)
    plt.show()
all_columns(led_df)


# 2. How many countries are involved in the data collection and their row count

# In[165]:


led_df.Country.nunique()


# In[166]:


country_count = led_df.groupby('Country')['Country'].count()
country_count


# 3. Comparison of Mean life expectancy of all developing and developed countries throughout the years

# In[167]:


#Create a dataframe that has status (developing,developed) and mean of life expectancy
mean_developing = led_df.groupby('Status')['Life_Expectancy'].mean()
mean_developing.head()


# In[168]:


plt.figure(figsize=(5, 5))
plt.title('Comparison of mean life expectancy of developing and developed countries')
mean_developing.plot(x='Status',y='Life_Expectancy',kind='bar',color='orange')
plt.xlabel('Status');
plt.ylabel('Life_Expectancy');
plt.show() 


# By looking at the bar graph we can conclude that the average life expectancy of all developed countries is better than developing countries

# 4. A lineplot to visualize the changing trends in life expectancy throughout the years

# In[169]:


sns.lineplot(data=led_df,x='Year', y='Life_Expectancy', marker='*',color='orange')
plt.title('Life Expectancy by Year')
plt.show()


# 5. A Correlation Matrix with all the variables to get an understanding on which variables are most influential.

# In[170]:


led_df_developing = led_df[led_df['Status']=='Developing']


# In[171]:


corr_led_df_developing = pd.DataFrame(led_df_developing,columns=led_df_column)
corr_matrix_developing = corr_led_df_developing.corr()
plt.figure(figsize=(25,6))
plt.ylim(18, 0)
plt.title('Correlation matrix heatmap for developing nations')
sns.heatmap(corr_matrix_developing,annot=True,fmt='.2g')
plt.show()


# In[172]:


led_df_developed = led_df[led_df['Status']=='Developed']


# In[173]:


corr_led_df_developed = pd.DataFrame(led_df_developed,columns=led_df_column)
corr_matrix_developed = corr_led_df_developed.corr()
plt.figure(figsize=(25,6))
plt.ylim(18, 0)
plt.title('Correlation matrix heatmap for developed nations')
sns.heatmap(corr_matrix_developed,annot=True,fmt='.2g')
plt.show()


# From the correlation heatmap above we can identify the attributes that have impacted life expectancy positively or negatively.

# Let us save and upload our work to Jovian before continuing

# In[29]:


import jovian


# In[30]:


jovian.commit()


# ## Asking and Answering Questions
# 
# TODO - write some explanation here.
# 
# 

# #### Q1: Which is the country that has the minimum life expectancy in both developed and developing nations

# In[174]:


mean_developing_country = led_df_developing.groupby('Country')['Life_Expectancy'].mean()


# In[175]:


mean_developing_country.sort_values(ascending=True).head(1)


# In[176]:


mean_developed_country = led_df_developed.groupby('Country')['Life_Expectancy'].mean()


# In[177]:


mean_developed_country.sort_values(ascending=True).head(1)


# SierraLeone a country in west africa is the amongst the developing nations that has the minimum life expectancy and Lithuania is the nation with minimum life expectancy amongst developed nations.

# #### Q2: What is the life Expectancy of countries with minimum and maximum alcohol consumption

# In[178]:


alcohol_consumption = led_df.groupby('Country')['Alcohol'].mean()


# In[179]:


max_alcohol_consumption=alcohol_consumption.sort_values(ascending=False).head(1)
max_alcohol_consumption


# In[181]:


min_alcohol_consumption=alcohol_consumption.sort_values(ascending=True).head(1)
min_alcohol_consumption


# In[182]:


alcohol_consumption_country=led_df[(led_df.Country=="Belarus")|(led_df.Country=="Afghanistan")]
alcohol_consumption_led=alcohol_consumption_country.groupby('Country')['Life_Expectancy'].mean()
alcohol_consumption_led
plt.figure(figsize=(10,3))
sns.barplot(x=alcohol_consumption_led,y=alcohol_consumption_led.index)


# From the bar graph above, we can infer that the life expectancy of the country wherein the alcohol consumption is maximum i.e Belarus life expectancy is also more as compared to Afghanistan that has the least alcool consumption. 

# #### Q3: What is the life expectancy of countries with minimum and maximum government expenditure on healthcare

# In[183]:


gov_expenditure = led_df.groupby('Country')['Percentage_Expenditure'].mean()


# In[184]:


max_gov_expenditure=gov_expenditure.sort_values(ascending=False).head(1)
max_gov_expenditure


# In[192]:


min_gov_expenditure=gov_expenditure.sort_values(ascending=True).head(1)
min_gov_expenditure


# In[186]:


gov_expenditure_country=led_df[(led_df.Country=="Switzerland")|(led_df.Country=="Egypt")]
gov_led=gov_expenditure_country.groupby('Country')['Life_Expectancy'].mean()
gov_led
plt.figure(figsize=(10,3))
sns.barplot(x=gov_led,y=gov_led.index)


# From the bar graph above, we can infer that the life expectancy of the country wherein the government spends maximum expenditure on healthcare is more as compared to minimum expenditure on healthcare country.

# #### Q4: What is the life expectancy of the countries that have good and poor education

# In[193]:


schooling_mean=led_df.groupby('Country')['Schooling'].mean()


# In[194]:


schooling_max=schooling_mean.sort_values(ascending=False).head(1)
schooling_max


# In[195]:


schooling_min=schooling_mean.sort_values(ascending=True).head(1)
schooling_min


# In[196]:


schooling_country=led_df[(led_df.Country=="Australia")|(led_df.Country=="SouthSudan")]
schooling_led=schooling_country.groupby('Country')['Life_Expectancy'].mean()
plt.figure(figsize=(10,3))
sns.barplot(x=schooling_led,y=schooling_led.index)


# The country which has a good education system, life expectancy is more as compared to the country with a poor education system.

# #### Q5: What is the life expectancy based on the income of people

# In[222]:


income_mean=led_df.groupby('Country')['Income_Composition'].mean()


# In[223]:


income_max=income_mean.sort_values(ascending=False).head(1)
income_max


# In[224]:


income_min=income_mean.sort_values(ascending=True).head(1)
income_min


# In[225]:


income_country=led_df[(led_df.Country=="Norway")|(led_df.Country=="Niger")]
income_led=income_country.groupby('Country')['Life_Expectancy'].mean()
plt.figure(figsize=(10,3))
sns.barplot(x=income_led,y=income_led.index)


# Let us save and upload our work to Jovian before continuing.

# In[226]:


import jovian


# In[227]:


jovian.commit()


# ## Inferences and Conclusion
# 
# **TODO** - Write some explanation here: a summary of all the inferences drawn from the analysis, and any conclusions you may have drawn by answering various questions.

# In[228]:


import jovian


# In[ ]:


jovian.commit()


# ## References and Future Work
# 
# **TODO** - Write some explanation here: ideas for future projects using this dataset, and links to resources you found useful.

# > Submission Instructions (delete this cell)
# > 
# > - Upload your notebook to your Jovian.ml profile using `jovian.commit`.
# > - **Make a submission here**: https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas/assignment/course-project
# > - Share your work on the forum: https://jovian.ml/forum/t/course-project-on-exploratory-data-analysis-discuss-and-share-your-work/11684
# > - Share your work on social media (Twitter, LinkedIn, Telegram etc.) and tag [@JovianML](https://twitter.com/jovianml)
# >
# > (Optional) Write a blog post
# > 
# > - A blog post is a great way to present and showcase your work.  
# > - Sign up on [Medium.com](https://medium.com) to write a blog post for your project.
# > - Copy over the explanations from your Jupyter notebook into your blog post, and [embed code cells & outputs](https://medium.com/jovianml/share-and-embed-jupyter-notebooks-online-with-jovian-ml-df709a03064e)
# > - Check out the Jovian.ml Medium publication for inspiration: https://medium.com/jovianml
# 
# 
#  

# In[ ]:


import jovian


# In[ ]:


jovian.commit()


# In[ ]:




