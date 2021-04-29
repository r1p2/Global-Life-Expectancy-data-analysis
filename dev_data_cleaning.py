#!/usr/bin/env python
# coding: utf-8

# ## Data Analysis with Python: Zero to Pandas - Course Project Guidelines
# #### (remove this cell before submission)
# 
# Important links:
# - Make submissions here: https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas/assignment/course-project
# - Ask questions here: https://jovian.ml/forum/t/course-project-on-exploratory-data-analysis-discuss-and-share-your-work/11684
# - Find interesting datasets here: https://jovian.ml/forum/t/recommended-datasets-for-course-project/11711
# 
# 
# This is the starter notebook for the course project for [Data Analysis with Python: Zero to Pandas](https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas). You will pick a real-world dataset of your choice and apply the concepts learned in this course to perform exploratory data analysis. Use this starter notebook as an outline for your project . Focus on documentation and presentation - this Jupyter notebook will also serve as a project report, so make sure to include detailed explanations wherever possible using Markdown cells.
# 
# ### Evaluation Criteria
# 
# Your submission will be evaluated using the following criteria:
# 
# * Dataset must contain at least 3 columns and 150 rows of data
# * You must ask and answer at least 4 questions about the dataset
# * Your submission must include at least 4 visualizations (graphs)
# * Your submission must include explanations using markdown cells, apart from the code.
# * Your work must not be plagiarized i.e. copy-pasted for somewhere else.
# 
# 
# Follow this step-by-step guide to work on your project.
# 
# 
# ### Step 1: Select a real-world dataset 
# 
# - Find an interesting dataset on this page: https://www.kaggle.com/datasets?fileType=csv
# - The data should be in CSV format, and should contain at least 3 columns and 150 rows
# - Download the dataset using the [`opendatasets` Python library](https://github.com/JovianML/opendatasets#opendatasets)
# 
# Here's some sample code for downloading the [US Elections Dataset](https://www.kaggle.com/tunguz/us-elections-dataset):
# 
# ```
# import opendatasets as od
# dataset_url = 'https://www.kaggle.com/tunguz/us-elections-dataset'
# od.download('https://www.kaggle.com/tunguz/us-elections-dataset')
# ```
# 
# You can find a list of recommended datasets here: https://jovian.ml/forum/t/recommended-datasets-for-course-project/11711
# 
# ### Step 2: Perform data preparation & cleaning
# 
# - Load the dataset into a data frame using Pandas
# - Explore the number of rows & columns, ranges of values etc.
# - Handle missing, incorrect and invalid data
# - Perform any additional steps (parsing dates, creating additional columns, merging multiple dataset etc.)
# 
# 
# ### Step 3: Perform exploratory analysis & visualization
# 
# - Compute the mean, sum, range and other interesting statistics for numeric columns
# - Explore distributions of numeric columns using histograms etc.
# - Explore relationship between columns using scatter plots, bar charts etc.
# - Make a note of interesting insights from the exploratory analysis
# 
# ### Step 4: Ask & answer questions about the data
# 
# - Ask at least 4 interesting questions about your dataset
# - Answer the questions either by computing the results using Numpy/Pandas or by plotting graphs using Matplotlib/Seaborn
# - Create new columns, merge multiple dataset and perform grouping/aggregation wherever necessary
# - Wherever you're using a library function from Pandas/Numpy/Matplotlib etc. explain briefly what it does
# 
# 
# ### Step 5: Summarize your inferences & write a conclusion
# 
# - Write a summary of what you've learned from the analysis
# - Include interesting insights and graphs from previous sections
# - Share ideas for future work on the same topic using other relevant datasets
# - Share links to resources you found useful during your analysis
# 
# 
# ### Step 6: Make a submission & share your work
# 
# - Upload your notebook to your Jovian.ml profile using `jovian.commit`.
# - **Make a submission here**: https://jovian.ml/learn/data-analysis-with-python-zero-to-pandas/assignment/course-project
# - Share your work on the forum: https://jovian.ml/forum/t/course-project-on-exploratory-data-analysis-discuss-and-share-your-work/11684
# - Browse through projects shared by other participants and give feedback
# 
# 
# ### (Optional) Step 7: Write a blog post
# 
# - A blog post is a great way to present and showcase your work.  
# - Sign up on [Medium.com](https://medium.com) to write a blog post for your project.
# - Copy over the explanations from your Jupyter notebook into your blog post, and [embed code cells & outputs](https://medium.com/jovianml/share-and-embed-jupyter-notebooks-online-with-jovian-ml-df709a03064e)
# - Check out the Jovian.ml Medium publication for inspiration: https://medium.com/jovianml
# 
# 
# 
# 
# 
# ### Example Projects
# 
# Refer to these projects for inspiration:
# 
# * [Analyzing StackOverflow Developer Survey Results](https://jovian.ml/aakashns/python-eda-stackoverflow-survey)
# 
# * [Analyzing Covid-19 data using Pandas](https://jovian.ml/aakashns/python-pandas-data-analysis) 
# 
# * [Analyzing your browser history using Pandas & Seaborn](https://medium.com/free-code-camp/understanding-my-browsing-pattern-using-pandas-and-seaborn-162b97e33e51) by Kartik Godawat
# 
# * [WhatsApp Chat Data Analysis](https://jovian.ml/PrajwalPrashanth/whatsapp-chat-data-analysis) by Prajwal Prashanth
# 
# * [Understanding the Gender Divide in Data Science Roles](https://medium.com/datadriveninvestor/exploratory-data-analysis-eda-understanding-the-gender-divide-in-data-science-roles-9faa5da44f5b) by Aakanksha N S
# 
# * [2019 State of Javscript Survey Results](https://2019.stateofjs.com/demographics/)
# 
# * [2020 Stack Overflow Developer Survey Results](https://insights.stackoverflow.com/survey/2020)
# 
# 
# 
# **NOTE**: Remove this cell containing the instructions before making your submission. You can do using the "Edit > Delete Cells" menu option.

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

# In[4]:


# Change this
data_dir = './life-expectancy-who'


# In[5]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[23]:


project_name = "global-life-expectancy" # change this (use lowercase letters and hyphens only)


# In[24]:


get_ipython().system('pip install jovian --upgrade -q')


# In[25]:


import jovian


# In[27]:


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


# In[2]:


led_df_raw = pd.read_csv('./life-expectancy-who/led.csv')
led_df_raw


# In[60]:


# Renaming few columns in the dataset to make them easier to read

led_df_raw.rename(columns={'thinness1-19years':'Thinness_Aged_10-19_yrs'}, inplace=True)
led_df_raw.rename(columns={'thinness5-9years':'Thinness_Aged_5-9_yrs'}, inplace=True)
led_df_raw.rename(columns={'Lifeexpectancy':'Life_Expectancy'}, inplace=True)
led_df_raw.rename(columns={'AdultMortality':'Adult_Mortality'}, inplace=True)
led_df_raw.rename(columns={'infantdeaths':'Infant_Deaths'}, inplace=True)
led_df_raw.rename(columns={'percentageexpenditure':'Percentage_Expenditure'}, inplace=True)
led_df_raw.rename(columns={'under-fivedeaths':'Under_Five_Deaths'}, inplace=True)
led_df_raw.rename(columns={'Totalexpenditure':'Total_Expenditure'}, inplace=True)
led_df_raw.rename(columns={'Incomecompositionofresources':'Income_Composition_Of_Resources'}, inplace=True)


# In[3]:


led_df_raw.info()


# In[4]:


# To find the number of missing values if present in the dataset
count_nan_in_df  = led_df_raw.isnull().sum().sum()
print ('Count of NaN: ' + str(count_nan_in_df))


# In[5]:


#Replacing NaN (Null) values with the mean value for that particular column
led_df_raw.fillna(led_df_raw.mean(axis=0,skipna=True), inplace=True)
led_df_raw


# In[6]:


count_nan_in_df  = led_df_raw.isnull().sum().sum()
print ('Count of NaN: ' + str(count_nan_in_df))


# In[7]:


#To check if any duplicate records are present
duplicateRowsDF = led_df_raw[led_df_raw.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# In[8]:


#Extract a copy of the data into a new data frame led_df so that We can continue to modify further without affecting the original data frame.
led_df=led_df_raw.copy()
led_df


# In[9]:


import jovian


# In[10]:


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

# In[11]:


import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# 1. Let's find out how many countries are involved in this analysis

# In[12]:


led_df.describe()


# In[13]:


led_df.Country.nunique()


# 2. Mean life expectancy of all developing countries throughout the years

# In[ ]:





# In[ ]:





# **TODO** - Explore one or more columns by plotting a graph below, and add some explanation about it

# In[ ]:





# In[ ]:





# **TODO** - Explore one or more columns by plotting a graph below, and add some explanation about it

# In[ ]:





# In[ ]:





# **TODO** - Explore one or more columns by plotting a graph below, and add some explanation about it

# In[ ]:





# In[ ]:





# Let us save and upload our work to Jovian before continuing

# In[25]:


import jovian


# In[106]:


jovian.commit()


# ## Asking and Answering Questions
# 
# TODO - write some explanation here.
# 
# 

# > Instructions (delete this cell)
# >
# > - Ask at least 5 interesting questions about your dataset
# > - Answer the questions either by computing the results using Numpy/Pandas or by plotting graphs using Matplotlib/Seaborn
# > - Create new columns, merge multiple dataset and perform grouping/aggregation wherever necessary
# > - Wherever you're using a library function from Pandas/Numpy/Matplotlib etc. explain briefly what it does
# 
# 

# #### Q1: TODO - ask a question here and answer it below

# In[ ]:





# In[ ]:





# In[ ]:





# #### Q2: TODO - ask a question here and answer it below

# In[ ]:





# In[ ]:





# In[ ]:





# #### Q3: TODO - ask a question here and answer it below

# In[ ]:





# In[ ]:





# In[ ]:





# #### Q4: TODO - ask a question here and answer it below

# In[ ]:





# In[ ]:





# In[ ]:





# #### Q5: TODO - ask a question here and answer it below

# In[ ]:





# In[ ]:





# In[ ]:





# Let us save and upload our work to Jovian before continuing.

# In[28]:


import jovian


# In[29]:


jovian.commit()


# ## Inferences and Conclusion
# 
# **TODO** - Write some explanation here: a summary of all the inferences drawn from the analysis, and any conclusions you may have drawn by answering various questions.

# In[30]:


import jovian


# In[31]:


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

# In[32]:


import jovian


# In[35]:


jovian.commit()


# In[ ]:




