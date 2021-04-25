#!/usr/bin/env python
# coding: utf-8

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

# In[10]:


# Change this
dataset_url = 'https://www.kaggle.com/mayuneko/life-expectancy-global-trend' 


# In[11]:


import opendatasets as od
od.download(dataset_url)


# The dataset has been downloaded and extracted.

# In[1]:


# Change this
data_dir = './life-expectancy-global-trend'


# In[2]:


import os
os.listdir(data_dir)


# Let us save and upload our work to Jovian before continuing.

# In[69]:


project_name = "global-life-expectancy-dev1" # change this (use lowercase letters and hyphens only)


# In[70]:


get_ipython().system('pip install jovian --upgrade -q')


# In[71]:


import jovian


# In[72]:


jovian.commit(project=project_name)


# ## Data Preparation and Cleaning
# 
# **TODO** - Write some explanation here.
# 
# 

# > Instructions (delete this cell):
# >
# > - Load the dataset into a data frame using Pandas
# > - Explore the number of rows & columns, ranges of values etc.
# > - Handle missing, incorrect and invalid data
# > - Perform any additional steps (parsing dates, creating additional columns, merging multiple dataset etc.)

# In[54]:


import pandas as pd
import numpy as np


# In[14]:


population_df = pd.read_csv('./life-expectancy-global-trend/population_total.csv')
life_expectancy_df = pd.read_csv('./life-expectancy-global-trend/life_expectancy_years.csv')
countries_raw_df = pd.read_csv('./life-expectancy-global-trend/countries_total.csv', index_col=None, header=0, engine='python')


# In[40]:


#Removing unwanted columns and records from countries file
selected_columns = ['name','region']
countries_df=countries_raw_df[selected_columns].copy()

countries_df=countries_df.drop(labels=[1],axis=0)

countries_df=countries_df.reset_index(drop=True)

#To check if any duplicate records are present
duplicateRowsDF = countries_df[countries_df.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# In[41]:


countries_df


# In[67]:


population_region_df = pd.merge(population_df,countries_df,left_on='geo',right_on='name',how='inner')
population_df


# In[ ]:





# In[ ]:





# In[36]:


import jovian


# In[ ]:


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

# **TODO** - Explore one or more columns by plotting a graph below, and add some explanation about it

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





# **TODO** - Explore one or more columns by plotting a graph below, and add some explanation about it

# In[ ]:





# In[ ]:





# Let us save and upload our work to Jovian before continuing

# In[25]:


import jovian


# In[26]:


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




