#importing the modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.api as sm
from scipy import stats
#loading the dataset
df=pd.read_csv("/content/c4_epa_air_quality.csv")
df.head()
#table of descriptive statistics for all the columns
df.describe(include = "all")
#Sample with replacement
sample_data = df.sample(n = 50,replace = True,random_state = 42)
sample_data.head(10)
#applying the mean function on the sample data
sample_data["aqi"].mean()
#sample and population mean is different
#Applying central limit theorem
list1 = []
for i in range(1000):
  list1.append(df["aqi"].sample(n =50,replace =True).mean())
list1
#applying the mean function on the datset
np.mean(list1)
#sample and population mean is same after following central limit theorem
#creating a sampled dataframe 
sample_df = pd.DataFrame(list1)
sample_df["mean"] =sample_df[0]
sample_df
#dropping the first column with index 0
sample_df.drop(columns = 0, inplace =True)
sample_df
#Create a hsitogram to evaluate the mean distribution graph
plt.figure(figsize = (10,5))
plt.hist(sample_df["mean"])
plt.axvline(np.mean(sample_df["mean"]), color = "red")
plt.show()
# Evaluating the standard error
error = sample_data["aqi"].std()/np.sqrt(len(sample_data))
error
# The relationship between the sampling and normal distributions
xmin,xmax = plt.xlim()
x = np.linspace(xmin,xmax,100)
y =stats.norm.pdf(x, sample_data["aqi"].mean(),error)
plt.plot(x,y)
plt.axvline(error.mean(),color = "green")
plt.show()
