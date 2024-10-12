# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:32:08 2024

@author: ishma
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #for step 4

#Step 1

df = pd.read_csv("Project_1_Data.csv") #no need for file path since its in the downloads folder
print(df.info()) #check that line 12 actually worked

#Step 2
plt.figure()
plt.scatter(df['Step'], df['X'], color='red', label = 'X')

plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Scatter Plot of X vs. Step')


plt.figure()

plt.scatter(df['Step'], df['Y'], color='green', label = 'Y')
plt.scatter(df['Step'], df['Z'], color='blue', label = 'Z')

plt.xlabel('Step')
plt.ylabel('Values')
plt.title('Scatter Plot of Y and Z vs. Step') #Separate plots for X and Y,Z because of overlap making X impossible to see
plt.legend(loc='upper right')

#its clear that Z and X have a wide range of values at each step
avg_z = df.groupby('Step')['Z'].mean()

plt.figure()
plt.plot(avg_z.index,avg_z.values)
plt.xlabel('Step')
plt.ylabel('Average Z')
plt.title('Average Z at Each Step')

avg_x = df.groupby('Step')['X'].mean()

plt.figure()
plt.plot(avg_x.index,avg_x.values)
plt.xlabel('Step')
plt.ylabel('Average X')
plt.title('Average X at Each Step')
