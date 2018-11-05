
# coding: utf-8

# In[59]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D   


# In[3]:


input_matrix = np.array([[1,4],[1,5],[2,4],[2,5],[3,1],[3,2],[4,1],[4,2]])
target_vector = np.array([0, 0, 0, 0, 1, 1, 1, 1])
print(target_vector)


# In[15]:


np.random.seed(42)
w1 = np.random.uniform(-2,2)
w2 = np.random.uniform(-2,2)
b = np.random.uniform(-2,2)
print(w1, w2, b)


# In[16]:


def hardlim(p1, p2, w1, w2, b):
    if(w1*p1 + w2*p2 +b < 0):
        a = 0
    else:
        a = 1
    return a


# In[17]:


for i in range(len(input_matrix)):
    a = hardlim(input_matrix[i][0], input_matrix[i][1], w1, w2, b)
    t = target_vector[i]
    e = t - a
    w1 = w1 + e*input_matrix[i][0]  
    w2 = w2 + e*input_matrix[i][1]
    b = b + e


# In[18]:


print(w1, w2, b)


# In[19]:


output_vector = []
for i in range(len(input_matrix)):
    a = hardlim(input_matrix[i][0], input_matrix[i][1], w1, w2, b)
    output_vector.append(a)

print(output_vector)


# In[79]:


feature1 = [p[0] for p in input_matrix]
feature2 = [p[1] for p in input_matrix]

df = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2, 'Class': output_vector})

p1 = [-b/w1, 0]
p2 = [0, -b/w2]
p3 = [4, (b-4*w1)/w2]

d = {'p1': [-b/w1, 0, 4], 'p2': [0, -b/w2, (-b-4*w1)/w2]}
data = pd.DataFrame(d)

fig, ax = plt.subplots()
ax.plot(data.p1, data.p2, color = "red")
l = Line2D([0,w1],[0,w2])                                    
ax.add_line(l)
ax.scatter(df.Feature1, df.Feature2, c = df.Class)

plt.show()


