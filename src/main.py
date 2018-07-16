
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn import datasets
import matplotlib


# In[2]:


iris = datasets.load_iris()
df = pd.DataFrame(data=iris['data'], columns=iris['feature_names'])
i = pd.Series(iris['target'])
names = i.map(lambda x: iris['target_names'][x])
df['index'] = i
df['category_name'] = names
df


# In[3]:


import edward as ed
import tensorflow as tf
from edward.models import Bernoulli, MultivariateNormalTriL, Normal
from edward.util import rbf

ys = df['index'].values
xs = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']].values
N = xs.shape[0]
D = xs.shape[1]
print("Number of data points: {}".format(N))
print("Number of features: {}".format(D))


# In[4]:


X = tf.placeholder(tf.float32, [N, D])
f = MultivariateNormalTriL(loc=tf.zeros(N), scale_tril=tf.cholesky(rbf(X)))
y = Bernoulli(logits=f)


# In[ ]:


qf = Normal(loc=tf.get_variable("qf/loc", [N]),
            scale=tf.nn.softplus(tf.get_variable("qf/scale", [N])))


# In[ ]:


inference = ed.KLqp({f: qf}, data={X: xs, y: ys})
inference.run(n_iter=5000)

