#!/usr/bin/env python
# coding: utf-8

# # Polynomial linear regression 
# 
# 
# y= a + b1x + b2(x^2) + b3(x^3) + ....bn+(x^n)

# bir ara yönetici tanımlayalım ve bu yöneticinin seviyesi region manager ile country arasında olsun yani 4.5
# 

# In[13]:


import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df= pd.read_csv("polynomial.csv", ";")


# In[15]:


df


# In[16]:


plt.scatter(df['deneyim'],df['maas'])

plt.xlabel('Deneyim(yıl)')
plt.ylabel('Maaş')

plt.show()


# In[18]:


#veriler dogrusal yapıda dağılmıyorsa ve biz bu veri setine lineer model uygularsak hiç uygun olmayan bir tahmin çizgisi görürüz

reg= LinearRegression()
reg.fit(df[['deneyim']], df['maas'])

plt.xlabel('Deneyim(yıl)')
plt.ylabel('Maaş')

plt.scatter(df['deneyim'],df['maas'])

x_ekseni=df['deneyim']
y_ekseni=reg.predict(df[['deneyim']])

plt.plot(x_ekseni, y_ekseni, color="green", label="Linear Regression")
plt.legend()
plt.show()


# Polynomial regression uygulamaya karar verdik.

# In[26]:



polynomial_regression= PolynomialFeatures(degree = 4)
x_polynomial = polynomial_regression.fit_transform(df[['deneyim']])

reg= LinearRegression()
reg.fit(x_polynomial, df['maas'])


# Model artık hazır ve eğitilmiş, eldeki verilere göre modelimiz nasıl bir sonuç grafiği oluşturuyor görelim.

# In[27]:


y_head= reg.predict(x_polynomial)
plt.plot(df['deneyim'],y_head, color="red", label="polynomial regression ")
plt.legend()
plt.scatter(df['deneyim'],df['maas'])
plt.show()


# Simdi n=3 ve n=4 icin yapmaya calısalım. En iyi sonucu 4 ile bulduk. peki neden hep n degerini en yuksek secmemeliyiz?
# İslemciyi yormamak icin n icin zamandan tasarruf saglayacak ve cpu yormayacak en uygun degeri secmeliyiz

# In[28]:


x_polynomial1=polynomial_regression.fit_transform([[4.5]])
reg.predict(x_polynomial1)


# In[ ]:




