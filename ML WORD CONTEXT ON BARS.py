#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[5]:


#get data and train-test split 80-20

start = '2003-01-01'
end = '2018-01-01'

data = get_pricing('spy', start_date = start, end_date = end , fields = 'price').pct_change()[1:]

train_pct = 0.1
train = data.iloc[:-int(train_pct*len(data))]
test = data.iloc[-int(train_pct*len(data)):]


# In[6]:


#check that train and test dont overlap
print(train.tail())
print(test.head())


# In[7]:



plt.hist(test, bins = 20)
#plot histogram for various subsets of train of the same size as test to compare distributions
for i in range(int(len(train)/len(test))):
    alpha_tune = float(1)/int(len(train)/len(test))
    plt.hist(train[i* len(test):(i+1)*len(test)], bins = 20, alpha = i* alpha_tune)


print(alpha_tune)


# It seems that they do not have identical distributions afterall. Check if code is running as expected, are there larger sample sizes in play?,save eatch of their distributions datao n a table e.g pandas and then plot

# In[8]:


subset_stats = pd.DataFrame()

for i in range(int(len(train)/len(test))):
    subset_stats[i] = train[i* len(test):(i+1)*len(test)].describe()


# In[9]:


subset_stats


# the above implies that they have eqal sample sizes

# In[10]:




for i in xrange(subset_stats.shape[1]):
    subset = subset_stats[i]
    mu = subset['mean']
    stdev = subset['std']
    variance = stdev ** 2
    def normalise(subset):
        for i in range(len(subset)):
            subset[i] = (subset[i]-mu)/ stdev
        return subset
        
    subset = normalise(subset)
    mean = subset['mean']
    sigma = subset['std']
    variance = sigma ** 2
            
    x = np.linspace(mean - 3*sigma, mean + 3*sigma, 100)
    f = np.exp(-np.square(x-mean)/2*variance)/(np.sqrt(2*np.pi*variance))
    f = f -np.mean(f)
    #print(f)
    plt.plot(f)


# showsthat they atuall do have similar distributions

# In[11]:


subset_stats[1]['mean']


# In[12]:


type(train)


# back to application of word context to returns

# In[13]:


train = data.iloc[:-int(train_pct*len(data))]
train = pd.Series(list(train)+ list(test))
print(train.shape)

def discretise(train):
    
    mean = train.mean()
    std = train.std()
    disc_train = pd.DataFrame(np.zeros((train.shape)))

    for i in range(len(train)):
        disc_train[0][i] = np.ceil(train[i]/std)
    
    return disc_train




disc_train = discretise(train)

np.min(disc_train)
np.max(disc_train)

disc_train.shape

size_of_context = 21





def generate_context(train, size = 7):
    shaper = np.zeros((len(train), size))
    #print(shaper.shape)
    context = pd.DataFrame(shaper)
    #print(context.shape)
    #print(train.shape)
    for i in range(int(size/2),len(train)-int(size/2)):
        #print(i)
        context.iloc[i][0] = train.iloc[i][0]
        for j in range(np.int(size/2)):
            #print(j)
            context.iloc[i][j] = train.iloc[i+j][0]
        for j in range(1,np.int(size/2)):
            #print(j)
            context.iloc[i][j+np.int(size/2)+1]= train.iloc[i-j][0]
            
    return context

context = generate_context(disc_train, size_of_context)[int(size_of_context/2):]


# In[ ]:





# In[14]:


context.shape


# In[29]:


plt.hist(context);

#pd.DataFrame(np.array(context[context>0]).reshape(context.shape[0]*context.shape[1])).dropna().mean()


# In[122]:


def retrieve_interest(context, min_sd= 2):#min_sd refers to the lower limit of day returns' context to retrieve
    interest = pd.DataFrame(np.ones(context.shape)*1000)
    for i in range(len(context)):
        if min_sd > 0:
            if context.iloc[i][0]>= min_sd:
                interest.iloc[i] = context.iloc[i]
                #print context.iloc[i][0]
        elif min_sd < 0:
            if context.iloc[i][0]<= min_sd:
                interest.iloc[i] = context.iloc[i]
                
            
    return interest



    
interest = retrieve_interest(context,1)
interest = interest[interest != 1000]
interest


# In[ ]:





# In[123]:


from collections import Counter

interest = (interest[interest !=0]*100)

mean = np.zeros(interest.shape[1])
median = np.zeros(interest.shape[1])
mode = np.zeros(interest.shape[1])
for i in range(interest.shape[1]):
    #print i
    mean[i] = np.mean(pd.DataFrame(np.array(interest.iloc[:][i]).reshape(len(interest))).dropna())
    median[i] = np.median(pd.DataFrame(np.array(interest.iloc[:][i]).reshape(len(interest))).dropna())
    data = Counter(interest.iloc[:][i])
    counts = data.most_common
    mode[i] = data.most_common(1)[0][0]
    




# In[1]:


#print(mean)
#mean = mean[mean!=0]
#plt.stem(range(0,len(mean)),np.cumsum(mean))
#plt.plot(np.cumsum(pd.DataFrame(mean).pct_change()))
#plt.plot(mean[1:])
#mean is less informative as it eradicates tthe effect of -ve values 
print(median)
median[10:12] = median[0]
median[0] = 0
plt.plot(np.cumsum(median))
print(mode)
mode[10:12] = mode[0]
mode[0] = 0
plt.plot(mode)


# In[ ]:


mean


# In[ ]:


import seaborn as sb


# In[ ]:





# In[ ]:





# In[ ]:





# #leave interets for now. Perform decision trees on context using context day returns as features and returns

# In[2]:


from sklearn.tree import DecisionTreeClassifier


def rediscretise(y):
    #y = [1 if yy >0 else 0 for yy in y]
    yy = []#np.array(len(y))
    for i in y:#range(len(y)):
        if i > 0:
            i = 1
        elif i == 0:
            i = 0
        elif i < 0:
            i = -1
        yy.append(i)
    yy = np.array(yy)
    return yy

wait_days  = 3 #defines number of days to hold
n_samp = np.int(context.shape[0]/1.7)
n_feat = np.int(context.shape[1]/2)+1
print(n_feat)
sub_context1 = context.iloc[:n_samp][:]#top part
#print(sub_context1.shape)
X1 = sub_context1.iloc[:,:n_feat]#left features
print(X1.shape)
y1_raw = sub_context1.iloc[:,n_feat+1:n_feat+1  + wait_days]#right features
#print(y1.shape)
y1 = np.sum(y1_raw, axis = 1)
y1_analysis = y1
print(len(y1[y1>0]),'!')
y1 = np.array(y1)
#y1 = rediscretise(y1)




sub_context2 = context.iloc[:context.shape[0]][:]
#print(sub_context1.shape)
X2 = sub_context2.iloc[:,:n_feat]#left features
print(X2.shape)
y2_raw = sub_context2.iloc[:,n_feat:n_feat + wait_days]#right features
#print(y2.shape)
y2 = np.sum(y2_raw, axis = 1)
#print(y2)
y2 = np.array(y2)
#y2 = rediscretise(y2)






# In[3]:


for i in range(len(y1)):
    r[i] = y1[i]* np.sum(data[i+n_feat+1:i+n_feat +1 + wait_days])
#plt.plot(np.cumsum(r))
print y1
plt.plot(np.cumsum(r))
plt.figure()
plt.plot(np.cumsum(data))


# In[4]:


plt.hist(y1_analysis, bins = 20);
float(len(y1_analysis[y1_analysis < 0]))/ len(y1_analysis)


# In[5]:


from sklearn.decomposition import PCA

max_depth = 40 #max depth of tree
errnp = np.zeros((max_depth,2))
errors = pd.DataFrame(errnp)
hit_rates = pd.DataFrame(errnp, columns = ['train', 'test'])
#print(errors)
for i in xrange(1,max_depth):
    pca_model = PCA(n_components = 5)
    pca_X1 = pca_model.fit(X1).transform(X1)
    pca_X2 = pca_model.fit(X2).transform(X2)

    model = DecisionTreeClassifier(max_depth = i)
    model = model.fit(pca_X1,y1)
    y1_hat = model.predict(pca_X1)
    y2_hat = model.predict(pca_X2)
    RMSE_1 = np.sqrt(sum((y1 - y1_hat)**2))/len(y1)
    RMSE_2 = np.sqrt(sum((y2 - y2_hat)**2))/len(y2)
    hit_rate1 = float((np.sum(y1 == y1_hat)))/ len(y1)
    hit_rate2 = float((np.sum(y2 == y2_hat)))/ len(y2)
    #print(hit_rate1, hit_rate2)
    #print(RMSE_1, RMSE_2)
    errors.iloc[i][0] = RMSE_1
    errors.iloc[i][1] = RMSE_2
    hit_rates.iloc[i][0] = hit_rate1
    hit_rates.iloc[i][1] = hit_rate2
    #print(i)
hit_rates.head(5)
hit_rates.tail(5)
#print(hit_rates.loc[:]['train'])
#plt.plot(hit_rates.iloc[:]['train'], hit_rates.iloc[:]['test'])
plt.figure()
plt.plot(hit_rates)
#plt.legend(hit_rates.columns)


# In[ ]:





# In[6]:


from sklearn.linear_model import LogisticRegression

LR_model = LogisticRegression()
LR_model = LR_model.fit(X1,y1)
y1_hat = LR_model.predict(X1)
y2_hat = LR_model.predict(X2)

RMSE_1 = np.sqrt(sum((y1 - y1_hat)**2))/len(y1)
RMSE_2 = np.sqrt(sum((y2 - y2_hat)**2))/len(y2)
hit_rate1 = float((np.sum(y1 == y1_hat)))/ len(y1)
hit_rate2 = float((np.sum(y2 == y2_hat)))/ len(y2)
#print(hit_rate1, hit_rate2)
#print(RMSE_1, RMSE_2)

print(hit_rate1, hit_rate2)
print(RMSE_1, RMSE_2)
print(X1.shape)


# In[ ]:


DT_model = DecisionTreeClassifier(max_depth = 10)
DT_model = DT_model.fit(X1,y1)
y1_hat = DT_model.predict(X1)
y2_hat = DT_model.predict(X2)

RMSE_1 = np.sqrt(sum((y1 - y1_hat)**2))/len(y1)
RMSE_2 = np.sqrt(sum((y2 - y2_hat)**2))/len(y2)
hit_rate1 = float((np.sum(y1 == y1_hat)))/ len(y1)
hit_rate2 = float((np.sum(y2 == y2_hat)))/ len(y2)
#print(hit_rate1, hit_rate2)
#print(RMSE_1, RMSE_2)

print(hit_rate1, hit_rate2)
print(RMSE_1, RMSE_2)


# In[7]:


from sklearn.decomposition import PCA

pca_model = PCA(n_components = 5)
pca_X1 = pca_model.fit(X1).transform(X1)
pca_X2 = pca_model.fit(X2).transform(X2)

model = DecisionTreeClassifier(max_depth = 3)
model = model.fit(pca_X1,y1)
pca_y1_hat = model.predict(pca_X1)
pca_y2_hat = model.predict(pca_X2)

RMSE_1 = np.sqrt(sum((y1 - y1_hat)**2))/len(y1)
RMSE_2 = np.sqrt(sum((y2 - y2_hat)**2))/len(y2)
pca_hit_rate1 = float((np.sum(y1 == pca_y1_hat)))/ len(y1)
pca_hit_rate2 = float((np.sum(y2 == pca_y2_hat)))/ len(y2)

print(pca_hit_rate1, pca_hit_rate2)
print(RMSE_1, RMSE_2)




# In[ ]:





# #TEST THE MODEL ON NEW DATA OF A DIFFERENT STOCK

# In[8]:


start = '2003-01-01'
end = '2018-01-01'

data = get_pricing('msft', start_date = start, end_date = end, fields = 'price').pct_change()[1:]


# In[9]:


disc_train = discretise(data)
context = generate_context(disc_train, size_of_context)[int(size_of_context/2):]


# In[ ]:


def rediscretise(y):
    #y = [1 if yy >0 else 0 for yy in y]
    yy = []#np.array(len(y))
    for i in y:#range(len(y)):
        if i > 0:
            i = 1
        elif i == 0:
            i = 0
        elif i < 0:
            i = -1
        yy.append(i)
    yy = np.array(yy)
    return yy

wait_days  = 3 #defines number of days to hold
n_samp = np.int(context.shape[0]/1.7)
n_feat = np.int(context.shape[1]/2)+1
print(n_feat)
sub_context1 = context
#print(sub_context1.shape)

X1 = sub_context1.iloc[:,:n_feat]#left features
print(X1.shape)
y1_raw = sub_context1.iloc[:,n_feat:n_feat+1 + wait_days+1]#right features
#print(y1.shape)
y1 = np.sum(y1_raw, axis = 1)
y1_analysis = y1
print(len(y1[y1>0]),'!')
y1 = np.array(y1)
#y1 = rediscretise(y1)


# In[ ]:





DT_model = DT_model.fit(X1,y1)
y1_hat = DT_model.predict(X1)


RMSE_1 = np.sqrt(float(sum((y1 - y1_hat)**2))/len(y1))

hit_rate1 = float((np.sum(y1 == y1_hat)))/ len(y1)


print 'test hit rate: ', hit_rate1
#print('test RMSE: ', RMSE_1)


# In[ ]:


"""pca_model = PCA(n_components = 5)
pca_X1 = pca_model.fit(X1).transform(X1)


#DT_model = DecisionTreeClassifier(max_depth = 5)
DT_model = DT_model.fit(pca_X1,y1)
y1_hat = DT_model.predict(pca_X1)


RMSE_1 = np.sqrt(sum((y1 - y1_hat)**2))/len(y1)

pca_hit_rate1 = float((np.sum(y1 == pca_y1_hat)))/ len(y1)
"""


# In[ ]:


def get_returns(y_hat,y, data):
    returns = np.zeros(data.shape)
    for i in xrange(len(y_hat)):
        returns[i] = y_hat[i]*(np.sum(data[i+11:i+11+wait_days]))
    return returns

returns = get_returns(y1_hat, y1, data)
rets = returns#[returns !=0]
plt.plot(np.cumsum(rets))
plt.figure()
plt.plot(data[10:])


#STOPPED HERE>>> ERROR IN CODE? COMPILER? TURNING AN ARRAY INTO AN INT UNWANTED


# In[ ]:


len(y1_hat[y1_hat != -1])


# In[ ]:


plt.plot(np.cumsum(data))


# In[ ]:


plt.hist(y1,bins =30);


# In[ ]:


plt.hist(y1_hat, bins = 30);


# In[ ]:


data.head(20)/ data.std()


# In[ ]:


context[0][10]


# In[ ]:


plt.hist(data, bins = 10)[1]# use this to reform disc_train - dicretise function


# In[ ]:


context


# In[ ]:


data


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




