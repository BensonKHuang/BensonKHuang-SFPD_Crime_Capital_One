
# coding: utf-8

# In[1]:


# Basic Libraries
import numpy as np
import pandas as pd

# Feature Scaling
from sklearn.preprocessing import RobustScaler

# Visaulization
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# Classifier (machine learning algorithm)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import LeavePOut
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV

# Evaluation
from sklearn.model_selection import cross_val_score, cross_val_predict

# Parameter Tuning
from sklearn.model_selection import GridSearchCV

# Settings
pd.options.mode.chained_assignment = None # Stop warning when use inplace=True of fillna

train = pd.read_csv("sfpd-dispatch/sfpd_dispatch_data_subset.csv", index_col=0)
#train.head()
# test = pd.read_csv("test_final.csv", index_col=0)


# In[2]:


train.describe()


# In[3]:


train.isnull().sum()


# In[4]:


# This fills all the nan spots with the text 'nan'
train = train.fillna('nan')


# In[5]:


# takes about 4 minutes
#Dictionary the values for processing

train['call_type'] = train['call_type'].map({'Medical Incident' : 0, 'Alarms' : 1, 'Structure Fire' : 2, 'Traffic Collision' : 3, 'Outside Fire' : 4, 'Other' : 5, 'Train / Rail Incident' : 6, 'Citizen Assist / Service Call' : 7, 'Electrical Hazard' : 8, 'Elevator / Escalator Rescue' : 9, 'Fuel Spill' : 10, 'Gas Leak (Natural and LP Gases)' : 11, 'Water Rescue' : 12, 'Vehicle Fire' : 13, 'Smoke Investigation (Outside)' : 14, 'Odor (Strange / Unknown)' : 15, 'HazMat' : 16 }).astype(int)
train['call_final_disposition'] = train['call_final_disposition'].map({'Code 2 Transport' : 0, 'Fire' : 1, 'Code 3 Transport' : 2, 'Patient Declined Transport' : 3, 'No Merit' : 4, 'Medical Examiner' : 5, 'Against Medical Advice' : 6, 'Cancelled' : 7, 'Unable to Locate' : 8, 'SFPD' : 9, 'Gone on Arrival' : 10, 'Other' : 11}).astype(int)
train['city'] = train['city'].map({'nan' : 0, 'San Francisco' : 1, 'Presidio' : 2, 'Treasure Isla' : 3, 'Yerba Buena' : 4, 'Hunters Point' : 5, 'Fort Mason' : 6 }).astype(int)
train['battalion'] = train['battalion'].map({'B99': 0, 'B01': 1, 'B02': 2, 'B03': 3, 'B04': 4, 'B05': 5, 'B06': 6, 'B07': 7, 'B08': 8, 'B09': 9, 'B10': 10 }).astype(int)
train['original_priority'] = train['original_priority'].map({'I' : 1, '2' : 2, '3' : 3, 'A' : 4, 'B' : 5, 'C' : 6, 'E' : 7}).astype(int)
train['priority'] = train['priority'].map({'I' : 1, '2' : 2, '3' : 3, 'E' : 4}).astype(int)
train['als_unit'] = train['als_unit'].map({True : 1, False : 0}).astype(int)
train['call_type_group'] = train['call_type_group'].map({'nan' : 0, 'Non Life-threatening' : 1, 'Potentially Life-Threatening' : 2, 'Alarm' : 3, 'Fire' : 4}).astype(int)

#Create a travel duration column
for k in range(0,train.shape[0]):
    train['call_date'][k] = int(train['call_date'][k][2:4])
    train['watch_date'][k] = int(train['watch_date'][k][2:4])
    
    timestamps = ['received_timestamp','entry_timestamp','dispatch_timestamp','response_timestamp','on_scene_timestamp','transport_timestamp','hospital_timestamp','available_timestamp']
    
    for stamp in timestamps:
        if train[stamp][k]=='nan':
            train[stamp][k] = 0
        else:
            train[stamp][k] = int(train[stamp][k][17:19]) + 60 * int(train[stamp][k][14:16]) + 3600 * int(train[stamp][k][11:13]) + 3600 * 24 * int(train[stamp][k][8:10])

train['unit_type'] = train['unit_type'].map({'ENGINE' : 0, 'MEDIC' : 1, 'PRIVATE' : 2, 'TRUCK' : 3, 'CHIEF' : 4, 'RESCUE CAPTAIN' : 5, 'RESCUE SQUAD' : 6, 'SUPPORT' : 7, 'INVESTIGATION' : 8 }).astype(int)


# In[6]:


#remove noisy data such as neighbouhood_district and call_number for faster processing
drop_elements=['call_number','incident_number','call_date','watch_date','neighborhood_district','zipcode_of_incident','station_area','address','city','location','supervisor_district']
train = train.drop(drop_elements, axis = 1)


# In[7]:


#encode missing times as 99999, will be used for checks
train['travel_duration'] = train['on_scene_timestamp']-train['dispatch_timestamp']
for k in range(0, train.shape[0]):
    temp = int(train['travel_duration'][k])
    if (temp > 1500):
        train['travel_duration'][k] = 99999
    if (temp <= 0):
        train['travel_duration'][k] = 99999


# In[8]:


#set bins for graph
ranges = list(np.arange(0,2000,60))
incidences_per_unit_type = pd.DataFrame(train.groupby(pd.cut(train.travel_duration, ranges)).count()["unit_type"])
incidences_per_unit_type.columns = ["count"]

#set parameters for plot
ax = incidences_per_unit_type.plot.bar(figsize=(15,12), fontsize = 12, legend = False, color = "palegreen")#counts per elevation
ax.set_xlabel("Time to Arrive (s)", fontsize = 16)
ax.set_ylabel("Count of Fire Department Calls", fontsize = 16)
ax.set_title("Count of Fire Department Calls to Response Time", fontsize = 20)
plt.savefig("bin_check.png")


# In[9]:


def get_cat(col, cat):
    for token in col:
        if(token > 2000):
            cat.append('Other')
        else:
            chunks = []
            temp = int(int(token)/60)
            min = temp * 60
            max = min + 60
            chunks = ['(', str(min), ', ', str(max), ']']
            category = ''.join(chunks)
            cat.append(str(category))
            
Response_time = []

get_cat(train["travel_duration"].tolist(), Response_time)
train["response_cat"] = Response_time

train['response_cat']


# In[10]:


#set minute categories
train['response_cat'] = train['response_cat'].map({'(0, 60]' : 1, '(60, 120]' : 2, '(120, 180]' : 3, '(180, 240]' : 4, '(240, 300]' : 5, '(300, 360]' : 6, '(360, 420]' : 7, '(420, 480]' : 8, '(480, 540]' : 9, '(540, 600]' : 10, '(600, 660]' : 11, '(660, 720]' : 12, '(720, 780]' : 13, '(780, 840]' : 14, '(840, 900]' : 15, '(900, 960]' : 16, '(960, 1020]' : 17, '(1020, 1080]' : 18, '(1080, 1140]' : 19, '(1140, 1200]' : 20, '(1200, 1260]' : 21, '(1260, 1320]' : 22, '(1320, 1380]' : 23, '(1380, 1440]' : 24, '(1440, 1500]' : 25, '(1500, 1560]' : 26, 'Other' : 0 }).astype(int)


# In[11]:


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(16,12))
incidence_count_matrix_long = pd.DataFrame({'count' : train.groupby( [ "temp","response_cat"] ).size()}).reset_index()
incidence_count_matrix_pivot = incidence_count_matrix_long.pivot("temp","response_cat","count") 
ax = sns.heatmap(incidence_count_matrix_pivot, annot=False, fmt="d", linewidths=1, square = False, cmap="YlGnBu")
ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.xlabel("Response Time (minutes)", fontsize = 24, color="steelblue")
ax = plt.ylabel("Resource Dispatched", fontsize = 24, color="steelblue")
ax = plt.title("Count of Dispatch to Response Time", fontsize = 24, color="steelblue")
plt.savefig('Dispatch_response.png')


# In[12]:


train_original = pd.read_csv("sfpd-dispatch/sfpd_dispatch_data_subset.csv", index_col=0)


# In[13]:


def get_day_time(col, days, hours, months, years):
    for token in col:
        day = int(token.split()[0].split("-")[2])
        month = int(token.split()[0].split("-")[1])
        year = int(token.split()[0].split("-")[0])
        hour = int(token.split()[1].split(":")[0])
        days.append(day) 
        months.append(month)
        years.append(year)
        hours.append(hour)
        
hours = []
days = []
months = []
years = []

get_day_time(train_original["received_timestamp"].tolist(), days, hours, months, years)
train_original["received_hour"] = hours
train_original["received_day"] = days
train_original["received_month"] = months
train_original["received_year"] = years


# In[14]:


get_ipython().magic('matplotlib inline')
plt.figure(figsize=(16,12))
incidence_count_matrix_long = pd.DataFrame({'count' : train_original.groupby( [ "received_hour","received_day"] ).size()}).reset_index()
incidence_count_matrix_pivot = incidence_count_matrix_long.pivot("received_hour","received_day","count") 
ax = sns.heatmap(incidence_count_matrix_pivot, annot=False, fmt="d", linewidths=1, square = False, cmap="YlGnBu")
ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.xlabel("Day", fontsize = 24, color="steelblue")
ax = plt.ylabel("Hour", fontsize = 24, color="steelblue")
ax = plt.title("Count of Calls (Day x Hour)", fontsize = 24, color="steelblue")
plt.savefig("call_day.png")


# In[15]:


train.head()


# In[42]:


def get_hour(col, hours):
    for token in col:
        hour = int(token.split()[1].split(":")[0])
        hours.append(hour)
        
#knn_data_subset.csv is a cleaned up data set only containing unit_type, latitude, longitude, and received_timestamp
hours = []
train_knn = pd.read_csv("sfpd-dispatch/knn_data_subset.csv", index_col=0)
get_hour(train_knn["received_timestamp"].tolist(), hours)
train_knn["hour"] = hours


# In[43]:


y = train_knn.unit_type
train_knn = train_knn.drop("received_timestamp", axis = 1)
train_knn = train_knn.drop("unit_type", axis = 1)
X = train_knn
X.head()


# In[163]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.8)


# In[164]:


print ("\nX_train:\n")
print(X_train.head())
print (X_train.shape)
print ("\nX_test:\n")
print(X_test.head())
print (X_test.shape)


# In[170]:


#odd number, cho
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def get_k(X_train, y_train, X_test, k, acc):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    acc.append(accuracy_score(y_test, predictions))


# In[171]:


k = []
acc = []
for i in range(2, 50):
    get_k(X_train, y_train, X_test, i, acc)
    k.append(i)
print(k)
print(acc)


# In[178]:


plt.figure(figsize=(12,9))
plt.scatter(k,acc)
ax = plt.xticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.yticks(fontsize = 12,color="steelblue", alpha=0.8)
ax = plt.xlabel('K (# of nearest neighbors)', fontsize = 24, color="steelblue")
ax = plt.ylabel('Accuracy to test data', fontsize = 24, color="steelblue")
ax = plt.title('Finding K in KNN Classifier', fontsize = 24, color="steelblue")
plt.grid(True)
plt.savefig("knn.png")


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

mapdata = np.loadtxt("sfpd-dispatch/sf_map_copyright_openstreetmap_contributors.txt")
asp = mapdata.shape[0] * 1.0 / mapdata.shape[1]
lon_lat_box = (-122.5247, -122.3366, 37.699, 37.8299)
clipsize = [[-122.5247, -122.3366],[ 37.699, 37.8299]]


# In[18]:


# trainP = train[train.call_type == 'Medical Incident']
np.seterr(divide='ignore', invalid='ignore')
trainP = train_original.dropna()


# In[179]:


import warnings
warnings.filterwarnings("ignore")
g= sns.FacetGrid(train_original, col="call_type", col_wrap=4, size=5, aspect=1/asp)

#Show the background map
for ax in g.axes:
    ax.imshow(mapdata, cmap=plt.get_cmap('gray'),alpha = 0.6, 
              extent=lon_lat_box, 
              aspect=asp)
#Kernel Density Estimate plot
sns.kdeplot(train_original.longitude, train_original.latitude, shade = True)
g.map(sns.kdeplot, "longitude", "latitude", clip=clipsize)
plt.savefig('call_density_plot.png')


# In[ ]:


#Create a gif using the 25 images created to show heatmap over response time!

my_list=list(range(1,27))

for x in range(1, 27):
    trainP = train[train['response_cat'] == x]
    plt.figure(figsize=(20,20*asp))
    plt.title('Response within ' + str(x) + ' minutes', fontsize = 48, color="steelblue")
    try:
        ax = sns.kdeplot(trainP.longitude, trainP.latitude, clip=clipsize)
    except ZeroDivisionError:
        print("\n")
    ax.imshow(mapdata, cmap=plt.get_cmap('gray'), 
                extent=lon_lat_box)
    plt.savefig(str(x)+'_density_plot.png')


# In[27]:


sum = 0
count = 0
for i in train['travel_duration'].tolist():
    if(i > 1500):
        continue
    else:
        sum += i
        count += 1
print("avg response time is " + str(sum/count) + "seconds")


# In[28]:


sum = 0
count = 0
trainc = train[train.unit_type == 0]
for i in trainc['travel_duration'].tolist():
    if(i > 1500):
        continue
    else:
        sum += i
        count += 1
print("avg engine response time is " + str(sum/count) + "seconds")


# In[29]:


sum = 0
count = 0
trainc = train[train.unit_type == 1]
for i in trainc['travel_duration'].tolist():
    if(i > 1500):
        continue
    else:
        sum += i
        count += 1
print("avg medic response time is " + str(sum/count) + "seconds")

