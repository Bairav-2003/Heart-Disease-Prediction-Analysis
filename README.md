# Heart-Disease-Prediction-Analysis
~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("/content/Heart.csv")

df.head()

df.info()

df.describe()

df1=df.copy()

def chng(sex):
    if sex == 0:
        return 'female'
    else:
        return 'male'
df1['sex'] = df1['sex'].apply(chng)

def chng2(prob):
    if prob == 0:
        return 'Heart Disease'
    else:
        return 'No Heart Disease'
df1['target'] = df1['target'].apply(chng2)

sns.countplot(x='sex',hue='target',data= df1)
plt.title('Gender v/s target\n')


sns.countplot(x='cp',hue='target',data=df1)
plt.title("chest pain vs heart disease")

sns.barplot(x='target',y='trestbps',data=df1)

df1.loc[df1['cp'] == 0, 'cp'] = 'asymptomatic'
df1.loc[df1['cp'] == 1, 'cp'] = 'atypical angina'
df1.loc[df1['cp'] == 2, 'cp'] = 'non anginal pain'
df1.loc[df1['cp'] == 3, 'cp'] = 'typical angina'

sns.countplot(x='cp',hue='target',data=df1)
plt.title("chest pain vs heart disease")

sns.barplot(y='chol',x='target',data=df1)
sns.countplot(x='fbs',hue='target',data=df1)
plt.title("fasting blood sugar vs heart disease")

sns.countplot(x='restecg',hue='target',data=df1)
plt.title("fasting blood sugar vs heart disease")

sns.barplot(x='target',y='thalach',data=df1)

sns.barplot(x='target',y='exang',data=df1)

sns.barplot(x='target',y='ca',data=df1)

sns.countplot(x='thal',hue='target',data=df1)

x= df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal']]
y= df['target']

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,)


from sklearn.linear_model import LinearRegression
le = LinearRegression()
le.fit(x_train,y_train)
y_pred = le.predict(x_test)

from sklearn import metrics
MSE = metrics.mean_squared_error(y_test,y_pred)
print("MSE is {}".format(MSE))
r2 = metrics.r2_score(y_test,y_pred)
print("R Squared Error is {} ".format(r2))

print("enter values of certain parameters: chest pain\t resting bp\tcholestrol\tfasting blood sugar\te_induced angina\tmajor vessels\tthalassemia")

outcome=le.predict([[41,0,1,130,204,0,0,172,0,1.4,2,0,2]])

print(outcome[0])

if(outcome[0]>0.5000):
  print("the person have higher chances of getting heart disease")
else:
  print("the person have very less or no chances of getting heart disease")
~~~



