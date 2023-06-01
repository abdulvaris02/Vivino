!pip install seaborn
!pip install scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('winequalityN.csv').dropna()

def print_summarize_dataset(data_frame):
    print(f"Dataset dimension: {data_frame.shape}\n")
    print(f"First 10 rows of dataset:\n{data_frame.head(10)}\n")
    print(f"Statistical summary:\n{data_frame.describe()}")
    
print_summarize_dataset(df)

def get_test(data_frame):
    print(data_frame.corr(method = 'pearson')['alcohol'])
    
get_test(df)
# From this corrolation we can see that corrolation between alcohol and density is the highest. Let's see plot of them.

def get_plot(data_frame):
  sns.lmplot(x ='alcohol', y ='density', data = data_frame, hue ='type', markers=["o", "d"], height=6, aspect = 1.2)
  
get_plot(df)

# Let's see some other corrolations also

def get_correlation(data_frame):
    color = sns.color_palette('coolwarm', 100)
    plt.figure(figsize = (10, 8))
    sns.heatmap(data_frame.corr(), cmap = color, linewidth=0.15)
    
get_correlation(df)

# With help of this plot we can see other corollations between features of our dataset. 
# Again with help of one of the best libraries of python we can see corrolations below.
   


def get_pair_plot(data_frame):
    sns.pairplot(data_frame, hue='type', diag_kind="hist", height = 2)
    plt.show()
    
get_pair_plot(df)

# From these plots we can see some correlations: fixed acidity and pH, a little one between residual sugar and density, 
# between free sulfur dioxide and total sulfur dioxide is higher then in others. And now let's deduce these plots.
def draw_jointplot(data_frame, param_1, param_2, p_color = 'blue'):
    sns.jointplot(x = data_frame[param_1], y = data_frame[param_2], kind = 'hex', color = p_color)
    
draw_jointplot(df, 'fixed acidity', 'pH', 'red')
draw_jointplot(df, 'residual sugar', 'density', 'm')
draw_jointplot(df, 'free sulfur dioxide', 'total sulfur dioxide')

Alcohol = df.groupby(by=df['alcohol'])
def get_top_alhcohols(param):
    c = param.count()['quality'].sort_values(ascending=False).head(10)
    plt.figure(figsize=(11, 13))
    explode=len(c.values)*[0.2]
    textprops = dict(color =\"black\", fontsize=10)
    wp = { 'linewidth' : 2, 'edgecolor' : "black" }
    plt.pie(c, labels=c.index, explode=explode,autopct='%1.1f%%', wedgeprops =wp, textprops=textprops, shadow=True)
    plt.legend(prop = {'size' : 10}, loc = 'upper left')
     plt.show()
                
get_top_alhcohols(Alcohol)
                     
def sorting_for_test(df):
    del df['type']
    reviews = [] 
    for i in df['quality']:
        reviews.append('1' if i==3 else '2' if i>3 and i<6 else '3' if i>7 and i<9 else '4')
        df['reviews'] = reviews
    return df['reviews'].value_counts()
    
test_df = sorting_for_test(df)

def machine_learning(df):\n",
    "    X = df.drop(['quality', 'reviews'], axis = 1)\n",
    "    y = df['reviews']\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 40)\n",
    "    sc = StandardScaler()\n",
    "    X_train = sc.fit_transform(X_train)\n",
    "    X_test = sc.transform(X_test)\n",
    "    \n",
    "\n",
    "    model_1 = DecisionTreeClassifier()\n",
    "    model_1.fit(X_train, y_train)\n",
    "    model_1_acc = accuracy_score(y_test, model_1.predict(X_test))\n",
    "    print('%s: %f' % (\"DecisionTree\", model_1_acc.mean()), '\\n')\n",
    "\n",
    "    model_2 = KNeighborsClassifier()\n",
    "    model_2.fit(X_train, y_train)\n",
    "    model_2_acc = accuracy_score(y_test, model_2.predict(X_test))\n",
    "    print('%s: %f' % (\"KNeighbors\", model_2_acc.mean()), '\\n')\n",
    "\n",
    "    model_3 = RandomForestClassifier()\n",
    "    model_3.fit(X_train, y_train)\n",
    "    model_3_acc = accuracy_score(y_test, model_3.predict(X_test))\n",
    "    print('%s: %f' % (\"RandomForest\", model_3_acc.mean()), '\\n')\n",
    "    \n",
    "    model_4 = GradientBoostingClassifier()\n",
    "    model_4.fit(X_train, y_train)\n",
    "    model_4_acc = accuracy_score(y_test, model_4.predict(X_test))\n",
    "    print('%s: %f' % (\"GradientBoosting\", model_4_acc.mean()), '\\n')\n",
    "\n",
    "    model_5 = SVC()\n",
    "    model_5.fit(X_train, y_train)\n",
    "    model_5_acc = accuracy_score(y_test, model_5.predict(X_test))\n",
    "    print('%s: %f' % (\"SVC\", model_5_acc.mean()), '\\n\\nRESULT')
         models = pd.DataFrame({'Model' : ['DecisionTree', 'KNeighbors', 'RandomForest',  'GradientBoosting', 'SVC'], 'Score' : [model_1_acc, model_2_acc, model_3_acc, model_4_acc, model_5_acc]})
    return models
