import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

data=pd.read_csv("Bengaluru_House_Data.csv")

#checking FUNC
def hmap(data):
    sb.heatmap(data.isnull())
    plt.show()
def csv(data):
    d=data.iloc[:,:]
    print(d)

    
#filling empty and removing unwanted col 
data["balcony"]=data["balcony"].fillna(0)
data.drop(["society","location","availability"],axis=1,inplace=True)
data=data.replace(to_replace=r'^[0-9]+.*[0-9]*\s-\s[0-9]*.*[0-9]*', value=np.nan, regex=True)
data=data.replace(to_replace=r'^[0-9]+.*[0-9]*[A-z,a-z]+.*[A-z,a-z]*', value=np.nan, regex=True)
data.dropna(inplace=True)#remove row
#hmap(data)


categorical_columns = ['area_type']
for column in categorical_columns:
    temp = pd.get_dummies(data[column], prefix=column)
    data = pd.merge(
        left=data,
        right=temp,
        left_index=True,
        right_index=True,
    )
    data = data.drop(columns=column)
data.to_csv('bng.csv',index = False)

data=data.reindex(columns=['area_type_Built-up  Area','area_type_Carpet  Area','area_type_Plot  Area','area_type_Super built-up  Area','total_sqft','size','bath','balcony','price'])
#Training


x=data.loc[:,data.columns!="price"]
y=data.price

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=.1,random_state=30)#30

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(xtrain,ytrain)

yPred=reg.predict(xtest)
#print(yPred)


from sklearn.metrics import mean_squared_error
print("Mean Squared Error:",mean_squared_error(ytest,yPred))
print("Regression Score:",reg.score(xtest,ytest)*100,"%\n")
hmap(data)

def PdictLoop():
    area=int(input('''Enter the associaste number of the Type of Aear you want
        1.Built-Up Area
        2.Carpet Area
        3.Plot Area
        4.Super Bulit-Up Area
'''))
    ownp=[]
    for i in range(1,5):
        if area>4:
            print("Please Select in the Above Option Only\nTRY AGAIN!!!")
            return
        if i==area:
            ownp.append(1)
        else:
            ownp.append(0)
    
    ownp.append(int(input("Enter The Estimate Sq.feet:")))
    ownp.append(int(input("Enter The BHK Requirment:")))
    ownp.append(int(input("Enter The Bathroom Requirment:")))
    ownp.append(int(input("Enter The Balcony Requirment:")))
    
    return ownp

ownp=PdictLoop()
requ=[ownp]
#print(requ)
psq=reg.predict(requ)
print("As per you Requirment Predicted Price is ",psq[0],"/sqft")


