import os,sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if True:
   
    crop = pd.read_csv('crop_production_dataset.csv')
    crop['Cropid'] = crop.groupby(['Crop']).ngroup()
    crop['Cropid'] = crop['Cropid']+1
    crop['Seasonid'] = crop.groupby(['Season']).ngroup()
    crop['Seasonid'] = crop['Seasonid']+1
    crop = crop.dropna(how='any',axis=0)

    data_cid_label=crop.iloc[:,[5,9]]
    group_cid_label= data_cid_label.groupby(['Cropid','Crop'])
    B1=group_cid_label.first()
    index = dict(B1.index)
    print('Crop Name and ID -',index)

    cid=int(input("Enter Crop ID From Above List:"))
    #Years=int(input("Enter Year:"))
    Years=2022
    print("Price For Year:",Years)    
    Season=int(input("Enter Season(1-Autumn,2-Kharif,3-Rabi,4-Summer,5-Whole Year,6-Winter):"))
    Totalarea = float(input("Enter Area:"))
    Stateval="Maharashtra"
    print("For State:",Stateval)
    
    crop=crop.loc[((crop['Cropid'] == cid))]    
    crop.to_csv("temp.csv")
    crop = pd.read_csv('temp.csv')


    
    X = np.array(crop[['Seasonid', 'Cropid', 'Area']])
    y = np.array(crop['Production'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X1 = np.array(crop[['Seasonid', 'Cropid', 'Production']])
    y1 = np.array(crop['Price_Kg'])
    X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size = 0.2, random_state = 0)


    #Train Production Data regressor Model+++++++++++++++++++++++++++
    Pregressor = RandomForestRegressor(n_estimators=10, random_state=5)
    Pregressor.fit(X_train, y_train)
    Pytest_pred = Pregressor.predict(X_test)

    #Train Rate Data regressor Model+++++++++++++++++++++++++++
    Rregressor = RandomForestRegressor(n_estimators=10, random_state=5)
    Rregressor.fit(X_train1, y_train1)
    Rytest_pred = Rregressor.predict(X_test1)

    x=cid
    crop1=crop.loc[((crop['Cropid'] == x))]
    if len(crop1)>=1:
        Rnewpredicttest = np.array([Season,x,Totalarea]).reshape(1, 3)            
        PYnewtest_pred = Pregressor.predict(Rnewpredicttest)
        Productionresult=int(PYnewtest_pred[0])

        Rnewpredicttest1 = np.array([Season,x,Productionresult]).reshape(1, 3)
        RYnewtest_pred = Rregressor.predict(Rnewpredicttest1)
        Rateresult=int(RYnewtest_pred[0])

        print("Rate result Using RandomForest-",Rateresult)
        Raccuracy=round(accuracy_score(y_test1, Rytest_pred.round())*100, 2)
        print("Rate accuracy Using RandomForest- ",Raccuracy,"%")
        print("RandomForest -")
        print("Prodction:-"+str(Productionresult))
        print("Rate: Rs."+str(Rateresult)+"/-")

        newtrain = np.hsplit(X_train, 3)
        X_trainnew=newtrain[2]
        plt.scatter(X_trainnew, y_train, color = 'blue')

        newtest = np.hsplit(X_test, 3)
        X_testnew=newtest[2]
        plt.scatter(X_testnew,Pytest_pred,color = 'green')

        newptest = np.hsplit(Rnewpredicttest, 3)
        X_testnewpredict=newptest[2]
        plt.scatter(X_testnewpredict,PYnewtest_pred,color = 'red')
        plt.title('Scatter Plot Crop Production Using RandomForest', fontsize=14)
        plt.xlabel('Totalarea', fontsize=14)
        plt.ylabel('Production', fontsize=14)
        pngval='CropProductionRandomForest_scatterploat.png'
        plt.savefig(pngval)
        plt.clf()
                
    else:
        print("Crop Data Not Available.!")
                
