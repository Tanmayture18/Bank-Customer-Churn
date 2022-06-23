import pandas as pd
import numpy as np
import pickle
from sklearn.pipeline import Pipeline





if __name__=='__main__':
    df=pd.read_csv('data.csv')
    df.drop(['RowNumber','CustomerId','Surname','Geography'],axis=1,inplace=True)
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    df['Gender']=le.fit_transform(df['Gender'])
    # Balancing dataset
    legit=df[df.Exited==0]
    fraud=df[df.Exited==1]
    legit_sample=legit.sample(n=2037)
    df1=pd.concat([legit_sample,fraud],axis=0)
    X=df1.drop('Exited',axis=1)
    y=df1.Exited
    from sklearn.model_selection import train_test_split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    from sklearn.preprocessing import StandardScaler
    # sc=StandardScaler()
    # X_train=sc.fit_transform(X_train)
    # X_test=sc.fit_transform(X_test)

    from sklearn.svm import SVC
    pipe=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf'))])
    
    # clf=SVC(kernel='rbf')
    # clf.fit(X_train,y_train)
    pipe.fit(X_train,y_train)

    file=open('model.pk1','wb')
    pickle.dump(pipe,file)   
    file.close() 


    