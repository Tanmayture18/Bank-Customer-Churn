from flask import Flask,render_template,request
import pickle

app = Flask(__name__)
file=open('model.pk1','rb')
pipe=pickle.load(file)
file.close()

@app.route("/",methods=["GET","POST"])
def hello_world():
    if request.method=="POST":
        mydict=request.form
        CreditScore=int(mydict['CreditScore'])
        Gender=int(mydict['Gender'])
        Age=int(mydict['Age'])
        Tenure=int(mydict['Tenure'])
        Balance=int(mydict['Balance'])
        NumOfProducts=int(mydict['NumOfProducts'])
        HasCrCard=int(mydict['HasCrCard'])
        IsActiveMember=int(mydict['IsActiveMember'])
        EstimatedSalary=int(mydict['EstimatedSalary'])

        inputfeatures=[CreditScore,Gender,Age,Tenure,Balance,NumOfProducts,HasCrCard,IsActiveMember,EstimatedSalary]
        
        

        pred=pipe.predict([inputfeatures])

        return render_template('show.html',inf=pred)
    return render_template('index.html')

if __name__=='__main__':
    app.run(debug=True)   
