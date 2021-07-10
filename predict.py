from predictcancellation import cleananomalies

import pickle

#loading model
with open("./models/RFCmodel.pkl", "rb") as f:
    model = pickle.load(f)

def process_data():
    import pandas as pd
    
    df = pd.read_csv("./temp/input.csv", sep=',')
    
    temp_df = cleananomalies(filepath="./temp/input.csv")
    
    temp_df.drop(['is_canceled', 'Canceled', 'Check-Out'], axis=1, inplace=True)
    
    columns_list = list(temp_df.columns)
    
    predictions = []
    
    for i in range(len(temp_df)):
        feature_array=[]
        for col in columns_list:
            try:
                feature_array.append(temp_df[col][i])
            except TypeError as e:
                print(e)
                feature_array.append(0)
        prediction = model.predict([feature_array])
        print("Prediction of Row {} is {}".format(i, prediction[0]))
        predictions.append(prediction[0])
    # predictions = model.predict([temp_df])
    
    print(predictions)
        
    df["Prediction"] = predictions
    
    #stream = io.StringIO()
    df.to_csv("./temp/export.csv", index=False)

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)