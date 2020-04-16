import pandas as pd
import os
import joblib
import numpy as np

TEST_DATA = os.environ.get("TEST_DATA")

MODEL = os.environ.get("MODEL")

def predict():
    df = pd.read_csv(TEST_DATA)
    test_index = df.id.values
    predictions = None

    for FOLD in range(5):
        label_encoders = joblib.load("models/{}_{}_label_encoder.pkl".format(MODEL, FOLD))
        cols = joblib.load("models/{}_{}_columns.pkl".format(MODEL, FOLD))
        df = pd.read_csv(TEST_DATA)
        for c in cols:
            lbl = label_encoders[c]
            df.loc[:, c] = lbl.transform(df[c].values.tolist())

        # Ready to predict
        clf = joblib.load("models/{}_{}.pkl".format(MODEL, FOLD))

        df = df[cols]
        preds = clf.predict_proba(df)[:, 1]

        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds

    predictions  /= 5

    submit  = pd.DataFrame(np.column_stack((test_index, predictions)),columns=["id","ACTION"])
    submit["id"] = submit.id.astype('int')
    return submit

if __name__ == "__main__":
    submission = predict()
    submission.to_csv("models/{}.csv".format(MODEL), index=False)


