import joblib
import pandas as pd

FILENAME = "data/Completed_model.joblib"
TESTNAME = "data/test"

loaded_model = joblib.load(FILENAME)
test = pd.read_csv(TESTNAME)
test_feat = test.drop('condition', axis=1)
test_label = test['condition']

result = loaded_model.score(test_feat, test_label)
print(result)
