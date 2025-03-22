from kfp import dsl 

# load the data 
@dsl.component(base_image="python:3.12", packages_to_install=[ "scikit-learn", "numpy", "joblib" ])
def load_data(train_data: dsl.OutputPath('train_data'), test_data: dsl.OutputPath('test_data')):
   # import the following package
   from sklearn.datasets import load_iris
   from sklearn.model_selection import train_test_split
   import joblib

   iris_data = load_iris()
   X_data = iris_data.get('data')
   y_data = iris_data.get('target')

   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=.1)

   # write training data out
   with open(train_data, "wb") as f:
      joblib.dump( (X_train, y_train), f)

   # write test data out   
   with open(test_data, "wb") as f:
      joblib.dump( (X_test, y_test), f)

@dsl.component(base_image='python:3.12', packages_to_install=['scikit-learn', 'numpy', 'joblib'])
def train_model(train_data: dsl.InputPath('train_data'), model: dsl.OutputPath('model')):
   from sklearn import svm 
   import joblib

   # Load training data
   with open(train_data, "rb") as f:
      X_train, y_train = joblib.load(f)

   # Create a model for training
   svm_model = svm.SVC()
   svm_model.fit(X_train, y_train)

   # Save svm_model
   with open(model, 'wb') as f:
      joblib.dump(svm_model, f)

# Predict component
@dsl.component(base_image='python:3.12', packages_to_install=['scikit-learn', 'numpy', 'joblib'])
def predict(test_data: dsl.InputPath('test_data'), model: dsl.InputPath('model')) -> float:

   from sklearn import metrics 
   import joblib

   # Load the test data
   with open(test_data, 'rb') as f:
      X_test, y_test = joblib.load(f)

   # Load the model
   with open(model, 'rb') as f:
      model = joblib.load(f)

   y_prediction = model.predict(X_test)

   return metrics.accuracy_score(y_prediction, y_test)

# Pipeline
@dsl.pipeline(name='Iris pipeline', description='A simple pipeline for training Iris dataset')
def iris_pipeline() -> float:
   # no parameters for ds.OutputPath
   prep_dataset = load_data() # -> output = train_data, test_data
   
   train_result = train_model(train_data = prep_dataset.outputs['train_data']) # -> output = model

   # call the predict
   result = predict(test_data = prep_dataset.outputs['test_data'], model = train_result.outputs['model'])

   return result.output
