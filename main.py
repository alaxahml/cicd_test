from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import argparse




def load_data():
   print("Loading dataset")
   iris_dataset = load_iris()
   X, y = iris_dataset.data, iris_dataset.target
   print(f"Loaded dataset with {len(X)} samples.")
   return train_test_split(X, y, test_size=1 / 3)




def train(data, criterion, max_depth):
print("Training a DecisionTreeClassifier")


   print("Unpacking training and evaluation data…")
   X_train, X_test, y_train, y_test = data


   print("Instantiating model…")
   model_parameters = {
       "criterion": criterion,
       "splitter": "best",
       "max_depth": max_depth,
   }
   print(model_parameters)
   model = DecisionTreeClassifier(**model_parameters)


   print("Fitting model...")
   model.fit(X_train, y_train)


   print("Evaluating model...")
   y_pred = model.predict(X_test)
   evaluation = {
       "f1_score": f1_score(y_test, y_pred, average="macro"),
       "accuracy": accuracy_score(y_test, y_pred),
   }
   print(evaluation)




if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--criterion", type=str)
   parser.add_argument("--max-depth", type=int)
   args = parser.parse_args()


   data = load_data()
   train(
       data,
       criterion=args.criterion,
       max_depth=args.max_depth,
   )