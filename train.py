import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.azureml
import seaborn as sns
import argparse
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, MaxAbsScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve

import collections
import shutil 

# brute force delete local model directory if it exists
shutil.rmtree('model', ignore_errors=True)



def split_dataset(X, Y):

    le = LabelEncoder()
    Y = le.fit_transform(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, 
                                                        Y, 
                                                        test_size = 0.2,
                                                        random_state=123)


    return X_train, X_test, Y_train, Y_test

# remove the column we'll predict

def prepareDataset(df):
    Y = df['ArrDelay15'].values
    synth_df = df.drop(columns=['ArrDelay15'])
    print(collections.Counter(Y))
    return synth_df, Y

# mlflow autolog metrics
def analyze_model(clf, X_test, Y_test, preds):
        accuracy = accuracy_score(Y_test, preds)
        print(f'Accuracy', float(accuracy))
        mlflow.log_metric(f'Accuracy', float(accuracy))

        precision = precision_score(Y_test, preds, average="macro")
        print(f'Precision', float(precision))
        mlflow.log_metric(f'Precision', float(precision))
        
        recall = recall_score(Y_test, preds, average="macro")
        print(f'Recall', float(recall))
        mlflow.log_metric(f'Recall', float(recall))
        
        f1score = f1_score(Y_test, preds, average="macro")
        print(f'F1 Score', float(f1score))
        mlflow.log_metric(f'F1 Score', float(f1score))
        
        mlflow.lightgbm.log_model(clf, artifact_path="outputs", registered_model_name="fd_model_mlflow_proj")
        mlflow.lightgbm.save_model(clf, path="model")

        class_names = clf.classes_
        fig, ax = plt.subplots()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names)
        plt.yticks(tick_marks, class_names)
        sns.heatmap(pd.DataFrame(confusion_matrix(Y_test, preds)), annot=True, cmap='YlGnBu', fmt='g')
        ax.xaxis.set_label_position('top')
        plt.tight_layout()
        plt.title('Confusion Matrix', y=1.1)
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.show()
        fig.savefig("ConfusionMatrix.png")
        mlflow.log_artifact("ConfusionMatrix.png")
        plt.close()

        preds_proba = clf.predict_proba(X_test)[::,1]
        fpr, tpr, _ = roc_curve(Y_test, preds_proba, pos_label = clf.classes_[1])
        auc = roc_auc_score(Y_test, preds_proba)
        plt.plot(fpr, tpr, label="data 1, auc=" + str(auc))
        plt.legend(loc=4)
        plt.show()
        plt.close()

# read in the data, use local file or param for cloud run
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, help="input data path", default=".")

    args = parser.parse_args()
    print(args.data)

    data = pd.read_csv(args.data+'/flightdelayweather_ds_clean.csv')

    X, y = prepareDataset(data)

    #Split the input dataset
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    print(X_train.dtypes)
    print(y_train)

    # Run LightGBM classifier
    
    clf = LGBMClassifier(learning_rate=0.24945760279230222, max_bin=511,
               min_child_samples=29, n_estimators=80, num_leaves=21,
               reg_alpha=0.0020334241010261135, reg_lambda=0.04344763354508823, metric='auc', is_unbalance='false')


# Analyze the model

model = clf.fit(X_train, y_train)
preds = model.predict(X_test)
analyze_model(clf, X_test, y_test, preds)
