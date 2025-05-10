from pyspark.ml.classification import LinearSVC
import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

spark = SparkSession.builder \
    .appName("xgbja") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

class SVMClassifierWrapper:
    def __init__(self,
                 label_col="label",
                 features_col="features",
                 max_iter=200,
                 reg_param=0.01,
                 tol=0.0001,
                 fit_intercept=True,
                 aggregation_depth=2,
                 standardization=True,
                 threshold=0.0,
                 max_block_size_in_mb=0.0,
                 prediction_col="prediction",
                 raw_prediction_col="rawPrediction"):
        self.label_col = label_col
        self.features_col = features_col
        self.max_iter = max_iter
        self.reg_param = reg_param
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.aggregation_depth = aggregation_depth
        self.standardization = standardization
        self.threshold = threshold
        self.max_block_size_in_mb = max_block_size_in_mb
        self.prediction_col = prediction_col
        self.raw_prediction_col = raw_prediction_col
        self.model = self._create_model()

    def _create_model(self):
        return LinearSVC(
            labelCol=self.label_col,
            featuresCol=self.features_col,
            maxIter=self.max_iter,
            regParam=self.reg_param,
            tol=self.tol,
            fitIntercept=self.fit_intercept,
            aggregationDepth=self.aggregation_depth,
            standardization=self.standardization,
            threshold=self.threshold,
            maxBlockSizeInMB=self.max_block_size_in_mb,
            predictionCol=self.prediction_col,
            rawPredictionCol=self.raw_prediction_col
        )

    def get_model(self):
        return self.model
    
def metrics_calculator(y_true, y_pred_class, y_pred_prob, model_name, has_probability=True):
    '''
    This function calculates performance metrics for a given model.
    If probabilities are not available, AUC is omitted.
    '''
    metrics = [
        accuracy_score(y_true, y_pred_class),
        precision_score(y_true, y_pred_class, average='binary', zero_division=1),
        recall_score(y_true, y_pred_class, average='binary', zero_division=1),
        f1_score(y_true, y_pred_class, average='binary', zero_division=1)
    ]
    index = ['Accuracy', 'Precision', 'Recall', 'F1-score']

    if has_probability:
        metrics.append(roc_auc_score(y_true, y_pred_prob))
        index.append('AUC')

    result = pd.DataFrame(data=metrics,
                          index=index,
                          columns=[model_name])
    result = (result * 100).round(2).astype(str) + '%'
    return result

def model_evaluation(clf, test_predictions, model_name, has_probability=True):
    '''Displays model evaluation including classification report, confusion matrix, ROC curve (if applicable), and metrics.'''

    # Check if probability column exists
    columns = test_predictions.columns
    has_probability = has_probability and 'probability' in columns

    # Extract predictions and true labels
    y_true = test_predictions.select("label").rdd.map(lambda row: row['label']).collect()
    if has_probability:
        y_pred_prob = test_predictions.select("probability").rdd.map(lambda row: row[0][1]).collect()
        y_pred_class = [1 if prob >= 0.5 else 0 for prob in y_pred_prob]
    else:
        y_pred_class = test_predictions.select("prediction").rdd.map(lambda row: row['prediction']).collect()
        y_pred_prob = None  

    # Classification report
    print("\n\t  Classification report for test set")
    print("-" * 55)
    print(classification_report(y_true, y_pred_class))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # ROC Curve (only if probabilities are available)
    if has_probability:
        fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
        auc = roc_auc_score(y_true, y_pred_prob)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.show()

    # Performance metrics
    result = metrics_calculator(y_true, y_pred_class, y_pred_prob, model_name, has_probability)
    print(result)

def main():

    df_train = spark.read.csv('/Users/vothao/churn-prediction-frontend/model/data/balanced_training_data.csv', header=True, inferSchema=True)
    df_train = df_train.drop('customerID')

    df_test = spark.read.csv('/Users/vothao/churn-prediction-frontend/model/data/balanced_test_data.csv', header=True, inferSchema=True)
    df_test = df_test.drop('customerID')

    numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
    categorical_columns = ['Dependents', 'InternetService', 'OnlineSecurity',
                           'TechSupport', 'StreamingTV', 'StreamingMovies', 
                           'Contract', 'PaperlessBilling', 'PaymentMethod']

    # StringIndexer
    indexers = [StringIndexer(inputCol=col, outputCol=col + "_Index") for col in categorical_columns]
    label_indexer = StringIndexer(inputCol='Churn', outputCol='label').fit(df_train)
    indexers.append(label_indexer)

    # VectorAssembler + StandardScaler
    feature_columns = numerical_columns + [col + "_Index" for col in categorical_columns]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol='num_features')
    scaler = StandardScaler(inputCol='num_features', outputCol='features')

    # Pipeline
    pipeline = Pipeline(stages=indexers + [assembler, scaler])
    pipeline_model = pipeline.fit(df_train)
    df_transformed = pipeline_model.transform(df_train)
    df_test_transformed = pipeline_model.transform(df_test)

    df_train = df_transformed.select('features', 'label')
    df_test = df_test_transformed.select('features', 'label')

    # Model
    svm_classifier = SVMClassifierWrapper()
    svm_model = svm_classifier.get_model()
    svm_model = svm_model.fit(df_train)

    # Feature importance
    svm_coefficients = svm_model.coefficients.toArray()
    svm_importance_df = pd.DataFrame({
        'feature': numerical_columns + categorical_columns,
        'coefficient': svm_coefficients
    })
    svm_importance_df['abs_coefficient'] = svm_importance_df['coefficient'].abs()
    svm_importance_df = svm_importance_df.sort_values(by='abs_coefficient', ascending=False).reset_index(drop=True)

    # Plot feature importance
    plt.figure(figsize=(8, 10))
    sns.barplot(
        data=svm_importance_df,
        x='abs_coefficient',
        y='feature',
        orient='h',
        color='royalblue'
    )
    plt.title('Feature Importance (SVM)', fontsize=15)
    plt.xlabel('Absolute Coefficient', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Save model
    svm_model.write().overwrite().save("model/svm_spark_model")

    # Model evaluation
    test_predictions = svm_model.transform(df_test)
    model_evaluation(svm_model, test_predictions, "SVM", has_probability=False)

if __name__ == "__main__":
    main()