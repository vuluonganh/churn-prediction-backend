import numpy as np
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from xgboost.spark import SparkXGBClassifier
from pyspark.sql import SparkSession
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, roc_curve

spark = SparkSession.builder \
    .appName("xgbja") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
    .getOrCreate()

class XGBoostClassifierWrapper:
    def __init__(self,
                 label_col="label",
                 features_col="features",
                 num_workers=2,
                 missing=np.nan,
                 learning_rate=0.1,
                 max_depth=5,
                 n_estimators=200,
                 reg_alpha=0.2,
                 verbose=True,
                 enable_sparse_data_optim=False,
                 use_gpu=False):
        self.label_col = label_col
        self.features_col = features_col
        self.num_workers = num_workers
        self.missing = missing
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.reg_alpha = reg_alpha
        self.verbose = verbose
        self.enable_sparse_data_optim = enable_sparse_data_optim
        self.use_gpu = use_gpu
        self.model = self._create_model()

    def _create_model(self):
        return SparkXGBClassifier(
            label_col=self.label_col,
            features_col=self.features_col,
            num_workers=self.num_workers,
            missing=self.missing,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            reg_alpha=self.reg_alpha,
            verbose=self.verbose,
            enable_sparse_data_optim=self.enable_sparse_data_optim,
            use_gpu=self.use_gpu,
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
    xgb_wrapper = XGBoostClassifierWrapper()
    xgb_model = xgb_wrapper.get_model()
    model = xgb_model.fit(df_train)

    # Feature importance
    importances = model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': numerical_columns + categorical_columns,
        'importance': list(importances.values())
    }).sort_values(by='importance', ascending=False)

    print(importance_df)
    plt.figure(figsize=(8, 10))
    sns.barplot(
        data=importance_df,
        x='importance',
        y='feature',
        orient='h',
        color='royalblue'
    )
    plt.title('Feature Importance (XGBoost)', fontsize=15)
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Feature Name', fontsize=12)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Save model
    model.write().overwrite().save("model/xgb_spark_model")

    # Test
    test_predictions = model.transform(df_test)
    model_evaluation(model, test_predictions, "XGBoost Model")
if __name__ == "__main__":
    main()