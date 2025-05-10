import findspark
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, mean, sum, when
from imblearn.over_sampling import SMOTENC
from pyspark.sql import SparkSession

findspark.init()
spark = SparkSession.builder.appName("MyApp").master("local[*]").config("spark.driver.memory", "4g").config("spark.executor.memory", "4g").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
spark.conf.set("spark.sql.repl.eagerEval.enabled", True)

data = spark.read.csv('/Users/vothao/churn-prediction-frontend/model/data/Customer-Churn-Prediction_final.csv', inferSchema=True, header=True)

data = data.withColumn("TotalCharges", col("TotalCharges").cast("double"))

numerical_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_columns = ['Dependents','InternetService', 'OnlineSecurity',
                       'TechSupport', 'StreamingTV', 'StreamingMovies', 
                       'Contract', 'PaperlessBilling', 'PaymentMethod']

data.select('tenure', 'MonthlyCharges', 'TotalCharges').describe()

# Missing Vaue check
null_counts = data.select(
    [sum(when(col(c).isNull() | (col(c) == ""), 1).otherwise(0)).alias(c) for c in data.columns]
)

null_counts.show()


data = data.fillna({"TotalCharges": data.select(mean(col("TotalCharges"))).collect()[0][0]})

## Check duplicated
total_rows = data.count()
distinct_rows = data.distinct().count()
duplicates = total_rows - distinct_rows

if duplicates > 0:
    print(f"Number of duplicate rows: {duplicates}")
else:
    print("No duplicate rows found.")

## Check outliers
pandas_df = data.toPandas()

###Boxplot for numerical columns
n_cols = 3
n_rows = -(-len(numerical_columns) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()
for i, column in enumerate(numerical_columns):
    sns.boxplot(y=pandas_df[column], ax=axes[i])
    axes[i].set_title(f"Boxplot for {column}")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel(column)
for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

## Churn rate
churn_rate = data.groupBy("Churn").count().withColumn("Percentage", (col("count") / data.count()) * 100)
churn_rate.show()

## Churn Distribution by Demographic Factors
for column in categorical_columns:
    print(f"Churn Distribution by {column}:\n")

    churn_distribution = (
        data.groupBy(column, "Churn")
        .count()
        .withColumnRenamed("count", "Total")
    )
    churn_distribution.show()

## Visualization for Churn Rate
churn_rate_pd = churn_rate.toPandas()
sns.barplot(x="Churn", y="Percentage", data=churn_rate_pd, hue="Churn", palette={'No': 'skyblue', 'Yes': 'salmon'})
plt.xlabel('Churn')
plt.ylabel('Percentage')
plt.title("Churn Rate")
plt.show()

## Visualization for Churn Distribution by Demographic Factors
# Histogram for numerical columns with Churn
n_cols = 3
n_rows = -(-len(categorical_columns) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()
for i, column in enumerate(categorical_columns):
    sns.countplot(x=column, hue='Churn', data=pandas_df, palette={'No': 'skyblue', 'Yes': 'salmon'}, ax=axes[i])
    axes[i].set_title(f"Churn Distribution by {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Count")
    axes[i].tick_params(axis='x', rotation=45)

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

## Histogram for categorical columns without Churn
n_cols = 3
n_rows = -(-len(categorical_columns) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, column in enumerate(categorical_columns):
    if i < len(axes):
        sns.histplot(data=pandas_df, x=column, color='skyblue', discrete=True, ax=axes[i])
        axes[i].set_title(f"Distribution of {column}")
        axes[i].set_xlabel(f"{column}")
        axes[i].set_ylabel("Density")

for j in range(len(categorical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

## Histogram for categorical columns without Churn
n_rows = -(-len(numerical_columns) // n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.histplot(pandas_df[column], kde=True, ax=axes[i], color='skyblue', bins=30)
    axes[i].set_title(f"Distribution of {column}")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Count")

for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

## Histogram for numerical columns with Churn
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.histplot(data=pandas_df, x=column, hue="Churn", kde=True, ax=axes[i], bins=30, palette={'No': 'skyblue', 'Yes': 'salmon'}
    )
    axes[i].set_title(f"Distribution of {column} by Churn")
    axes[i].set_xlabel(column)
    axes[i].set_ylabel("Count")

for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Feature Engineering
##Oversampling
train, test = data.randomSplit([0.8, 0.2], seed=42)
churn_df_test = test.filter(col("Churn") == "Yes")
non_churn_df_test = test.filter(col("Churn") == "No")
churn_count_test = churn_df_test.count()
non_churn_test_balanced = non_churn_df_test.sample(
    withReplacement=False, fraction=1.0, seed=42
).limit(churn_count_test)
test_balanced = churn_df_test.union(non_churn_test_balanced)
test_ids = test_balanced.select("customerID")
train_balanced = data.join(test_ids, on="customerID", how="left_anti")

train_pd = train_balanced.toPandas()
###SMOTENC
X = train_pd.drop(columns=['Churn','customerID'])
y = train_pd['Churn']

categorical_indices = [X.columns.get_loc(col) for col in categorical_columns if col in X.columns]

smote_nc = SMOTENC(categorical_features=categorical_indices, random_state=42)
X_resampled, y_resampled = smote_nc.fit_resample(X, y)

resampled_data = pd.DataFrame(X_resampled, columns=X.columns)
resampled_data['Churn'] = y_resampled

#Save to CSV
resampled_data.to_csv('/Users/vothao/churn-prediction-frontend/model/data/balanced_training_data.csv', index=False)
test_balanced.toPandas().to_csv('/Users/vothao/churn-prediction-frontend/model/data/balanced_test_data.csv', index=False)