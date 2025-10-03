"""
Predictive modeling with Machine Learning for Smart/Ultra mobile plan recommendation

This project develops a machine learning model to recommend optimal mobile plans for Megaline customers.
The goal is to create a binary classification model (Ultra=1, Smart=0) that achieves minimum 75% accuracy
and can generalize well on unseen data for production deployment.

Methodology:
1. Data loading and exploration
3. Data Splitting
4. Model Selection and Training
5. Final Model Evaluation
6. Sanity Testing

"""

# Data analysis and visualization
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

# Data splitting and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

# Load DataFrame.
df = pd.read_csv('/datasets/users_behavior.csv')

# General information and data sample.
df.info()
df.head(10)

# EDA
# Data visualizations by plan type
def plans_histogram(df, column, plan_labels={0: 'Smart', 1: 'Ultra'}):
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df,
                 x=column,
                 hue='is_ultra',
                 kde=True,
                 alpha=0.6,
                 palette='pastel'
                )
    plt.title(f'{column} Distribution by Plan Type')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.legend(title='Plan', labels=[plan_labels[0], plan_labels[1]])
    plt.grid(axis='y', alpha=0.3)
    plt.show()

# Calls by plan
plans_histogram(df, 'calls')
# Minutes by plan
plans_histogram(df, 'minutes')
# Messages by plan
plans_histogram(df, 'messages')
# Data usage by plan
plans_histogram(df, 'mb_used')


# Data Splitting
# Initialize variables
features = df.drop('is_ultra', axis=1)
target = df['is_ultra']

# First split: 60% train - 40% temp (val + test)
features_train, features_temp, target_train, target_temp = train_test_split(features, 
                                                                            target,
                                                                            test_size=0.4, 
                                                                            random_state=42,
                                                                            stratify=target
                                                                           )

# Second split: 20% val - 20% test (from 40% temporary)
features_valid, features_test, target_valid, target_test = train_test_split(features_temp,
                                                                            target_temp,
                                                                            test_size=0.5,
                                                                            random_state=42,
                                                                            stratify=target_temp
                                                                           )

# Proportion verification
print('Training Set Proportions:\n', target_train.value_counts(normalize=True))
print('\nValidation Set Proportions:\n', target_valid.value_counts(normalize=True))
print('\nTest Set Proportions:\n', target_test.value_counts(normalize=True))


# Model Selection and Training
# Metrics to capture results
models_df = pd.DataFrame(columns=['model', 'train_accuracy', 'validation_accuracy', 'overfitting'])

# Decision Tree model
best_depth = 0
best_train_acc_1 = 0
best_valid_acc_1 = 0

for depth in range(1, 11):
    model_1 = DecisionTreeClassifier(random_state=42, max_depth=depth)
    model_1.fit(features_train, target_train)

    train_acc_1 = model_1.score(features_train, target_train)
    valid_acc_1 = model_1.score(features_valid, target_valid)
    
    if (valid_acc_1 > best_valid_acc_1) or (valid_acc_1 == best_valid_acc_1 and train_acc_1 > best_train_acc_1):
        best_depth = depth
        best_train_acc_1 = train_acc_1
        best_valid_acc_1 = valid_acc_1

print("Model: Decision Tree")
print("Accuracy with max_depth equal to", best_depth)
print("Training set:", best_train_acc_1)
print("Validation set:", best_valid_acc_1)

# Prepare data to be added to DataFrame
results_1 = {'model': 'DecisionTreeClassifier',
             'train_accuracy': round(best_train_acc_1, 4),
             'validation_accuracy': round(best_valid_acc_1, 4),
             'overfitting': (best_train_acc_1 - best_valid_acc_1)
            }

# Random Forest model
best_est = None
best_train_acc_2 = 0
best_valid_acc_2 = 0

for est in [50, 100, 150, 200]:
    model_2 = RandomForestClassifier(n_estimators=est, max_depth=best_depth, random_state=42)
    model_2.fit(features_train, target_train)
    
    train_acc_2 = model_2.score(features_train, target_train)
    valid_acc_2 = model_2.score(features_valid, target_valid)

    if valid_acc_2 > best_valid_acc_2:
        best_est = est
        best_train_acc_2 = train_acc_2
        best_valid_acc_2 = valid_acc_2
        
print("\nModel: Random Forest")
print("Accuracy with n_estimators equal to", best_est)
print("Training set:", best_train_acc_2)
print("Validation set:", best_valid_acc_2)

# Prepare data to be added to DataFrame
results_2 = {'model': 'RandomForestClassifier',
             'train_accuracy': round(best_train_acc_2, 4),
             'validation_accuracy': round(best_valid_acc_2, 4),
             'overfitting': (best_train_acc_2 - best_valid_acc_2)
            }

# Logistic Regression model
model_3 = LogisticRegression(random_state=42)
model_3.fit(features_train, target_train)

train_acc_3 = model_3.score(features_train, target_train)
valid_acc_3 = model_3.score(features_valid, target_valid)

print("\nModel: Logistic Regression")
print("Training set:", train_acc_3)
print("Validation set:", valid_acc_3)

# Prepare data to be added to DataFrame
results_3 = {'model': 'LogisticRegression',
             'train_accuracy': round(train_acc_3, 4),
             'validation_accuracy': round(valid_acc_3, 4),
             'overfitting': (train_acc_3 - valid_acc_3)
            }

# Add best results to metrics DataFrame
models_df.loc[len(models_df)] = results_1 
models_df.loc[len(models_df)] = results_2
models_df.loc[len(models_df)] = results_3
models_df.head()


# Final Model Evaluation
# Combine train and valid to get 80% of the data
features_train_val = pd.concat([features_train, features_valid])
target_train_val = pd.concat([target_train, target_valid])

# Train final model
final_model = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=42)
final_model.fit(features_train_val, target_train_val)
test_score = final_model.score(features_test, target_test)

print("Final model accuracy on the test set is:", test_score)


# Sanity Testing
# Create first model that predicts the most frequent class in the data
dummy_most_frq = DummyClassifier(strategy="most_frequent", random_state=42)
dummy_most_frq.fit(features_train, target_train)

dummy_acc_frq = dummy_most_frq.score(features_test, target_test)
print("Accuracy of the most frequent baseline model is:", dummy_acc_frq)

# Create second model that predicts randomly
dummy_random = DummyClassifier(strategy="uniform", random_state=42)
dummy_random.fit(features_train, target_train)

dummy_acc_random = dummy_random.score(features_test, target_test)
print("Accuracy of the random prediction baseline model is:", dummy_acc_random)

# Results comparison
print(f"""\nThe final model has an accuracy of {round(test_score, 4)}, 
showing a {round(test_score - dummy_acc_frq, 4)} difference from the most frequent baseline and 
{round(test_score - dummy_acc_random, 4)} from the random baseline, confirming that the final model performs significantly better.
        """)