import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC, SVC
from time import perf_counter
import warnings
warnings.filterwarnings(action='ignore')

# Read the CSV file into a DataFrame
df = pd.read_csv('message.csv', encoding="ISO-8859-1")

# Drop the 'Unnamed: 0' column
df.drop('Unnamed: 0', axis=1, inplace=True)
# Rename the columns
df.columns = ['Label', 'Text', 'Label_Number']

# Print the first few rows of the DataFrame
df.head()

# Perform data exploration and analysis
df.shape
df.info()
df.isnull().sum()
df['Label_Number'].value_counts()

# Visualize the distribution of labels using a countplot
plt.style.use('dark_background')
plt.figure(figsize=(10, 7))
sns.countplot(data=df, x='Label')

# Download the 'punkt' tokenizer from NLTK
nltk.download('punkt')

# Define a function to count the number of words in a text
def count_words(text):
    words = word_tokenize(text)
    return len(words)

# Apply the count_words function to the 'Text' column and create a new 'count' column
df['count'] = df['Text'].apply(count_words)

# Group the data by 'Label_Number' and calculate the average word count
df.groupby('Label_Number')['count'].mean()

# Define a function to clean the text by converting it to lowercase and removing non-alphabetic characters
def clean_str(string, reg=RegexpTokenizer(r'[a-z]+')):
    string = string.lower()
    tokens = reg.tokenize(string)
    return " ".join(tokens)

# Clean the 'Text' column using the clean_str function
df['Text'] = df['Text'].apply(lambda string: clean_str(string))

# Remove the word 'subject' from the 'Text' column
df['Text'] = [' '.join([item for item in x.split() if item not in 'subject']) for x in df['Text']]

# Perform stemming on the 'Text' column using the PorterStemmer from NLTK
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stemming(text):
    return ''.join([stemmer.stem(word) for word in text])

df['Text'] = df['Text'].apply(stemming)

# Perform feature extraction using CountVectorizer
x = df.loc[:, 'Text']
y = df.loc[:, 'Label_Number']

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
x = cv.fit_transform(df.Text)
y = df.Label

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define a dictionary of models to train
models = {
    'Random Forest': {'model': RandomForestClassifier(), 'perf': 0},
    'Gradient Boosting': {'model': GradientBoostingClassifier(), 'perf': 0},
    'MultinomialNB': {'model': MultinomialNB(), 'perf': 0},
    'Logisitc Regression': {'model': LogisticRegression(), 'perf': 0},
    'KNN': {'model': KNeighborsClassifier(), 'perf': 0},
    'Decision Tree': {'model': DecisionTreeClassifier(), 'perf': 0},
    'SVM (Linear)': {'model': LinearSVC(), 'perf': 0},
    'SVM (RBF)': {'model': SVC(), 'perf': 0}
}

# Train the models and measure the training time
for name, model in models.items():
    start = perf_counter()
    model['model'].fit(x_train, y_train)
    duration = perf_counter() - start
    duration = round(duration, 2)
    model['perf'] = duration

    print(f'{name:20} trained in {duration} sec')

# Calculate the accuracy of the models on the test set
models_accuracy = []
for name, model in models.items():
    accuracy = model['model'].score(x_test, y_test)
    models_accuracy.append([name, accuracy, model['perf']])

models_accuracy1 = []
for name, model in models.items():
    accuracy = model['model'].score(x_train, y_train)
    models_accuracy1.append([name, accuracy, model['perf']])

# Create a DataFrame to display the model accuracies and training times
df_accuracy = pd.DataFrame(models_accuracy)
df_accuracy.columns = ['Model', 'Test Accuracy', 'Training time (sec)']
df_accuracy.sort_values(by='Test Accuracy', ascending=False, inplace=True)
df_accuracy.reset_index(drop=True, inplace=True)

# Display the model accuracies on the test set
df_accuracy

# Visualize the model accuracies on the test set using a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x='Model', y='Test Accuracy', data=df_accuracy)
plt.title('Accuracy on the test set\n', fontsize=15)
plt.ylim(0.825, 1)
plt.show()

# Visualize the training times for each model using a bar plot
plt.figure(figsize=(15, 6))
sns.barplot(x='Model', y='Training time (sec)', data=df_accuracy)
plt.title('Training time for each model in sec', fontsize=15)
plt.ylim(0, 20)
plt.show()

# Perform hyperparameter tuning for MultinomialNB using GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV

parameters = {'alpha': [0.2, 1, 2, 5, 10], 'fit_prior': [True, False]}
grid = GridSearchCV(MultinomialNB(), param_grid=parameters)
grid.fit(x_train, y_train)
df_results = pd.DataFrame(grid.cv_results_)
df_results = df_results[['params', 'mean_test_score']]
df_results.sort_values(by='mean_test_score', ascending=False, inplace=True)

print(df_results)
grid.best_params_

# Train the MultinomialNB model with the best hyperparameters
alpha, fit_prior = grid.best_params_['alpha'], grid.best_params_['fit_prior']
model = MultinomialNB(alpha=alpha)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# Print the classification report and accuracy score of the MultinomialNB model
print(classification_report(y_test, y_pred))
print('Accuracy:', round(accuracy_score(y_test, y_pred), 3) * 100, '%')

# Define a function to display the predicted labels for a given DataFrame
def display_result(df, number=2):
    for i in range(number):
        msg = df['Text'].iloc[i]
        label = df['Label'].iloc[i]

        msg_vec = cv.transform([msg])
        pred_label = model.predict(msg_vec)
        print('Real:', label, 'Predicted:', pred_label[0])
        print('Email:', msg)

# Display the predicted labels for some examples from the ham and spam emails
df_spam = df[df['Label'] == 'spam']
df_ham = df[df['Label'] == 'ham']
display_result(df_ham, 6)



# In[ ]:




