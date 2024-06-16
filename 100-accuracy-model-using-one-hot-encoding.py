#!/usr/bin/env python
# coding: utf-8

# # There're some twists in this notebook, I tried to make it as beginner-friendly as possible. 
# # If you find yourself feeling lost at any point, don't worry. Take a break, revisit the previous lines, and continue when you're ready.

# In[1]:


import regex as re
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# ## First we load and view the data

# In[2]:


main_data = pd.read_csv("/kaggle/input/disease-symptom-description-dataset/dataset.csv")


# In[3]:


main_data.head(10)


# In[4]:


main_data.sample(5)


# In[5]:


main_data.shape


# ## Let's check if the data is balanced or not

# In[6]:


main_data.Disease.value_counts()


# ## There's two approaches to handle this type of data
# ###  Label_Encoding
# ###  One-Hot-Encoding-Style
# ## and we are going to discuss both

# # Label_Encoding

# #### We are going to label encode the Disease column first, then the rest

# In[7]:


df = main_data.copy() # We take a copy of the original data incase we needed the original data later
df.dropna(axis=1, how='all', inplace=True) # Dropping rows which are all NaN
df.fillna(0, inplace=True)                 # Replacing the NaN with 0

# Creating a custom label encoder so we can specify which number the encoding starts from
class CustomLabelEncoder(LabelEncoder):
    def __init__(self, start=0):
        self.start = start
        super().__init__()

    def fit_transform(self, y):
        encoded = super().fit_transform(y)
        encoded += self.start
        return encoded

# Flatten the 'Disease' column into a single Series
flattened_series = df['Disease'].astype(str)

# Create and fit label encoder for the 'Disease' column
encoder = CustomLabelEncoder(start=200) # Here we tell the label encoder to start encoding from 200


# *Why?* you might ask
# Because if we just imported and fitted the usual label encoder, it will start indexing from 0.
# *So?*
# In the next step, we will label encoding the **rest** of the data, and that encoder will start from 0 to 131.
# So we are trying to prevent different values from getting encoding the same way.
# 
# *BUT WHY ARE WE DOING THEM SEPARATLY?!* you might ask.
# When I first wrote the code I thought this way would be easier than just encoding
# the entire dataset, then separate the features from the targets in the label_mapping dictionary.
# 
# If you find this was complicated or impractical, that's okay, just label_encode the entire data then seperate the features from the labels. The end result will be the same: converting string into int

# In[8]:


encoded_values = encoder.fit_transform(flattened_series)
df['Disease'] = encoded_values

mapping_data = {'label_encoder': encoder}

# Saving the mapping of the label column "Disease" to use later
label_mapping = {k: v for k, v in zip(mapping_data['label_encoder'].classes_, range(200, 200+len(mapping_data['label_encoder'].classes_)))}

df.head()


# In[9]:


label_mapping


# #### Now we are going to use the label encoder to encode the rest of the data

# In[10]:


# Stack the entire data into a single Series.
# We are stacking the entire data because there're similar values in different columns. **REMEMBER THIS**
encode_df = df.copy() # Again, taking a copy because we might need the original later.
encode_df = encode_df.drop(["Disease"], axis = 1)
flattened_series = encode_df.stack().astype(str)

# Create and fit label encoder.
encoder = LabelEncoder()
encoded_values = encoder.fit_transform(flattened_series)

# Reshape the encoded values back to the original DataFrame shape.
F_encoded_df = pd.DataFrame(encoded_values.reshape(encode_df.shape), columns=encode_df.columns,
                            index=encode_df.index)

# Store the mapping data for future use
Fmapping_data = {'label_encoder': encoder}
feature_mapping = {k: v for k, v in zip(Fmapping_data['label_encoder'].classes_, 
                                        Fmapping_data['label_encoder'].\
                                        transform(Fmapping_data['label_encoder'].classes_))}
F_encoded_df.head(3)


# In[11]:


feature_mapping


# In[12]:


label_encoded_df = pd.concat([df['Disease'], F_encoded_df], axis = 1)
label_encoded_df.head()


# #### So now we have a dataset called **label_encoded_df** that has the same data as **main_data** dataset but label-encoded.
# #### And we saved the mapping of the target column in a dict called *label_mapping*, and the mapping of the features in a dict called *feature_mapping*.

# ### Let's create and compile the model

# In[13]:


# Creating X and y
model_features = label_encoded_df.columns.tolist()
model_features.remove("Disease")
X = label_encoded_df[model_features]
y = label_encoded_df["Disease"]


# In[14]:


# One_hot_encoding the y column to use it as a multicalss in the model output layer
y_encoded = pd.get_dummies(y)
y_encoded.shape


# In[15]:


# The column names are the mapping of the target column. **REMEMBER THIS**
y_encoded.head()


# #### We can't use the StandardScaler in the same manner like we usually do because this dataset has reccurenting, similar values in different columns, and StandardScaler apply the scaling column-wise.
# #### So as we did earlier with the label_encoder when encoding the features, we are going to scale the entire X all at once.

# In[16]:


# Reshape the data
X_reshaped = X.values.reshape(-1, 1)
scaler = StandardScaler().fit(X_reshaped)
X_scaled_reshaped = scaler.transform(X_reshaped)
# Reshape back to original shape
X_scaled = X_scaled_reshaped.reshape(X.shape)
X_df = pd.DataFrame(X_scaled)
X_df.head()


# #### As you can see the NaN values that were encoded to 0 are now ALL scaled to 0.696026. If we applied the StandardScaler as we normally do, these 0 values in different columns would have different values after scaling.

# In[17]:


X_train, X_test, y_train, y_test = train_test_split(X_df, y_encoded, test_size = 0.25, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[18]:


X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
X_eval_tensor = tf.convert_to_tensor(X_eval.values, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float64)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float64)
y_eval_tensor = tf.convert_to_tensor(y_eval, dtype=tf.float64)


# In[19]:


X_train_tensor


# In[20]:


y_train_tensor


# In[21]:


with tf.device('/GPU:0'):
    model_1 = keras.Sequential([
        layers.Input(shape=(X_train_tensor.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(128, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dense(128, activation='tanh'),
        layers.Dropout(0.1),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(y_train_tensor.shape[1], activation='softmax')])
    
    model_1.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max')
    history = model_1.fit(X_train_tensor, y_train_tensor, epochs=500, callbacks=[early_stopping],
                batch_size=16, validation_data=(X_eval_tensor, y_eval_tensor))


# In[22]:


model_1.evaluate(X_test_tensor, y_test_tensor)


# ### Looks great so far...99% accuracy
# ## LET'S TEST IT MANUALLY

# In[23]:


def encode_user_input(user_input, mapping=feature_mapping):
    '''
    This function takes user input and transform it to the same encoding 
    the original data, which the model was trained on, has.

    Args:
        user_input (str): The user input.
        mapping (dict): The mapping the label_encoder used earlier.

    Returns:
        str: encoded user input.
    '''
    encoded_input = []
    for symptom in user_input:
        for key in mapping.keys():
            if symptom.strip().lower() == key.strip().lower():
                encoded_input.append(mapping[key])
                break  # Break out of inner loop if a match is found
    return encoded_input


# In[24]:


# let's take a random row from the original data.
user_input = ['itching','skin_rash','nodal_skin_eruptions','dischromic _patches']
# This row should result in "Fungal infection".
encoded_input = encode_user_input(user_input)
encoded_input


# In[25]:


# Transforming the encoded user input to a tensor.
input_tensor = tf.cast(encoded_input, tf.float32)
input_tensor


# In[26]:


# Checking the number of dimensions.
input_tensor.ndim == X_train_tensor[1].ndim


# ### Let's check if the data is encoded in the same way the original data was

# In[27]:


label_encoded_df.iloc[0][1:5]


# ### Great, here is the entire row

# In[28]:


label_encoded_df.head(1)


# ### itching, skin_rash,..., then all NaN, or 130 after the label_encoding. So we need to 'pad' the user input to match the original data

# In[29]:


padding_value = tf.constant(130, dtype=tf.float32)
desired_length = X_train_tensor[1].shape[0]
padding_length = desired_length - tf.shape(input_tensor)[0]
padding_tensor = tf.fill((padding_length,), padding_value)
final_input = tf.concat([input_tensor, padding_tensor], axis=0)
final_input


# In[30]:


target_index = y_encoded.columns.tolist() # If you remember, the column names after the one-hot-encoding ARE the mapping of the target values.


# ### Scaling the user input:

# In[31]:


final_array = final_input.numpy()
final_reshaped = final_array.reshape(-1, 1)
X_scaled = scaler.transform(final_reshaped)
final_tensor = tf.convert_to_tensor(X_scaled)
final_tensor = tf.squeeze(final_tensor)
final_tensor


# In[32]:


X_df.head(1)


# ### And finally using the trained model to predict the user input

# In[33]:


import numpy as np
predict_proba = model_1.predict(tf.expand_dims(final_input, axis = 0)) # Expanding dims to get (1,17)
predicted_class_index = np.argmax(predict_proba) # Getting the 'index' of our prediction
prediction_encode = target_index[predicted_class_index] # Getting to mapping of that 'index' using y column names
inverse_label_encoding = {v: k for k, v in label_mapping.items()} # Inverse the label encoding
prediction = inverse_label_encoding[prediction_encode]
prediction


# ### This should've been 'Fungal infection'.
# ### Although getting 99% accuracy, looks like our model behaves poorly...

# ## Let's try another approach

# # One-Hot-Encoding-style

# ## As you remember, this is how our original data looks like

# In[34]:


main_data.head()


# ## Let's try to "one-hot-encode" this

# In[35]:


df = main_data.copy() # As usual, taking a copy from that data incase we needed the original later
# Combine all symptom columns into a single column
df['All Symptoms'] = df.apply(lambda row: ','.join(row.dropna()), axis=1)
# Drop duplicate symptoms within each cell
df['All Symptoms'] = df['All Symptoms'].apply(lambda x: ','.join(sorted(set(x.split(','))) if x else ''))
stay_cols= ['Disease', 'All Symptoms']
df = df[stay_cols]
df.head()


# In[36]:


df['All Symptoms'][0]


# ### Great, let's also remove the '_'
# #### if you notice, there's a "Fungal infection" in that row. We will fix that later

# In[37]:


def strip_to_basic_tokens(text):
    # Remove doble spaces and underscores
    text = re.sub(r'[_\s]+', ' ', text)
    # Split by commas and lowercase the tokens
    tokens = [token.strip().lower() for token in text.split(',')]
    return tokens

# Apply the function to 'All Symptoms' column
df['Basic Tokens'] = df['All Symptoms'].apply(strip_to_basic_tokens)
df['Basic Tokens'] = df['Basic Tokens'].apply(lambda x: ', '.join(x))
df = df.drop(['All Symptoms'], axis = 1)
df.head()


# In[38]:


df['Basic Tokens'][0]


# ### Looking good, now let's "one-hot-encode" it using Multi-Label Binarizer

# In[39]:


dfE = df.copy() # Taking a copy because we never know what might happen
dfE['Basic Tokens'] = dfE['Basic Tokens'].apply(lambda x: x.split(', '))

mlb = MultiLabelBinarizer()
# Fit and transform the 'Basic Tokens' column
one_hot_encoded = pd.DataFrame(mlb.fit_transform(dfE['Basic Tokens']), columns=mlb.classes_, index=df.index)

# Concatenate the one-hot encoded DataFrame with the original DataFrame
df_encoded = pd.concat([dfE, one_hot_encoded], axis=1)

# Drop the 'Basic Tokens' column
df_encoded = df_encoded.drop(columns=['Basic Tokens'])
df_encoded.head()


# In[40]:


df_encoded.shape


# ### Now let's drop the diseases column values that got encoded in the column names:

# In[41]:


disease_names = [key for key in label_mapping.keys()]
diseases = [strip_to_basic_tokens(disease) for disease in disease_names]
diseases_cleaned = [item[0] if isinstance(item, list) else item for item in diseases]
df_encoded = df_encoded.drop(diseases_cleaned, axis = 1)
df_encoded.shape


# ## Now we will create and compile a model the same way we did earlier

# In[42]:


model_features = df_encoded.columns.tolist()
model_features.remove("Disease")
X = df_encoded[model_features]
y = df_encoded["Disease"]


# In[43]:


y_encoded = pd.get_dummies(y)
y_encoded.shape


# In[44]:


y_encoded.head()


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.25, random_state=42)
X_eval, X_test, y_eval, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


# In[46]:


X_train_tensor = tf.convert_to_tensor(X_train.values, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test.values, dtype=tf.float32)
X_eval_tensor = tf.convert_to_tensor(X_eval.values, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float64)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float64)
y_eval_tensor = tf.convert_to_tensor(y_eval, dtype=tf.float64)


# In[47]:


X_train_tensor


# In[48]:


with tf.device('/GPU:0'):
    model_2 = keras.Sequential([
        layers.Input(shape=(X_train_tensor.shape[1],)),
        layers.Dense(160, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(200, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(240, activation='tanh'),
        layers.BatchNormalization(),
        layers.Dense(240, activation='tanh'),
        layers.Dropout(0.2),
        layers.Dense(200, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(160, activation='relu'),
        layers.Dense(y_train_tensor.shape[1], activation='softmax')])
    
    model_2.compile(optimizer= 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=4, mode='max')
    history = model_2.fit(X_train_tensor, y_train_tensor, epochs=500, callbacks=[early_stopping],
                batch_size=16, validation_data=(X_eval_tensor, y_eval_tensor))


# In[49]:


model_2.evaluate(X_test_tensor, y_test_tensor)


# ## Great! 100% accuracy and a 7.9e-04 loss! Now let's test i manually:

# In[50]:


# If you remember in the first model, we took a row from the origial data to test the model
# We aren't going to do this here, let's REALLY test it
user_input = ['stomach_pain','acidity','chest_pain'] # This should be GERD

original_data = df_encoded.copy()

# We will change the strip_to_basic_tokens function just a little bit to be able to deal with the user input
def strip_to_basic_tokens(symptoms):
    symptoms = [symptom.strip().lower().replace(' ', '_').replace('_', ' ') for symptom in symptoms]
    return [re.sub(r'\s+', ' ', symptom) for symptom in symptoms]
# Apply strip_to_basic_tokens function to user input
user_input_stripped = strip_to_basic_tokens(user_input)

# Initialize MultiLabelBinarizer with all symptoms
mlb = MultiLabelBinarizer(classes=df_encoded.columns)

# Fit and transform user input
user_input_encoded = pd.DataFrame(mlb.fit_transform([user_input_stripped]), columns=mlb.classes_)

# Concatenate user input with original data
final_user_input = pd.concat([pd.DataFrame(columns=original_data.columns), user_input_encoded], axis=0)
final_user_input = final_user_input.drop(['Disease'],axis = 1)
# Print the final user input shape
final_user_input.head()


# ## Great! Now the user input looks exactly like the df_encoded data.

# In[51]:


user_tensor = tf.convert_to_tensor(final_user_input.values, dtype=tf.float32)
user_tensor[0]


# ### After converting the user input to a tensor, we'll utilize the model to predict the disease the user may have:

# In[52]:


predict_proba = model_2.predict(user_tensor)
predicted_class_index = np.argmax(predict_proba)
prediction_encode = target_index[predicted_class_index]
inverse_label_encoding = {v: k for k, v in label_mapping.items()}
prediction = inverse_label_encoding[prediction_encode]
prediction


# ## WOOHOOO! The model is performing as expected. A 100% accuracy model

# ### Let's test it again

# In[53]:


user_input = ['continuous_sneezing','watering_from_eyes'] # This should be Allergy

original_data = df_encoded.copy()

# Apply strip_to_basic_tokens function to user input
user_input_stripped = strip_to_basic_tokens(user_input)

# Fit and transform user input
user_input_encoded = pd.DataFrame(mlb.fit_transform([user_input_stripped]), columns=mlb.classes_)

# Concatenate user input with original data
final_user_input = pd.concat([pd.DataFrame(columns=original_data.columns), user_input_encoded], axis=0)
final_user_input = final_user_input.drop(['Disease'],axis = 1)
# Print the final user input shape
final_user_input.head()


# In[54]:


user_tensor = tf.convert_to_tensor(final_user_input.values, dtype=tf.float32)
user_tensor[0]


# In[55]:


predict_proba = model_2.predict(user_tensor)
predicted_class_index = np.argmax(predict_proba)
prediction_encode = target_index[predicted_class_index]
inverse_label_encoding = {v: k for k, v in label_mapping.items()}
prediction = inverse_label_encoding[prediction_encode]
prediction


# ## So unlike model_1 that was trained on the label_encoded data, model_2 is actually behaving as it should.

# *What to do now?* Well, you can test the following:
# * Check if the symptom_severity has any significance when applied to the data.
# * Try different model architecture.
# * Try different approaches to prepare the data.
# * Just **have fun**.

# ## There's a different approach that will be appropriate to use here, NLP.
# ## But maybe in a future notebook...
