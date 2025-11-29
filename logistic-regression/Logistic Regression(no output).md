```python
import pandas as pd
import numpy as np
!pip install tqdm
!pip install matplotlib
```


```python
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt

```


```python
!pip install scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

```

# <span style="color:red">A POKEMON DATASET </span>
## USING EXTERNAL APIS TO READ OBJECTS 


```python
url= "https://pokeapi.co/api/v2/pokemon-species?limit=2000"
```


```python
res = requests.get(url)
data = res.json()
```


```python
df = pd.DataFrame(data['results'])
df

```

## Looping the details to get pokemons 


```python
details = []
for u in tqdm(df["url"]):
    poke_data = requests.get(u).json()
 
    
    details.append({
        "name": poke_data["name"],
        "id":poke_data["id"],
        "base_happiness": poke_data["base_happiness"], 
        "capture_rate": poke_data["capture_rate"],      
        "gender_rate": poke_data["gender_rate"],  
        "generation": poke_data["generation"]["name"],   
        "growth_rate": poke_data["growth_rate"]["name"], 
        "habitat":poke_data["habitat"]["name"] if poke_data.get("habitat") else None, 
        "is_legendary": poke_data["is_legendary"], 
        "is_mythical":poke_data["is_mythical"],
    })

```


```python
df2 = pd.DataFrame(details)
df2
```

## Merging the dfs



```python
df = pd.merge(df, df2)          # merge
df.set_index("id", inplace=True)
df
```

# <span style="color:blue">*DATA PRECPROCESSING AND CLEANING*</span>


```python
df.describe()
```

### Lets check if the pokemon with 0 happiness information is correct


```python
df.loc[df["base_happiness"]== 0]
```


```python
df[df.isnull().any(axis=1)]
```

Since after gen-3 pokemon dont have their habital information given in apis. We are going to use their generation to assign their region to their habitat


```python
generation_to_region = {
    "generation-i": "Kanto",
    "generation-ii": "Johto",
    "generation-iii": "Hoenn",
    "generation-iv": "Sinnoh",
    "generation-v": "Unova",
    "generation-vi": "Kalos",
    "generation-vii": "Alola",
    "generation-viii": "Galar",
    "generation-ix": "Paldea"
}
df.loc[df["habitat"].isna(), "habitat"] = df.loc[df["habitat"].isna(), "generation"].map(generation_to_region)

df["hngbd"] = (df["generation"].map(generation_to_region) == df["habitat"]) #habitat not given by default
df
```

Since the data is correct, we move onto visualization

# ***<span style="color:blue">DATA VIZUALIZATION</span>***
## <span style="color:green">***Pkmn catch_rate and base_happiness distribution***</span>


```python
bins = np.arange(0,df['base_happiness'].max()+10,10)
df['base_happiness'].plot(kind='hist', bins=bins,figsize=(20,10), title='Distribution of Base Happiness')
```


```python
bins = np.arange(0,df['capture_rate'].max()+10,10)
df['capture_rate'].plot(kind='hist',bins=bins, title='Distribution of Capture Rate')
```

### What does this tell us?

Majority of pokemon are between base happiness of 70-80 and capture_rate of 40-50

## ***<span style="color:green">CATCH RATE AND BASE HAPPINESS ACCORDING TO GENS</span>***


```python
df.groupby('generation')['capture_rate'].mean().plot(kind='bar', title='Distribution of capture rate per generation', figsize=(10,5))
df.groupby('generation')['capture_rate'].mean()
```


```python
df.groupby('generation')['base_happiness'].mean().plot(kind='barh', figsize=(10,5), title='Distribution of base happiness across gens')
df.groupby('generation')['base_happiness'].mean()
```

## ***Findings***
### **Capture Rate**
Higher catch rate = higher difficulty 
so therefore, Hoenn, Gen 3, gas the highest avergae difficulty of catching pokemon while Kalos, Gen 6, and Paldea, Gen 9, having the least average difficulty of catching pokemon

### **Base Happiness**
Kanto, Gen 1, and Sinnoh, Gen 4, are the 2 highest generation on base happiness while the recently released regions of Galar, Gen 8, and Paldea, Gen 9, being the least

## ***<span style="color:green">Legendary and Mythicals per generation and their catch rate and base happiness<span style="color:green">***



```python
df.groupby('generation')[['is_legendary','is_mythical']].sum()
```


```python
df.groupby('generation')[['is_legendary','is_mythical']].sum().plot(kind='bar', figsize=(10,6), title='Legendary and Mythical per Generation')

```

### ***Findings***
GEN 7,8,9 have the highest number of legendaries while GEN 4,7 have the highest number of mythical.

***Mythical are legendaries pkmn in game that are only obtainable through in-game events or through special means***

## <span style="color:green">***Capture Rate between legendary and mythical***</span>


```python

df_legendary_mythical = df[(df['is_legendary'] == True) | (df['is_mythical'] == True)]

df_legendary_mythical.groupby('generation')[['capture_rate', 'is_legendary', 'is_mythical']].apply(
    lambda x: pd.Series({
        'Legendary': x.loc[x['is_legendary'], 'capture_rate'].mean(),
        'Mythical': x.loc[x['is_mythical'], 'capture_rate'].mean()
    })
).plot(kind='bar', figsize=(10,6), title='Average Capture Rate of Legendary & Mythical per Generation')

```

### ***Findings***
The Average Capture Rate Difficulty for GEN 1,2,4 is significantly higher than their legendaries.
Other generations have much easier catch rate for their mythical pokemon compared to their legendary 
except Gen 5, Unova, where the catch rate for legendary and mythical are similar

## <span style="color:green">***Base Happiness between legendary and mythical***</span>


```python

df_legendary_mythical = df[(df['is_legendary'] == True) | (df['is_mythical'] == True)]

df_legendary_mythical.groupby('generation')[['base_happiness', 'is_legendary', 'is_mythical']].apply(
    lambda x: pd.Series({
        'Legendary': x.loc[x['is_legendary'], 'base_happiness'].mean(),
        'Mythical': x.loc[x['is_mythical'], 'base_happiness'].mean()
    })
).plot(kind='bar', figsize=(10,6), title='Average Base Happiness of Legendary & Mythical per Generation')

```

### ***Findings***
Something Interesting we can see there is that only Sinnoh, Gen 4 has legendary with average base happiness higher than that of mythicals.
Whereas,From Gen 7,8,9 the base happiness for mythical pokemons were removed. Whereas there is no base happiness for the legendaries of Kalos, Gen 6.

Gen 9 not only has the lowest base happiness among all pokemon but also among legendaries and mythical being 0.

# <span style='color: blue'>***Number of PKMN per Gens***</span>


```python
df['generation'].value_counts()
```


```python
df['generation'].value_counts().plot(
    kind='pie',
    figsize=(10,10),
    autopct='%1.5f%%',
    title='Percentage of PKMN per Generation'
)
```

### Findings

Unova, generation v, has the most pokemons while Kalos, generation vi, has the least amount of pokemons from their region
Since mega evolution gimich was introduced in Gen 6, previous gens pokemon mega evolution were the primarily main focus rather than their own Gens PKMN.
Every Mega evolution was given to previous gen pokemon in Gen 6 with major characters pokemon also being from previous gens.
Even within the Kalos pokemon league, in the anime, we can see the recurring starters from Gen 1 and Gen 3 
Kalos stood out more because of the buffs to the previous generation in the form of mega evolution rather than the region itself.


## <span style='color:blue'>***TRAINING THE MODEL***</span>


```python
X= df.drop(['is_legendary','is_mythical'], axis=1)
X= pd.get_dummies(X, drop_first=True)
```


```python
y = df["is_legendary"]
```


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) 
model.fit(X_train,y_train)

preds= model.predict(X_test)
accuracy = accuracy_score(y_test,preds)
print('Accuracy', accuracy)
```

The accuracy was 100%. Lets further evaluate it


```python
df
```

The type basically gave it away, lets now drop the type column as well


```python
X= df.drop(['is_legendary','is_mythical','type'], axis=1)
X= pd.get_dummies(X, drop_first=True)

y = df["is_legendary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) 
model.fit(X_train,y_train)

preds= model.predict(X_test)
accuracy = accuracy_score(y_test,preds)
print('Accuracy', accuracy)
```

Very accurate here as well, Legendary Pokemon do have a very high capture rate compared to normal ones. Some of the accuracy might be due to easy capture rate. For example, in pokemon black and white, Gen 5 Unova, games reshiram and zekrom the 2 legendary had every low capture rate. The game wouldnt move forward until you captured them and continue on a loop.

Let us remove the capture_rate as well


```python
X= df.drop(['is_legendary','is_mythical','type','base_happiness','capture_rate','gender_rate'], axis=1)
X= pd.get_dummies(X, drop_first=True)

y = df["is_legendary"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000) 
model.fit(X_train,y_train)

preds= model.predict(X_test)
accuracy = accuracy_score(y_test,preds)
print('Accuracy', accuracy)
```

# <p style='color: orange'>Let us see if it predicts correct by giving the pkmns name </p>
Lets us compare the accuracy rise and drop according to certain parameters


```python
X = df.drop(['type','is_legendary','is_mythical'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns


y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))
```


```python
def predict_by_name(pokemon_names, model, df, encoder_columns):
    for pokemon_name in pokemon_names:
        row = df[df['name'] == pokemon_name.lower()]

        if row.empty:
            return "Pokémon not found!"
            
        row = row.drop(columns=['type'], errors='ignore')
        
        row_encoded = pd.get_dummies(row)
        
        row_encoded = row_encoded.reindex(columns=encoder_columns, fill_value=0)
        
        pred = model.predict(row_encoded)
        
        print ({pokemon_name: 'Legendary' if pred == 0 else 'Mythical' if pred == 1 else 'Normal'})
```


```python
pokemon_names = [
    'Mew',
    'Manaphy',
    'Arceus', 
    'Celebi',
    'Darkrai',
    'Kyurem',
    'Mewtwo',
    'Lucario', 
    'Dragonite', 
    'Garchomp',
    'terapagos'
]

predict_by_name(pokemon_names,model,df,X.columns)
```

## Findings

It got Mew and Celebi wrong. Moving on to further droping of the paramters


### <p style='color:red'>Here we are going to drop capture_rate, base_happiness</p>


```python
X = df.drop(['type','is_legendary','is_mythical','capture_rate','base_happiness'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns


y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

predict_by_name(pokemon_names,model,df,X.columns)
```

### Findings
It got mew right this time thanks to the +0.2 rise on accuract but still got celebi wrong.
Dropping those 2 columns actually gave us better accuracy
### <p style='color:red'>Dropping Gender Rate</p>


```python
X = df.drop(['type','is_legendary','is_mythical','capture_rate','base_happiness','gender_rate'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns


y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))


predict_by_name(pokemon_names,model,df,X.columns)
```


```python
X = df.drop(['type','is_legendary','is_mythical','capture_rate','base_happiness','growth_rate'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns


y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))


predict_by_name(pokemon_names,model,df,X.columns)
```

***The dropping of gender rate and growth rate gave it a drastic decrease in accuracy. Also produced 0 percision error. Except for kyurem and mewtwo it got everything wrong for the legendary and mythicals***

### <p style='color:red'>Let us experiment by droping the legendary and mythical column one at a time </p>


```python
X = df.drop(['type','is_mythical','capture_rate','base_happiness'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns

y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
d2=model.fit(X_train, y_train)

print(d2)

```

***100% ACCURACY*** <br>
By not dropping just the is_legendary column it had an acuuracy of 100%. 
<br>
<hr>

## ***<p style='color: red'>Keeping the is_mythical column </p>***


```python
X = df.drop(['type','is_legendary','capture_rate','base_happiness'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns


y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

predict_by_name(pokemon_names,model,df,X.columns)
```

***By keeping the is_mythical column instead of is_legendary we can see 4% accuracy drop***

***REASONING*** <br>
Because there are way more legendary than the mythicals. And also due to their high capture rate, and similarity in gender rate keeping the is_legendary column yields way better, 100% accuracy results contrary to keeping the is_mythical column

Now to further test the gender rate between the legendary, mythical and non-legendary


```python
for label, group in df.groupby("type"):
    plt.scatter(group['type'], group["gender_rate"], label=label)

plt.xlabel("Pokémon Index")
plt.ylabel("Gender Rate")
plt.title("Gender Rate Distribution by Pokémon Type (Legendary / Normal / Mythical)")
plt.legend()

plt.show()
```

We can pretty much see that mythical pokemon all seem to have a gender_rate of -1.
Therefore, the 100% accuracy. If the is_legendary column is false, its either a normal type or
mythical. If its higher than -1 , it will automatically be a normal type. but what else is determining the 100% except the gender_rate? could growth rate be a part of it?


```python
df['growth_rate'].value_counts()
```


```python
for label, group in df.groupby("type"):
    plt.scatter(group['type'], group["growth_rate"], label=label)

plt.xlabel("Pokémon Index")
plt.ylabel("Growth Rate")
plt.title("Growth Rate Distribution by Pokémon Type (Legendary / Normal / Mythical)")
plt.legend()

plt.show()
```

Since there is no clear distinction between mythical and normal pokemon from either the gender_rate and growth_rate
We can conclude that the 100% acuuracy is due to the high number of train subject and less number of test subjects. Now let us have less train and more test


```python
X = df.drop(['type','is_mythical','capture_rate','base_happiness'], axis=1) #when dropping only type, it gaves us 100% accuracy
X= pd.get_dummies(X, drop_first=True) # this makes it into multiple columns

y = df['type']


encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y) # we only need one label for taget so we use encoder instead

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.6, random_state=42)


model = LogisticRegression(solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
# print('x_test \n',X_test)
# print('Prediction \n',y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

predict_by_name(pokemon_names,model,df,X.columns)
```

With 99% accuracy, it got celebi wrong. let us see whats different about celebi


```python
df.loc[df['name'] == 'celebi']
```


```python
df.loc[df['type']== 'Mythical']['growth_rate'].value_counts()
```


```python
to_load=df.loc[(df['type']== 'Mythical') & (df['growth_rate'] == 'medium-slow'),'name'].values
```


```python
predict_by_name(to_load,model,df,X.columns)
```


```python
to_load=df.loc[(df['type']== 'Mythical') & (df['growth_rate'] == 'slow'),'name'].values
```


```python
predict_by_name(to_load,model,df,X.columns)
```

We can see it getting some typing wrong. Now lets moving on to normal types. Let us see with the 99% accuracy can it predict every normal type correctly
But since number of normal types is way too many we will opt for selecting those with grwoth rate and gender rate similar to mythicals


```python
to_load=df.loc[(df['type']== 'Normal') & ((df['growth_rate'] == 'slow')|(df['growth_rate'] == 'medium-slow')) & (df['gender_rate']== -1),'name'].values
predict_by_name(to_load,model,df,X.columns)
```


```python
to_load=df.loc[(df['type']== 'Normal') & (df['gender_rate']== -1),'name'].values
predict_by_name(to_load,model,df,X.columns)
```

It got every single normal type correct

### <p style='color:red'> ***Conclusion*** </p>

We can conclude that 
- base_happiness and capture_rate mislead the prediction, when dropping them the accuracy of the model increased
- is_legendary as a parameter gave better accuracy than that of is_mythical
- growth_rate and gender_rate also plays a crucial rate of figuring out its rarity
  


```python
import os
print(os.environ['PATH'])
```


```python

```
