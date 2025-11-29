# Pokémon Legendary & Mythical Prediction Model

## Overview

This project applies **Logistic Regression** to classify Pokémon as **Legendary, Mythical, or Normal** based on selected attributes. Various experiments were performed to assess how different features affect model accuracy.

---

## Model Development

###  Initial Approach

```python
X = df.drop(['is_legendary', 'is_mythical'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df["is_legendary"]
```

* Trained a Logistic Regression model.
* **Achieved 100% accuracy**, but the feature `type` made classification trivial.

###  Feature Removal Experiments

| Removed Features                 | Result                                        |
| -------------------------------- | --------------------------------------------- |
| `type`                           | Still high accuracy (type is very predictive) |
| `capture_rate`, `base_happiness` | Accuracy improved                          |
| `gender_rate`, `growth_rate`     | Accuracy dropped significantly             |
| Keeping `is_legendary` only      | Perfect accuracy                           |
| Keeping `is_mythical` only       | ~4% drop in accuracy                       |

---

## Key Insights

### Most Influential Features

* `is_legendary` → strong classifier
* `gender_rate = -1` → often indicates Mythical Pokémon
* `capture_rate` & `base_happiness` → misleading (bias due to game design)
* `growth_rate` → weak indicator

### Overfitting Concern

High accuracy occurred due to:

* Limited dataset variety
* Strong indicator features (type-based)

A reversed train-test split (40% train → 60% test) dropped accuracy to **99%**, and Celebi was misclassified.

---

## Name-Based Prediction Test

```python
pokemon_names = [
    'Mew', 'Manaphy', 'Arceus', 'Celebi', 'Darkrai',
    'Kyurem', 'Mewtwo', 'Lucario', 'Dragonite',
    'Garchomp', 'Terapagos'
]
```

✔ Some Mythical Pokémon like **Mew** were initially misclassified
✔ Normal Pokémon with **gender_rate = -1** were correctly identified
 **Celebi was incorrectly classified under certain feature configurations**

---

##  Visualization Highlights

* **Mythicals consistently showed `gender_rate = -1`**
* **No clear distinction in `growth_rate` between Normal and Mythical**
* Legendaries often shared similarities with Mythicals in growth pattern

---

##  Final Conclusion

| Feature                          | Effect on Accuracy        |
| -------------------------------- | ------------------------- |
| `capture_rate`, `base_happiness` |  Remove (misleading)    |
| `gender_rate`                    |  Important for rarity   |
| `is_legendary`                   |  Best performance       |
| `is_mythical`                    |  Slightly less reliable |
| `growth_rate`                    |  Low impact             |

 **Best Model Configuration:**

```python
X = df.drop(['type', 'is_mythical', 'capture_rate', 'base_happiness'], axis=1)
```

Predictions achieved **100% accuracy**, logically driven by rarity and biological traits.

---

## Tech Stack

* **Python**, **Pandas**, **Matplotlib**
* **Scikit-Learn**: `LogisticRegression`, `LabelEncoder`, `train_test_split`

---

##  Suggested Improvements

✔ Use **more advanced models** (Random Forest, XGBoost, Neural Networks)
✔ Add **cross-validation**
✔ Expand to include **competitive stats / total base power**
✔ Introduce **synthetic augmentation for Mythical Pokémon**

---

##  Example Code — Name-Based Predictor

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
        print({pokemon_name: 'Legendary' if pred == 0 else 'Mythical' if pred == 1 else 'Normal'})
```

---

##  Credits

This project was conducted as a data-driven exploration of how well machine learning can predict Pokémon rarity using statistical traits.

---

##  Contact

Feel free to reach out for collaboration, improvements, or discussion!
