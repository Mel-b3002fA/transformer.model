""" 
toy_data = [
    [1, 2, 3, 4],  
    [4, 3, 2, 1],  
]
 """

""" import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Seed for reproducibility
np.random.seed(42)

# Generate synthetic classification data
X, y = make_classification(n_samples=500, n_features=7, n_informative=5, 
                           n_redundant=1, n_classes=5, n_clusters_per_class=1)

# Convert to DataFrame
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target'] = y

# Add some categorical data
df['cat_feature_1'] = np.random.choice(['A', 'B', 'C'], size=500)
df['cat_feature_2'] = np.random.choice(['X', 'Y'], size=500)

# Add a non-linear feature interaction
df['interaction'] = df['feature_0'] * df['feature_1'] + np.sin(df['feature_2'])

# Add some noise
df['random_noise'] = np.random.normal(0, 1, size=500)

# Preview the result
print(df.head()) """




import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder

def toy_data():
    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=7, n_informative=5, 
                               n_redundant=1, n_classes=5, n_clusters_per_class=1)

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y

    # Convert categorical features into numbers
    df['cat_feature_1'] = np.random.choice(['A', 'B', 'C'], size=500)
    df['cat_feature_2'] = np.random.choice(['X', 'Y'], size=500)

    # Use LabelEncoder to convert categorical features into integers
    label_encoder_1 = LabelEncoder()
    df['cat_feature_1'] = label_encoder_1.fit_transform(df['cat_feature_1'])
    
    label_encoder_2 = LabelEncoder()
    df['cat_feature_2'] = label_encoder_2.fit_transform(df['cat_feature_2'])

    df['interaction'] = df['feature_0'] * df['feature_1'] + np.sin(df['feature_2'])
    df['random_noise'] = np.random.normal(0, 1, size=500)

    return df
