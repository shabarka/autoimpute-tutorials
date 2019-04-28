
# Using AutoImpute to Explore Missing Data
---
This notebook introduces users to utlilty methods in the `Autoimpute` package. The tutorial includes:
1. Getting Started with the Utils Module
2. Exploring Data with Missing Values

## 1. Getting Started with the Uitls Module
---
First, let's examine what utility functions the `Autoimpute` package offers:


```python
import autoimpute.utils as au

def module_explore(m):
    methods = [f for f in dir(m) if not f.startswith("_")]
    statement = f"Available from {m.__name__}"
    print(f"{statement}\n{'-'*len(statement)}\n{methods}\n")

module_explore(au)
```

    Available from autoimpute.utils
    -------------------------------
    ['check_data_structure', 'check_missingness', 'check_nan_columns', 'check_predictors_fit', 'check_strategy_allowed', 'check_strategy_fit', 'checks', 'feature_corr', 'feature_cov', 'flux', 'helpers', 'inbound', 'influx', 'md_locations', 'md_pairs', 'md_pattern', 'outbound', 'outflux', 'patterns', 'proportions']
    


### Package Overview
---
The `utils` module contains checks to ensure datasets play nicely with imputation methods and functions to explore patterns in missing data. Note that `check_data_structure`, `check_missingness`, and `check_nan_columns` are decorators used to create new utility methods and build custom `Imputers`. This is a more advanced topic covered which we will cover in a future tutorial. For now, we'll explore the following methods to get started:
* `feature_cov` and `feature_corr`
* `proportions`, `md_locations`, `md_pattern`, and `md_pairs`
* `inbound`, `outbound`, `influx`, `outflux`, and `flux`

This tutorial explains how the methods above work in `Autoimpute`. They follow Van Buuren's (VB) Flexible Imputation of Missing Data, 2nd Edition, Section 4.1 closely. For a deeper understanding of the formulas behind each method, refer to his excellent text.

## 2. Exploring Data with Missing Values
---
Let's create an example dataframe that mimics the misingness structure from VB Section 4.1:


```python
import numpy as np
import pandas as pd

missing_data = pd.DataFrame({
    "A": [1, 5, 9, 6, 12, 11, np.nan, np.nan],
    "B": [2, 4, 3, 6, 11, np.nan, np.nan, np.nan],
    "C": [-1, -3, np.nan, np.nan, np.nan, -1, -2, 2]
})

missing_data
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>2.0</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.0</td>
      <td>4.0</td>
      <td>-3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6.0</td>
      <td>6.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.0</td>
      <td>11.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>11.0</td>
      <td>NaN</td>
      <td>-1.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>-2.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
</div>



### Covariance and Correlation
The `utils` module contains simple methods (`feature_cov` and `feature_corr`) to example the covariance and correlation matrix. Each method takes a dataframe as an argument. Missing values are **dropped by default**. Therefore, these methods return the covariance / correlation of observed features.


```python
# Covariance matrix after missing records dropped
au.feature_cov(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>17.066667</td>
      <td>11.35</td>
      <td>0.666667</td>
    </tr>
    <tr>
      <th>B</th>
      <td>11.350000</td>
      <td>12.70</td>
      <td>-2.000000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.666667</td>
      <td>-2.00</td>
      <td>3.500000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Correlation matrix after missing records dropped
au.feature_corr(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>1.000000</td>
      <td>0.765722</td>
      <td>0.114708</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.765722</td>
      <td>1.000000</td>
      <td>-1.000000</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.114708</td>
      <td>-1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>



### Locations and Patterns of Missingness
The `utils` module also contains methods to examine the locations and patterns of missingness. These methods help assess where data is missing, how often it is missing, and its co-occurence with missingness in other features.

The first of these methods is **`proportions`**. It returns the percent missing ("poms") and percent observed ("pobs") for each feature in a dataset. Note that the sum of these two columns should always equal 1. Each row is now a feature from the original dataset.


```python
au.proportions(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pobs</th>
      <th>poms</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.750</td>
      <td>0.250</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.625</td>
      <td>0.375</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.625</td>
      <td>0.375</td>
    </tr>
  </tbody>
</table>
</div>



Next is **`md_locations`**, which informs where data is missing within each feature. Here, 1 = missing; 0 = observed


```python
au.md_locations(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Next, **`md_pattern`** shows the row-wise patterns of missingness in our dataset. Let's start with the first row in the output below. There are 2 instances (count = 2) where every feature is observed (1). As a result, this row has no missing data (nmis = 0). Now examine the last row in the output below. There are 2 instances (count = 2) where column $A$ and $B$ having missing values while column $C$ is observed. As a result, this row has 2 of 3 features missing (nmis = 2).


```python
au.md_pattern(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
      <th>nmis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



**`md_pairs`** counts the number of missingness pair types between each set of features in a dataset. The pair types are:
1. `rr`: response-response pairs
2. `rm`: response-missing pairs
3. `mr`: missing-response pairs
4. `mm`: missing-missing pairs

The method returns a square matrix for each pair. In the output below, the name of each pair is capitalized to remain consistent with matrix notation in Latex used in this tutorial. `rr` and `mm` are symmetric, as the number of observed-observed or missing-missing patterns is the same regardless of which feature is first. In the output below, $RR_{A,B}$ indicates that there are 5 instances where $A$ and $B$ are both observed. Note that $RR_{A,B} = RR_{B,A} = 5$ Another example below, $MR_{A,C}$ indicates that there are 2 instances where $A$ is missing and $C$ is observed. Note that $MR_{A,C} = RM_{C,A} = 2$


```python
pairs = au.md_pairs(missing_data)
for pair_name, pair_data in pairs.items():
    print(f"{pair_name.upper()}\n{'-'*10}")
    print(f"{pair_data}")
```

    RR
    ----------
       A  B  C
    A  6  5  3
    B  5  5  2
    C  3  2  5
    RM
    ----------
       A  B  C
    A  0  1  3
    B  0  0  3
    C  2  3  0
    MR
    ----------
       A  B  C
    A  0  0  2
    B  1  0  3
    C  3  3  0
    MM
    ----------
       A  B  C
    A  2  2  0
    B  2  3  0
    C  0  0  3


### Missingness Statistics
The `utils` module includes statistics to assess the examine the effect of missing data on potential feature importance. These methods help assess which features may or may not be good candidates to be imputed or to assist in the imputation of other features.

**`inbound`** represents the proportion of useable cases in each column that can be used to impute the feature in each row. For this reason, the diagonal of the matrix is zero, as a feature cannot be useful to impute itself. A high value in an element indicates that the column is useful to impute the row. A low value in an element indicates that the column is not useful to impute the row. In the outbut below, we see that $I_{A,B} = 0$, because there are $0$ instances where $B$ is observed while $A$ is missing. This finding suggests that **$B$ is not helpful to impute $A$**. Extending this finding, we see the $C$ is always observed when $A$ is missing ($I_{A,C}$), so $C$ is useful to impute $A$ For features we are interested in imputing, we want them to have at least one (and preferably all) high values across their inbound row.


```python
au.inbound(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.000000</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.333333</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>C</th>
      <td>1.000000</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



**`outbound`** represents how well each column is connected to the rest of the data in each row. For this reason, the diagonal of the matrix is zero, as a feature cannot be well connected to itself. A high value in an element indicates that a row's observed features correspond with most of a column's missing features. A low value in an element indicates that a row's observed features correspond with few of a column's missing features. For example, $O_{B,C} = 0.6$. $A$ has 5 observed values. Of those 5, 3 from $C$ are missing, so outbound = 0.6. This finding suggests that observed in $B$ is well connected to missing in $C$, and $B$ may be helpful to impute $C$. We prefer features have high outbound values in the columns that they are used to impute.


```python
au.outbound(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>A</th>
      <th>B</th>
      <th>C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.0</td>
      <td>0.166667</td>
      <td>0.5</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.6</td>
    </tr>
    <tr>
      <th>C</th>
      <td>0.4</td>
      <td>0.600000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



**`flux`** collects five statistics in one method. They are:
1. `ainb`: average inbound
2. `aout`: average outbound
3. `influx`: influx coefficient
4. `outflux`: outflux coefficient
5. `pobs`: percentage observed

Of interest here are the **`influx`** and **`outflux`** statistics.

**`influx`** $\rightarrow I_{jk} = \frac{mr}{mr+rr}$. The number of `mr` pairs divided by the sum of `mr` and `rr. 0 = completely observed, 1 = completely missing. For two values with the same proportion of missing values, the one **with higher influx is "easier" to impute.**

**`outflux`** $\rightarrow O_{jk} = \frac{rm}{rm+mm}$. The number of `rm` pairs divided by the sum of `rm` and `mm`. 0 = completely missing, 1 = completely observed. For two values with the same proportion of missing values, the one with **higher outflux is better connected and thus a better imputer.**


```python
au.flux(missing_data)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ainb</th>
      <th>aout</th>
      <th>influx</th>
      <th>outflux</th>
      <th>pobs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>A</th>
      <td>0.500000</td>
      <td>0.333333</td>
      <td>0.125</td>
      <td>0.500</td>
      <td>0.750</td>
    </tr>
    <tr>
      <th>B</th>
      <td>0.666667</td>
      <td>0.300000</td>
      <td>0.250</td>
      <td>0.375</td>
      <td>0.625</td>
    </tr>
    <tr>
      <th>C</th>
      <td>1.000000</td>
      <td>0.500000</td>
      <td>0.375</td>
      <td>0.625</td>
      <td>0.625</td>
    </tr>
  </tbody>
</table>
</div>


