# Gradual Pattern Labelling

A gradual pattern (GP) is a set of gradual items (GI) and its quality is measured by its computed support value. A GI is
a pair (i,v) where i is a column and v is a variation symbol: increasing/decreasing. Each column of a data set yields 2
GIs; for example, column age yields GI age+ or age-. For example given a data set with 3 columns (age, salary, cars) and
10 objects. A GP may take the form: {age+, salary-} with a support of 0.8. This implies that 8 out of 10 objects have
the values of column age 'increasing' and column 'salary' decreasing.

The nature of data sets used in gradual pattern mining do not provide target labels/classes among their features so that intelligent classification algorithms may be applied on them. Therefore, most of the existing gradual pattern mining techniques rely on optimized algorithms for the purpose of mining gradual patterns. In order to allow the possibility of employing machine learning algorithms to the task of classifying gradual patterns, the need arises for labelling features of data sets. First, we propose an approach for generating gradual pattern labels from existing features of a data set. Second, we introduce a technique for extracting estimated gradual patterns from the generated labels. Our experiments show that our approaches mine gradual patterns with high efficiency and satisfactory accuracy.

In this study, we propose an approach that produces GP labels for data set features. In order to test the effectiveness of our approach, we further propose and demonstrate how these labels may be used to extract estimated GPs with an acceptable accuracy. We test the accuracy of the estimated GPs using 2 measures:

* verity of each estimated pattern
* error margin of their estimated support values from the *true* values

# Demonstration

## 1. Viewing original data set

We show the first 5 records of our test data sets *"breast_cancer.csv'' or "c2k.csv"*.

|Age   | BMI         | Glucose | Insulin | HOMA        | Leptin  | Adiponectin | Resistin | MCP.1   | Classification |
| -----| ----------- | ------- | ------- | ----------- | ------- | ----------- | -------- | ------- | -------------- |
| 48.0 | 23.5        | 70.0    | 2.707   | 0.467408667 | 8.8071  | 9.7024      | 7.99585  | 417.114 | 1.0 |  |
| 83.0 | 20.69049454 | 92.0    | 3.115   | 0.706897333 | 8.8438  | 5.429285    | 4.06405  | 468.786 | 1.0 |  |
| 82.0 | 23.12467037 | 91.0    | 4.498   | 1.009651067 | 17.9393 | 22.43204    | 9.27715  | 554.697 | 1.0 |  |
| 68.0 | 21.36752137 | 77.0    | 3.226   | 0.612724933 | 9.8827  | 7.16956     | 12.766   | 928.22  | 1.0 |  |
| 86.0 | 21.11111111 | 92.0    | 3.549   | 0.8053864   | 6.6994  | 4.81924     | 10.57635 | 773.92  | 1.0 |  |


## 2. GP labelling

We show the modified data set with the generated GP labels.

|Age   | BMI         | Glucose | Insulin | HOMA        | Leptin  | Adiponectin | Resistin | MCP.1   | Classification | GP Label |
| -----| ----------- | ------- | ------- | ----------- | ------- | ----------- | -------- | ------- | -------------- | ------- |
| 48.0 | 23.5        | 70.0    | 2.707   | 0.467408667 | 8.8071  | 9.7024      | 7.99585  | 417.114 | 1.0 | 1+2+3+4+5+6+7-8+ |
| 83.0 | 20.69049454 | 92.0    | 3.115   | 0.706897333 | 8.8438  | 5.429285    | 4.06405  | 468.786 | 1.0 | 1-2+4+5+6+7+8+ |
| 82.0 | 23.12467037 | 91.0    | 4.498   | 1.009651067 | 17.9393 | 22.43204    | 9.27715  | 554.697 | 1.0 | 1-2+4+5+7-8+ |
| 68.0 | 21.36752137 | 77.0    | 3.226   | 0.612724933 | 9.8827  | 7.16956     | 12.766   | 928.22  | 1.0 | 1-2+3+4+5+6+7+9- |
| 86.0 | 21.11111111 | 92.0    | 3.549   | 0.8053864   | 6.6994  | 4.81924     | 10.57635 | 773.92  | 1.0 | 1-2+4+5+6+7+9- |

## 3. Estimate GPs using labels
We present the preliminary results that show the accuracy of our extracted estimated GPs.

### a. breast_cancer data set

| Gradual Pattern    | Estimated Support  |  True Support | Percentage Error | Standard Deviation  |
| ------------------ | ------------------ | ------------- | ---------------- | ------------------- |
| ['2+', '8-'] | 0.216 | 0.871 | -75.201% | 0.463 |
| ['0+', '2+'] | 0.216 | 0.836 | -74.163% | 0.438 |
| ['7-', '8-'] | 0.5 | 0.931 | -46.294% | 0.305 |
| ['5-', '8-'] | 0.224 | 0.966 | -76.812% | 0.525 |
| ['7-', '5-'] | 0.207 | 0.922 | -77.549% | 0.506 |
| ['7-', '4-'] | 0.414 | 0.94  | -55.957% | 0.372 |
| ['5-', '4-'] | 0.509 | 0.966 | -47.308% | 0.323 |
| ['7-', '3-'] | 0.216 | 0.905 | -76.133% | 0.487 |
| ['5-', '3-'] | 0.5 | 0.845  | -40.828% | 0.244 |
| ['4-', '3-'] | 0.759 | 0.897 | -15.385% | 0.098 |
| ['7-', '4-', '3-'] | 0.207 | 0.862 | -75.986% | 0.463 |
| ['5-', '4-', '3-'] | 0.483 | 0.802 | -39.776% | 0.226 |
| ['1-', '8-'] | 0.466 | 0.862 | -45.94% | 0.28 |
| ['7-', '1-'] | 0.414 | 0.836 | -50.478% | 0.298 |
| ['5-', '1-'] | 0.569 | 0.888 | -35.923% | 0.226 |
| ['4-', '1-'] | 0.457 | 0.905 | -49.503% | 0.317 |
| ['3-', '1-'] | 0.466 | 0.81  | -42.469% | 0.243 |
| ['4-', '3-', '1-'] | 0.431 | 0.733 | -41.201% | 0.214 |
| ['2+', '4+'] | 0.25 | 0.897|  -72.129% | 0.457 |
| ['2+', '3+', '4+'] | 0.224 | 0.828 | -72.947% |  0.427 |
| ['8-', '6+'] | 0.44 | 0.957|  -54.023% |      0.366 |
| ['7-', '6+'] | 0.448 | 0.914 | -50.985% |      0.33 |
| ['1-', '6+'] | 0.422 | 0.879 | -51.991% |      0.323 |


### b. c2k data set

| Gradual Pattern  |   Estimated Support |  True Support | Percentage Error   | Standard Deviation |
| ------------------ | ----------------- | ------------- |  ----------------- | ------------------ |
| ['2+', '1-']  | 0.565  | 0.955 | -40.838%  |      0.276 |
| ['1+', '2+']  | 0.5  |   0.955 | -47.644%  |      0.322 |
| ['3+', '1-']  | 0.54  |  0.9   | -40.0%  |        0.255 |
| ['1+', '3+']  | 0.5  |   0.9   | -44.444%  |      0.283 |
| ['2+', '3+']  | 0.6  |   0.905 | -33.702%  |      0.216 |
| ['5+', '1-']  | 0.505  | 0.885 | -42.938%  |      0.269 |
| ['2+', '5+']  | 0.59  |  0.93  | -36.559%  |      0.24 |
| ['3+', '5+']  | 0.535  | 0.9   | -40.556%  |      0.258 |
| ['2+', '3+', '5+'] | 0.515  | 0.855 | -39.766%  |      0.24 |
| ['7+', '2+']  | 0.535  | 0.905 | -40.884%  |      0.262 |
| ['7+', '3+']  | 0.525  | 0.95  | -44.737%  |      0.301 |
| ['7+', '5+']  | 0.5  |   0.91  | -45.055%  |      0.29 |
| ['8+', '2+']  | 0.51  |  0.91  | -43.956%  |      0.283 |
| ['8+', '3+']  | 0.53  |  0.92  | -42.391%  |      0.276 |
| ['7+', '8+']  | 0.585  | 0.885 | -33.898%  |      0.212 |
| ['7+', '8+', '2+'] | 0.5  |   0.765 | -34.641%  |      0.187 |
| ['7+', '8+', '3+'] | 0.505  | 0.79  | -36.076%  |      0.202 |