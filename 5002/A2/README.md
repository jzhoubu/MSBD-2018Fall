# Ensemble Learning
Applying Ensemble Learning (Stacking) to Data Mining. 

## Usage
- Run `main.py`
    - Change variable `PATH` to the path where stores `testFeatures.csv`, `trainFeatures.csv`, `trainLabels.csv`.
    - The whole training procedure takes about 2 hour with procedure log , tested in Windows10 on my laptop.
    - The predicition, `A2_itsc_20556794_prediction.csv `, will automatically store in `PATH`.

## Feature Engineering
- One hot encoding for categorical feature 
- WoE transforation for categorical feature 
- Replace missing value with `np.nan`. [Removed] 
- Non-linear operation applying on Top10 important feature, eg. `capital-gain`/`capital-loss` 

|  Feature Name | Generate from  |  type | destription  |   
|:---:|:---:|:---:|:---:|
| workclass  | workclass | float  |  WoE encode |   
| education | education | float | WoE encode |   
| Marital-status | Marital-status | float | WoE encode |   
| race | race  | float | WoE encode |   
| sex | sex | float | WoE encode |   
| relationship | relationship | float | WoE encode |   
| occupation | occupation | float | WoE encode |   
| native-country | native-country | float |  WoE encode |   
| workclass_*  | workclass | float  |  One Hot encode  |   
| education_* | education | float | One Hot encode |   
| Marital-status_* | Marital-status | float |One Hot encode |   
| race_* | race  | float | One Hot encode |   
| sex_* | sex | float | One Hot encode |   
| relationship_* | relationship | float | One Hot encode |   
| occupation_* | occupation | float | One Hot encode |   
| native-country_* | native-country | float |  One Hot encode |   
| capital-gain_div_capital-loss | capital-gain, capital-loss | float | Non-Linear Operation, a/b | 
|capital-gain_div_age | capital-gain, age | float | Non-Linear Operation, a/b | 
| capital-gain_div_hours-per-week | capital-gain, hours-per-week | float | Non-Linear Operation, a/b | 
| capital-gain_div_fnlwgt | capital-gain, fnlwgt | float | Non-Linear Operation, a/b | 
| capital-loss_div_age | capital-loss, age | float | Non-Linear Operation, a/b | 
| capital-loss_div_hours-per-week | capital-loss, hours-per-week | float | Non-Linear Operation, a/b | 
| capital-loss_div_fnlwgt | capital-loss, fnlwgt | float | Non-Linear Operation, a/b | 
| age_div_hours-per-week | age, hours-per-week | float | Non-Linear Operation, a/b | 
| age_div_fnlwgt | age, fnlwgt | float | Non-Linear Operation, a/b | 
| hours-per-week_div_fnlwgt | hours-per-week, fnlwgt | float | Non-Linear Operation, a/b | 



## Consideration
- Missing Value Issue
> Missing value show up in three categorical feature. Indeed, transform missing value to `np.nan` and let LightGBM optimize spliting is quite enough, I believe. However, with 5-fold CV experiments, I find ohe can do better job.
- Categorical feature aggregation
> WOE transformation do the aggregation job, it considers the joint distribution of feature `x` and label `y`. I think data-driven is better than human knowledge driven.


## Data Mining Log
1.  GlanceData
> Quick glance at distribution on data
    - figure out distribution of missing value
    - figure out names of continuous features and categorical features.

##### Categorical Feature
<table border=0 >
    <tbody>
        <tr>
            <td width="40%" > <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_cate1.png"> </td>
            <td width="40%"> <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_cate2.png"> </td>
        </tr>
    </tbody>
</table>

##### Continous Feature
<table border=0 >
    <tbody>
        <tr>
            <td width="20%" > <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_num1.png"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_num2.png"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_num3.png"> </td>
            <td width="20%"> <img src="https://github.com/sysu-zjw/BDT-Homework/blob/master/img/5002A2_num4.png""> </td>
        </tr>
    </tbody>
</table>



2. FirstTrain
> Several machine learning models were trained to check out performance on 5fold-CV, such as AdaBoost, Gradient Boosting Classifier, Random Forest Classifier, and LightGBM.
    - Select LightGBM as main model because of better performance on 5fold-CV.

3. FeatureEngineering&SecondTrain
    - One hot encoding for categorical features
    - Weight of Evidence encoding for for categorical features. Tested with **better performance** than ohe
    - Both ohe and woe were used for feature engineering, in order to enlarge features, though WoE transformation would cover most importance of ohe feature.

4. ImportantFeatureEngineering
> With WoE transformation, it would be possible to compare **feature- importance** no matter the feature continuous feature or not. 
    - I generated non-linear features with Top 10 important feature by dividing them each other, like this `feature1_div_feature2`

5. DetectFake
> Take a glance on joint distribution of related feature. Then to decide whether some samples provide fake information.

6. Stacking
> Use 5 fold stack for ensemble learning.