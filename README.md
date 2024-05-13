# QTM347-Final-Project

## Introduction
This project seeks to investigate the factors related to college students and their educational institutions that may predict education loan repayment. With US college tuition at historically high levels, many students graduate saddled with substantial debt that often exceeds their financial capacity relative to their post-graduation income. This issue has gained significant political traction recently, highlighted by President Biden's loan forgiveness plan. Critics argue that broad debt relief initiatives could disproportionately benefit affluent individuals, who may not require financial aid. The core issue remains: does loan forgiveness truly assist those who need it most, such as students from lower-income backgrounds who see higher education as a pathway to upward social mobility? 
In a working paper, the National Bureau of Economic Research highlighted that some institutions are better than others at providing economic mobility to low-income students. However, this approach remains limited to correlations without identifying specific characteristics influencing loan repayment.
### Literature Review
Machine learning techniques offer transformative potential in educational research by handling high-dimensional datasets that challenge traditional methods. Studies such as Hilbert et al. (2021) highlight machine learning's ability to model intricate relationships within substantial, varied educational data, revealing socioeconomic factors influencing student loan repayment.
Hilbert et al. (2021) provide a comprehensive overview of supervised and unsupervised models relevant to education data, helping tailor predictive algorithms. Their insights on non-parametric models that avoid predefined functional forms are particularly helpful in understanding non-linear repayment factors. In light of this, we included KNN and decision tree methods in our models. 
Recent studies further illuminate factors affecting repayment. For instance, an arXiv paper used an Elastic Net model for linear regression to predict repayment rates (Luo and Zhang). So we included Ridge and Lasso regression into our study.  
### Motivation
The issue of student debt is socio-economic, with significant implications for equitable access to higher education. Understanding repayment dynamics across demographic groups will inform policies that ensure education remains accessible and financially sustainable. By identifying key repayment predictors, stakeholders can tailor aid and forgiveness schemes to better assist those in need, ensuring an equitable educational landscape.

## Set Up
### Dataset Description
We used data from https://www.kaggle.com/datasets/kaggle/college-scorecard. It provides a detailed examination of higher education institutions across the U.S. in financial and student outcomes aspects. It includes 6,543 rows and 3,232 columns, offering key metrics such as costs, debts, and post-college earnings for students who received federal financial aid. Amoung those 3232 columns, we picked 'RPY_3YR_RT' (Fraction of cohort who are not in default and whose loan balances have declined for three years) as the target variable.
### Data Cleaning
In preparing the College Scorecard dataset for Loan Repayment Prediction, we meticulously refined the initial set of 3,232 features down to 16 essential features. This reduction was achieved by eliminating non-numeric columns that are not relevant to university institutions, discarding columns unrelated to earnings, and synthesizing insights from related research. Additionally, we consolidated financial metrics by combining 'COSTT4_A' (Average cost of attendance for academic year institutions) and 'COSTT4_P' (Average cost of attendance for program-year institutions) into a single feature, 'ATDCOST'. To address missing values, we implemented the IterativeImputer package, which initializes missing entries with an initial guess (typically the mean or median) and subsequently employs the Bayesian Ridge estimator to model each feature with missing data as a dependent variable while utilizing the other features as predictors. This approach ensures a robust dataset that is primed for developing accurate predictive models, particularly with regards to SAT data. The IterativeImputer successfully filled 5,371 NaN values out of 6,543 rows while preserving the original distribution characteristics of the SAT scores.

![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/Before_IterativeImputer_SAT.png)
![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/After_IterativeImputer_SAT.png)

Next, we further refined the dataset by conducting a thorough correlation analysis. By utilizing both correlation plots and heat maps, we assessed the relationships between each feature and the target variable. We established a threshold to select only those features with an absolute correlation close to or exceeding 0.5. This analysis resulted in the successful filtration of 9 features, ultimately retaining 7 key features for our predictive modeling. These final features include "HIGHDEG," "MD_EARN_WNE_P8," "DEP_INC_AVG," "UG25ABV," "PAR_ED_PCT_PS," "RPY_3YR_RT," and "ATDCOST."

- Before Feature Selection
![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/Before_cleaning_corr.png)
![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/Before_cleaning_heatmap.png)
- After Feature Selection 
![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/After_cleaning_corr.png)
![](https://github.com/VEG0258/QTM347-Final-Project/blob/main/Data%20Cleaning/After_cleaning_heatmap.png)
### Problem Setup
For our analysis of the College Scorecard dataset, we utilized Python Version 3.9. Besides basic packages, such as Pandas, NumPy, Matplotlib, and Seaborn, for data manipulation, numerical calculations, and visualization, our modeling process leveraged Scikit-learn to provide comprehensive functionalities for implementing and evaluating machine learning models. Additionally, FancyImpute was used for advanced imputation methods like IterativeImputer to ensure robust handling of missing data across the dataset. The specific Methodologies utilized are listed below:

- ##### Lasso/Ridge Regression:
  - We will apply regularization techniques to refine our regression models. These techniques are particularly useful in preventing overfitting and improving model generalizability by penalizing the coefficients of the regression model.
  - Lambda Values: We will test a range of lambda values from 5 to -5, with a total of 101 points evenly distributed across this range.
- ##### K-Nearest Neighbors (KNN):
  - KNN will be used to predict outcomes based on the closest data points in the feature space.
  - K Values: We will evaluate K values from 1 to 15 to find the optimal number of nearest neighbors that balances bias and variance.
- ##### Partial Least Squares (PLS):
  - PLS is suitable for datasets with many collinear variables because it projects the predictors into a new space formed by orthogonal components, which are then used to predict the response.
  - Lambda Values: We will test a range of lambda values from 5 to -5, with a total of 101 points evenly distributed across this range.
- ##### Decision Trees:
  - Decision Trees will be explored for their intuitive understanding and ease of use. By partitioning the space into regions that minimize the prediction error, they offer clear insights into feature importance and decision-making.
  - Maximum Depth: We will vary the maximum depth from 1 to 10 to understand the depth at which the trees best generalize from the training data without overfitting.


## Reference
Chetty, Raj, et al. NBER WORKING PAPER SERIES MOBILITY REPORT CARDS: THE ROLE of COLLEGES in INTERGENERATIONAL MOBILITY. 2017.

“Data Home | College Scorecard.” Collegescorecard.ed.gov, collegescorecard.ed.gov/data.

Hilbert, Sven, et al. “Machine Learning for the Educational Sciences.” Review of Education, vol. 9, no. 3, Oct. 2021, https://doi.org/10.1002/rev3.3310.

Luo, Bin, and Qi Zhang. “Data-Driven Exploration of Factors Affecting Federal Student Loan Repayment.” Ar5iv, 26 Feb. 2024, ar5iv.labs.arxiv.org/html/1805.01586. Accessed 8 May 2024.
