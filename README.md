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
### Dataset Description
We used data from https://www.kaggle.com/datasets/kaggle/college-scorecard. It provides a detailed examination of higher education institutions across the U.S. in financial and student outcomes aspects. It includes 6,543 rows and 3,232 columns, offering key metrics such as costs, debts, and post-college earnings for students who received federal financial aid. 
### Data Cleaning
In preparing the College Scorecard dataset for Loan Repayment Prediction, we meticulously refined the initial set of 3,232 features down to 16 essential features. This reduction was achieved by eliminating non-numeric columns that are not relevant to university institutions, discarding columns unrelated to earnings, and synthesizing insights from related research. Additionally, we consolidated financial metrics by combining 'COSTT4_A' (Average cost of attendance for academic year institutions) and 'COSTT4_P' (Average cost of attendance for program-year institutions) into a single feature, 'ATDCOST'. To address missing values, we implemented the IterativeImputer package, which initializes missing entries with an initial guess (typically the mean or median) and subsequently employs the Bayesian Ridge estimator to model each feature with missing data as a dependent variable while utilizing the other features as predictors. This approach ensures a robust dataset that is primed for developing accurate predictive models.

## Reference
Chetty, Raj, et al. NBER WORKING PAPER SERIES MOBILITY REPORT CARDS: THE ROLE of COLLEGES in INTERGENERATIONAL MOBILITY. 2017.

“Data Home | College Scorecard.” Collegescorecard.ed.gov, collegescorecard.ed.gov/data.

Hilbert, Sven, et al. “Machine Learning for the Educational Sciences.” Review of Education, vol. 9, no. 3, Oct. 2021, https://doi.org/10.1002/rev3.3310.

Luo, Bin, and Qi Zhang. “Data-Driven Exploration of Factors Affecting Federal Student Loan Repayment.” Ar5iv, 26 Feb. 2024, ar5iv.labs.arxiv.org/html/1805.01586. Accessed 8 May 2024.
