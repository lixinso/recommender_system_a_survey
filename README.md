# Recommender System - A Survey

Xinsong Li, Dec 2020

[Public Repo. Unfinished article]

# Abstract

This survey provided a comprehensive summary about the recommender system state-of-art knowledge.

Recommender Systems usually classified into 2 types by 
- Content Based Filtering
- Collabrative Filtering

Depends on if the model is learned from underlying data, there are 2 types,
- Model Based 
- Memory Based

# Use Cases

- Movie Recommendation
i.e., Netflix

- Music Recommendation
i.e., Last.fm, Pandora Radio

- Product Recommendation
i.e., Amazon

- News Recommendation
i.e., Google News, Toutiao

- People Recommendation
i.e., LinkedIn

# System Architecture

![https://netflixtechblog.com/system-architectures-for-personalization-and-recommendation-e081aa94b5d8](https://miro.medium.com/max/1400/1*qqTSkHNOzukJ5r-b54-wJQ.png)

# Algorithms

## Cold Start Problem

## Collaborative Filtering

Collabrative Filtering is best suited to problem with known data on users, but lack of data for items or lack of feature extraction for items of interest.  [[6]](https://towardsdatascience.com/recommendation-systems-a-review-d4592b6caf4b)

Collabrative Filtering approaches build a model from user's past behavior (items previously purchased or selectec/rated) as well as similar decision made by others. [[10]](https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems)

## Content-Based Filtering	

## Knowledge-Based System

## Hybrid

## Item-Item

## User-Item

## User-User

## LDA

## Ranking

## Tag 

## Category

## Trending

## Feedback System

## Hybrid Approach 

## Deep Learning Approach

## [ALS - Alternating Least Squares](https://github.com/microsoft/recommenders/blob/master/examples/00_quick_start/als_movielens.ipynb)

## xTreme Deep Factorization Machines (xDeepFM)

# Reinforcement Learning for Recommender Systems

# Evaluation Metrics

## Rating Metrics

- Root Mean Square Error (RMSE)
Measure of average error in predicted ratings

- R Square (R^2)
Essentially how much of the total variation is explained by the model

- Mean Absolute Error (MAE)


- Explained Variance - 
How much of the variance in the data is explained by the model

## Ranking Metrics

- Precision
The proportion of recommended items that are relevant

- Recall
Measures the proportion of relevant items that are recommended

- Normalized  Discounted Cumulative Gain (NDCG)
Evaluates how well the predicted items for a user are ranked based on the relevance

- Mean Average Precision (MAP)
Average precision for each user normalized over all users

## Classification Metrics

- Area Under Curve (AUC) 
Integral area under the receiver operating characteristic curve


- Logistic Loss (Logloss)
The negative log-likelihood of the true labels given the prediction of a classifier


# Model Selection and Optimization


# Sparsity

The number of items sold on majority e-commerce sites is extremely large. The most active users will only have rated a small subset of the overall database. Thus, even the most popular items have very few ratings. [[10]](https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems)

# Industry Practice

## Last.fm

## Pandora Radio

# References

[1] http://ijcai13.org/files/tutorial_slides/td3.pdf

[2] [Hulu](https://web.archive.org/web/20170406065247/http://tech.hulu.com/blog/2011/09/19/recommendation-system.html)

[3] Xavier Amatriain and Justin Basilico. [System Architectures for Personalization and Recommendation](https://link.medium.com/PaefDwO9bab) (by Netflix Technology Blog) 

[4] https://github.com/mandeep147/Amazon-Product-Recommender-System

[5] https://github.com/smwitkowski/Amazon-Recommender-System

[6] Recommendation Systems: A Review https://towardsdatascience.com/recommendation-systems-a-review-d4592b6caf4b

[7] [Recommender system using Bayesian personalized ranking](https://towardsdatascience.com/recommender-system-using-bayesian-personalized-ranking-d30e98bba0b9)

[8] [Introduction to Recommender Systems in 2019](https://tryolabs.com/blog/introduction-to-recommender-systems/)

[9] [Introduction to recommender systems](https://towardsdatascience.com/introduction-to-recommender-systems-6c66cf15ada)

[10] https://en.wikipedia.org/wiki/Recommender_system#Mobile_recommender_systems

[11] Book: 推荐系统实践 项亮

[12] Book: Recommender systems:An introduction, Dietmar Jannach / Markus Zanker / Alexander Felfernig / Gerhard Friedrich, 2013

[14] [Robin Burke Recommender Systems: An Overview](https://www.researchgate.net/publication/220604600_Recommender_Systems_An_Overview)

[15] https://github.com/microsoft/recommenders

[16] https://github.com/kevinmcmahon/tagger

[17] https://github.com/timlyo/pyTag

[18] [Hybrid Recommendation Approaches](https://www.math.uci.edu/icamp/courses/math77b/lecture_12w/pdfs/Chapter%2005%20-%20Hybrid%20recommendation%20approaches.pdf)

[19] https://towardsdatascience.com/mixed-recommender-system-mf-matrix-factorization-with-item-similarity-based-cf-collaborative-544ddcedb330

[20] https://towardsdatascience.com/the-best-forecast-techniques-or-how-to-predict-from-time-series-data-967221811981

[21] [Trend or No Trend: A Novel Nonparametric Method for Classifying Time Series](https://dspace.mit.edu/handle/1721.1/85399)

[22] https://github.com/microsoft/recommenders/blob/master/examples/00_quick_start/als_movielens.ipynb

[23] https://github.com/microsoft/recommenders/tree/master/examples/03_evaluate

[24] Asela Gunawardana and Guy Shani: A Survey of Accuracy Evaluation Metrics of Recommendation Tasks

[25] Dimitris Paraschakis et al, "Comparative Evaluation of Top-N Recommenders in e-Commerce: An Industrial Perspective", IEEE ICMLA, 2015, Miami, FL, USA.

[26] Yehuda Koren and Robert Bell, "Advances in Collaborative Filtering", Recommender Systems Handbook, Springer, 2015.
Chris Bishop, "Pattern Recognition and Machine Learning", Springer, 2006.

[27] Errico, James H., et al. “Collaborative recommendation system.” U.S. Patent №8,949,899. 3 Feb. 2015.

[28] Davidson, James, et al. “The YouTube video recommendation system.” Proceedings of the fourth ACM conference on Recommender systems. 2010.

[29] Adomavicius, Gediminas, and Alexander Tuzhilin. “Toward the next generation of recommender systems: A survey of the state-of-the-art and possible extensions.” IEEE transactions on knowledge and data engineering 17.6 (2005): 734–749.

[30] James Loy, [Deep Learning based Recommender Systems, A gentle introduction to modern movie recommenders](https://towardsdatascience.com/deep-learning-based-recommender-systems-3d120201db7e]
