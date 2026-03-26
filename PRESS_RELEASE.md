<a name="press-release-project1"></a>

# Smarter Movie Recommendations: Collaborative Filtering Beats a Naive Average Baseline by About 15% on Rating Error

## Hook
When endless scrolling still lands you on something you have already seen, the system is usually optimizing a generic popularity story, not your personal pattern. Using MovieLens 25M, we show that collaborative filtering beats that strawman: it predicts star ratings with roughly 15% lower error than predicting one dataset-wide average for everyone. This way, predictions follow individual taste better than a naive average.

## Problem Statement
Streaming platforms face a personalization challenge at scale: with huge catalogs and diverse users, overly simple strategies create three recurring issues. First, **popularity bias** can dominate recommendations because popular titles have the most data, which can crowd out niche matches for individual taste. Second, **cold start** remains hard for brand-new users and new titles with little history, where defaults often fall back to generic “popular picks” instead of personalized fits. Third, **preferences change over time**, so treating older ratings the same as recent ones can misrepresent what a user wants today. The MovieLens 25M dataset illustrates the difficulty of personalization: ratings are sparse per user, many items are rarely rated (long-tail behavior), and the collection spans many years (1995-2019), so static shortcuts can miss evolving taste. A common strawman baseline in explicit-feedback rating prediction is to guess the **global average rating** for every prediction; in our pipeline evaluation on a sampled interaction set, that baseline lands around **RMSE 1.06** on held-out ratings, leaving clear headroom for collaborative filtering models that learn user-specific and item-specific structure rather than a single number for everyone.

## Solution Description

Our approach learns from real **user–movie ratings** instead of guessing the same “typical” score for everyone. The idea is simple: people who agree on some movies often agree on others, and movies that get similar reactions from many viewers often behave alike in the data—even when each person has rated only a handful of titles. The model turns those patterns into personalized predictions: a forecast of how many stars *you* would give a film you have not rated yet.

We compare the personalized approach to a weak baseline that is **always predicting one overall average rating**. Therefore, the improvement is easy to interpret as “learning you” versus “treating everyone the same.” On our evaluation setup, the personalized model tracks real ratings noticeably better than that baseline (about **15%** less error), which is the same story the figure in this release illustrates: structure in the data beats a single-number shortcut.

## Chart

### Understanding the Visualization

![MovieLens 25M: personalized collaborative filtering vs a global-mean baseline](visualizations/movielens_press_release.png)

*Figure 1: Three-panel figure. Top-left: distribution of the 25M ratings on the 0.5–5.0 star scale with the dataset mean marked. Top-right: held-out prediction error for a global-mean baseline versus SVD (RMSE and MAE), with a short summary of 3-fold cross-validation. Bottom: monthly rating activity over the collection period.*

The figure connects the press release narrative to measurable evidence: the rating data are not “one number,” a naive average is easy to beat with structure, and activity is spread across many years (so personalization and careful evaluation matter).

**Top-left — Rating distribution:** Counts of star ratings from **0.5–5.0** stars; most ratings fall between **3 and 5**, with a peak around **4.0**. The dashed line is the dataset **mean (~3.53)**—a single “typical” score hides a lot of variation across users and movies.

**Top-right — Prediction error (held-out data):** Compares predicting **every** test rating with the **global mean** versus **SVD** collaborative filtering. **Blue = RMSE**, **orange = MAE** (lower is better). SVD clearly beats the baseline; the figure annotates about **15%** better RMSE versus the global mean, with **3-fold cross-validation** in the same range so the result is not a one-off split.

**Bottom — Activity over time:** **Ratings per month** from the mid-1990s into 2020, with uneven peaks—reminder that the data span many years of participation, not a single static “average taste” moment.

**Takeaway:** Ratings cluster high but are not one number for everyone; a personalized model reduces error versus always guessing the dataset average.