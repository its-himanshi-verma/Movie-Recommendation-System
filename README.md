# MovieLens Movie Recommendation Dataset

## 📌 Overview

The **MovieLens `ml-latest-small` dataset** contains movie ratings and user-generated tags collected from [MovieLens](https://movielens.org), a movie recommendation platform operated by GroupLens Research at the University of Minnesota.

This dataset is commonly used for learning and experimenting with **movie recommendation systems, collaborative filtering, exploratory data analysis (EDA), and machine learning**.

### Dataset Statistics

| Attribute | Details |
|---|---|
| Dataset Name | MovieLens `ml-latest-small` |
| Number of Ratings | 100,836 |
| Number of Movies | 9,742 |
| Number of Users | 610 |
| Number of Tag Applications | 3,683 |
| Rating Scale | 0.5 to 5.0 stars |
| Number of Files | 4 CSV files |
| Data Collection Period | March 29, 1996 – September 24, 2018 |
| Dataset Generated On | September 26, 2018 |
| Data Format | CSV |
| Text Encoding | UTF-8 |

> **Note:** This is a development dataset intended for experimentation and education. It is not recommended for reporting shared research benchmarks.

## 🎯 Project Objectives

This dataset can be used to:

- Build a personalized movie recommendation system.
- Recommend movies based on users' previous ratings.
- Identify similar users and movies.
- Explore movie genres and rating patterns.
- Implement collaborative filtering and content-based filtering.
- Analyze user preferences and movie popularity.
- Evaluate recommendation algorithms using appropriate validation techniques.

## 📂 Dataset Structure

The dataset contains four CSV files.

### 1. `movies.csv`

Contains movie information, including titles and genres.

| Column | Description |
|---|---|
| `movieId` | Unique identifier for each movie |
| `title` | Movie title, usually including its release year |
| `genres` | Genres associated with the movie, separated by `\|` |

**Example:**

```csv
movieId,title,genres
1,Toy Story (1995),Adventure|Animation|Children|Comedy|Fantasy
2,Jumanji (1995),Adventure|Children|Fantasy
```

### 2. `ratings.csv`

Contains user ratings for movies.

| Column | Description |
|---|---|
| `userId` | Anonymized user identifier |
| `movieId` | Identifier of the rated movie |
| `rating` | Rating from 0.5 to 5.0 stars |
| `timestamp` | Rating time represented as Unix time in seconds |

**Example:**

```csv
userId,movieId,rating,timestamp
1,1,4.0,964982703
1,3,4.0,964981247
```

Each row represents one rating given by a user to a movie.

### 3. `tags.csv`

Contains user-generated tags describing movies.

| Column | Description |
|---|---|
| `userId` | Identifier of the user who added the tag |
| `movieId` | Identifier of the tagged movie |
| `tag` | User-generated keyword or phrase |
| `timestamp` | Time when the tag was applied, in Unix seconds |

**Example:**

```csv
userId,movieId,tag,timestamp
2,60756,funny,1445714994
2,89774,aliens,1445715207
```

Tags can provide additional information about movie themes, styles, and user perceptions.

### 4. `links.csv`

Contains identifiers linking MovieLens movies to external movie databases.

| Column | Description |
|---|---|
| `movieId` | MovieLens movie identifier |
| `imdbId` | Corresponding IMDb identifier |
| `tmdbId` | Corresponding The Movie Database identifier |

**Example:**

```csv
movieId,imdbId,tmdbId
1,114709,862
2,113497,8844
```

These identifiers can be used to connect movies with external metadata, subject to the relevant providers' terms of use.

## 🧠 Recommendation System Approaches

### 1. Popularity-Based Recommendation

Recommends movies based on popularity, rating averages, or rating counts.

**Example:** Recommend highly rated movies that have received a sufficient number of ratings.

### 2. Collaborative Filtering

Uses patterns in user ratings to identify similar users or movies.

- **User-based filtering:** Recommends movies liked by users with similar rating patterns.
- **Item-based filtering:** Recommends movies similar to those a user has rated highly.
- **Matrix factorization:** Learns latent user and movie features from the rating matrix.

### 3. Content-Based Filtering

Uses movie attributes, such as genres and potentially tags, to recommend movies similar to a user's preferences.

### 4. Hybrid Recommendation

Combines collaborative filtering and content-based filtering to leverage both rating behavior and movie characteristics.

## 🛠️ Technologies You Can Use

- **Python** — Programming language
- **Pandas** — Data loading and manipulation
- **NumPy** — Numerical operations
- **Matplotlib / Seaborn** — Data visualization
- **Scikit-learn** — Machine learning and evaluation utilities
- **Surprise** — Recommendation algorithms
- **Jupyter Notebook** — Interactive experimentation

These are suggested tools; they are not requirements imposed by the dataset.

## 🚀 Getting Started

### Step 1: Download the Dataset

Download the `ml-latest-small` dataset from the official GroupLens website:

[Download MovieLens Datasets](https://grouplens.org/datasets/movielens/)

Extract the archive into your project directory.

### Step 2: Install Dependencies

Install the basic Python libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Step 3: Load the Dataset

```python
import pandas as pd

movies = pd.read_csv("data/movies.csv")
ratings = pd.read_csv("data/ratings.csv")
tags = pd.read_csv("data/tags.csv")
links = pd.read_csv("data/links.csv")

print("Movies:", movies.shape)
print("Ratings:", ratings.shape)
print("Tags:", tags.shape)
print("Links:", links.shape)
```

Update the file paths if your CSV files are stored in a different directory.

### Step 4: Explore the Data

```python
# Display sample records
print(movies.head())
print(ratings.head())

# Check missing values
print(movies.isnull().sum())
print(ratings.isnull().sum())

# Calculate average movie ratings
movie_ratings = ratings.groupby("movieId")["rating"].agg(
    ["mean", "count"]
)

print(movie_ratings.sort_values("mean", ascending=False).head(10))
```

**Important:** When ranking movies, consider setting a minimum rating-count threshold. A movie with one five-star rating should not automatically outrank a movie with hundreds of consistently high ratings.

## 📊 Potential Analysis

You can investigate the following questions:

1. Which movies have the highest average ratings?
2. Which movies receive the most ratings?
3. What are the most common movie genres?
4. How do rating distributions vary across users?
5. Which movies are frequently rated similarly?
6. Can user rating patterns predict ratings for unseen movies?
7. How does collaborative filtering compare with content-based filtering?

## 📏 Model Evaluation

For rating-prediction models, useful metrics include:

- **RMSE (Root Mean Squared Error):** Measures the magnitude of prediction errors, penalizing larger errors more heavily.
- **MAE (Mean Absolute Error):** Measures the average absolute difference between predicted and actual ratings.

For top-N recommendation systems, consider:

- Precision@K
- Recall@K
- NDCG@K

Use a suitable train-test split and avoid data leakage. For recommendation systems, a time-based split may be useful when the goal is to predict future user behavior.

## ⚠️ Dataset Limitations

- The dataset contains anonymized user IDs but no demographic information.
- The data are historical and were generated in 2018.
- Ratings represent only the preferences recorded by the included users.
- Not every user has rated every movie, resulting in a sparse user-item rating matrix.
- Movie titles and metadata may contain errors or inconsistencies.
- The dataset is intended for development and experimentation, not as a shared research benchmark.
- Recommendations built from these data may not represent the preferences of the general population.

## 📜 License and Usage

The dataset is provided by GroupLens Research at the University of Minnesota.

According to the dataset's usage terms:

- Do not imply endorsement by the University of Minnesota or GroupLens Research.
- Acknowledge the dataset in publications resulting from its use.
- Redistribution is permitted under the same license conditions.
- Commercial or revenue-generating use requires prior permission from a faculty member of the GroupLens Research Project.

The dataset is supplied without guarantees regarding accuracy, suitability, or fitness for a particular purpose.

Before using or redistributing the data, review the original terms at the official source:

[MovieLens Dataset Terms and Downloads](https://grouplens.org/datasets/movielens/)

## 📚 Citation

If you use this dataset in a publication, cite the following paper:

Harper, F. Maxwell, and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context.* ACM Transactions on Interactive Intelligent Systems (TiiS), 5(4), Article 19, pages 1–19.

https://doi.org/10.1145/2827872

## 🔗 Useful Links

- [MovieLens](https://movielens.org/)
- [GroupLens Research](https://grouplens.org/)
- [MovieLens Dataset Downloads](https://grouplens.org/datasets/movielens/)
- [IMDb](https://www.imdb.com/)
- [The Movie Database (TMDB)](https://www.themoviedb.org/)

## 👩‍💻 Contributing

If you are using this dataset to develop a recommendation system, you can extend the project by adding:

- Data cleaning and exploratory analysis.
- Interactive rating and genre visualizations.
- Collaborative filtering algorithms.
- Content-based and hybrid recommendation models.
- Recommendation evaluation and comparison.
- A web application for personalized movie suggestions.

## 📄 Disclaimer

This repository describes the MovieLens dataset and possible applications. The dataset is maintained by GroupLens Research, and this repository does not imply any official affiliation with or endorsement by GroupLens or the University of Minnesota.


