"""MovieLens item-item recommendation and offline evaluation."""

from __future__ import annotations

import argparse
import math
import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


DATASET_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
RATING_COLUMNS = {"userId", "movieId", "rating", "timestamp"}
MOVIE_COLUMNS = {"movieId", "title", "genres"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent / "data" / "ml-latest-small",
        help="Directory containing MovieLens ratings.csv and movies.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "recommendation_output",
        help="Directory for evaluation and analysis CSV files",
    )
    parser.add_argument("--user-id", type=int, help="User ID to receive recommendations")
    parser.add_argument("--top-n", type=int, default=10, help="Number of movies to recommend")
    parser.add_argument(
        "--neighbors",
        type=int,
        default=50,
        help="Maximum similar movies considered per rated movie",
    )
    return parser.parse_args()


def load_movielens(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load MovieLens CSVs, downloading the small dataset if they are not present."""
    ratings_path = data_dir / "ratings.csv"
    movies_path = data_dir / "movies.csv"
    if not ratings_path.is_file() or not movies_path.is_file():
        data_dir.parent.mkdir(parents=True, exist_ok=True)
        archive_path = data_dir.parent / "ml-latest-small.zip"
        print(f"Downloading MovieLens dataset from {DATASET_URL}")
        with urllib.request.urlopen(
            DATASET_URL,
            context=ssl.create_default_context(cafile=certifi.where()),
            timeout=60,
        ) as response, archive_path.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            for filename in ("ml-latest-small/ratings.csv", "ml-latest-small/movies.csv"):
                target = data_dir / Path(filename).name
                with archive.open(filename) as source, target.open("wb") as destination:
                    destination.write(source.read())

    ratings = pd.read_csv(ratings_path)
    movies = pd.read_csv(movies_path)
    missing_ratings = RATING_COLUMNS.difference(ratings.columns)
    missing_movies = MOVIE_COLUMNS.difference(movies.columns)
    if missing_ratings:
        raise ValueError(f"ratings.csv is missing columns: {', '.join(sorted(missing_ratings))}")
    if missing_movies:
        raise ValueError(f"movies.csv is missing columns: {', '.join(sorted(missing_movies))}")
    if ratings.empty or movies.empty:
        raise ValueError("MovieLens ratings.csv and movies.csv must both contain data")
    if ratings[["userId", "movieId", "rating"]].isna().any().any():
        raise ValueError("ratings.csv contains missing userId, movieId, or rating values")
    if movies[["movieId", "title"]].isna().any().any():
        raise ValueError("movies.csv contains missing movieId or title values")
    return ratings, movies


def temporal_split(
    ratings: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out each user's most recent ratings while leaving at least one for training."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if ratings.empty:
        raise ValueError("Cannot split an empty ratings dataset")
    missing_columns = {"userId", "movieId", "timestamp"}.difference(ratings.columns)
    if missing_columns:
        raise ValueError(f"Ratings are missing columns: {', '.join(sorted(missing_columns))}")
    ordered = ratings.sort_values(["userId", "timestamp", "movieId"])
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in ordered.groupby("userId", sort=False):
        if len(group) < 2:
            train_parts.append(group)
            continue
        test_count = min(len(group) - 1, max(1, math.ceil(len(group) * test_fraction)))
        train_parts.append(group.iloc[:-test_count])
        test_parts.append(group.iloc[-test_count:])
    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train, test


class ItemItemRecommender:
    """Neighborhood recommender using cosine similarity over user-centered ratings."""

    def __init__(self, n_neighbors: int = 50, min_popularity: int = 10) -> None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1")
        if min_popularity < 1:
            raise ValueError("min_popularity must be at least 1")
        self.n_neighbors = n_neighbors
        self.min_popularity = min_popularity
        self._fitted = False

    def fit(self, ratings: pd.DataFrame, movies: pd.DataFrame) -> "ItemItemRecommender":
        if ratings.empty:
            raise ValueError("Cannot fit a recommender without ratings")
        missing_ratings = RATING_COLUMNS.difference(ratings.columns)
        missing_movies = MOVIE_COLUMNS.difference(movies.columns)
        if missing_ratings:
            raise ValueError(f"Ratings are missing columns: {', '.join(sorted(missing_ratings))}")
        if missing_movies:
            raise ValueError(f"Movies are missing columns: {', '.join(sorted(missing_movies))}")

        self.movies = movies.drop_duplicates("movieId").set_index("movieId", drop=False)
        known_ratings = ratings[ratings["movieId"].isin(self.movies.index)].copy()
        if known_ratings.empty:
            raise ValueError("No rated movies match the movie catalog")
        known_ratings["rating"] = pd.to_numeric(known_ratings["rating"], errors="raise")
        user_means = known_ratings.groupby("userId")["rating"].mean()
        centered = known_ratings.join(user_means.rename("user_mean"), on="userId")
        centered["centered_rating"] = centered["rating"] - centered["user_mean"]

        self.user_ratings = known_ratings.groupby(["userId", "movieId"])["rating"].mean().unstack()
        self.user_means = user_means
        self.user_seen = {
            user_id: set(group["movieId"])
            for user_id, group in known_ratings.groupby("userId")
        }
        self.centered = centered.groupby(["userId", "movieId"])["centered_rating"].mean().unstack()
        self.item_ids = self.user_ratings.columns.to_numpy()
        self.item_vectors = self.centered.reindex(columns=self.item_ids).fillna(0).T.to_numpy(dtype=float)
        neighbor_count = min(self.n_neighbors + 1, len(self.item_ids))
        self.neighbor_model = NearestNeighbors(
            n_neighbors=neighbor_count, metric="cosine", algorithm="brute"
        ).fit(self.item_vectors)

        item_stats = known_ratings.groupby("movieId")["rating"].agg(["count", "mean"])
        global_mean = float(known_ratings["rating"].mean())
        prior_weight = self.min_popularity
        item_stats["score"] = (
            item_stats["count"] / (item_stats["count"] + prior_weight) * item_stats["mean"]
            + prior_weight / (item_stats["count"] + prior_weight) * global_mean
        )
        self.popularity_scores = item_stats["score"].to_dict()
        self.popular_items = item_stats.sort_values(
            ["score", "count"], ascending=False
        ).index.to_list()
        self._fitted = True
        return self

    def recommend(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit before requesting recommendations")
        if n < 1:
            raise ValueError("n must be at least 1")
        if user_id in self.user_seen:
            seen = self.user_seen[user_id]
            user_mean = float(self.user_means.loc[user_id])
            rated_items = self.user_ratings.loc[user_id].dropna()
            rated_vectors = self.item_vectors[
                [int(np.where(self.item_ids == movie_id)[0][0]) for movie_id in rated_items.index]
            ]
            distances, indices = self.neighbor_model.kneighbors(rated_vectors)
            weighted_scores: dict[int, float] = {}
            similarity_totals: dict[int, float] = {}
            for source_id, source_rating, item_distances, neighbor_indices in zip(
                rated_items.index, rated_items.values, distances, indices
            ):
                centered_rating = float(source_rating) - user_mean
                for distance, neighbor_index in zip(item_distances, neighbor_indices):
                    candidate_id = self.item_ids[neighbor_index]
                    similarity = max(0.0, 1.0 - float(distance))
                    if candidate_id == source_id or candidate_id in seen or similarity <= 0:
                        continue
                    weighted_scores[candidate_id] = (
                        weighted_scores.get(candidate_id, 0.0) + similarity * centered_rating
                    )
                    similarity_totals[candidate_id] = (
                        similarity_totals.get(candidate_id, 0.0) + similarity
                    )
            scores = {
                movie_id: (
                    similarity_totals[movie_id]
                    * (user_mean + weighted_scores[movie_id] / similarity_totals[movie_id])
                    + self.popularity_scores[movie_id]
                )
                / (similarity_totals[movie_id] + 1.0)
                for movie_id in weighted_scores
            }
            ranked_ids = sorted(scores, key=lambda movie_id: (-scores[movie_id], movie_id))
        else:
            seen = set()
            scores = {}
            ranked_ids = []

        for movie_id in self.popular_items:
            if movie_id not in seen and movie_id not in scores:
                scores[movie_id] = float("nan")
                ranked_ids.append(movie_id)
            if len(ranked_ids) >= n:
                break

        recommendations = self.movies.loc[ranked_ids[:n], ["movieId", "title", "genres"]].copy()
        recommendations.insert(0, "score", [scores[movie_id] for movie_id in recommendations.index])
        return recommendations.reset_index(drop=True)

    def recommend_popular(self, user_id: int, n: int = 10) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Call fit before requesting recommendations")
        if n < 1:
            raise ValueError("n must be at least 1")
        seen = self.user_seen.get(user_id, set())
        movie_ids = [movie_id for movie_id in self.popular_items if movie_id not in seen][:n]
        recommendations = self.movies.loc[movie_ids, ["movieId", "title", "genres"]].copy()
        recommendations.insert(0, "score", float("nan"))
        return recommendations.reset_index(drop=True)


def evaluate_ranking(
    train: pd.DataFrame,
    test: pd.DataFrame,
    movies: pd.DataFrame,
    n: int = 10,
    relevance_threshold: float = 4.0,
    neighbors: int = 50,
) -> pd.DataFrame:
    """Compute macro Precision@K, Recall@K, and NDCG@K over users with relevant test items."""
    if n < 1:
        raise ValueError("n must be at least 1")
    recommender = ItemItemRecommender(n_neighbors=neighbors).fit(train, movies)
    relevant_by_user = (
        test[test["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )
    rows: list[dict[str, float | int | str]] = []
    recommenders = (
        ("Item-item", recommender.recommend),
        ("Popularity baseline", recommender.recommend_popular),
    )
    for algorithm, recommend in recommenders:
        for user_id, relevant in relevant_by_user.items():
            if not relevant:
                continue
            recommended = recommend(user_id, n)["movieId"].tolist()
            hits = [1 if movie_id in relevant else 0 for movie_id in recommended]
            precision = sum(hits) / n
            recall = sum(hits) / len(relevant)
            dcg = sum(hit / math.log2(rank + 2) for rank, hit in enumerate(hits))
            ideal_hits = min(len(relevant), n)
            idcg = sum(1 / math.log2(rank + 2) for rank in range(ideal_hits))
            rows.append(
                {
                    "Algorithm": algorithm,
                    "userId": int(user_id),
                    "Precision@K": precision,
                    "Recall@K": recall,
                    "NDCG@K": dcg / idcg if idcg else 0.0,
                }
            )
    if not rows:
        raise ValueError(
            "No relevant held-out ratings found; evaluation requires test ratings "
            f"of at least {relevance_threshold:g}"
        )
    per_user = pd.DataFrame(rows)
    metrics = (
        per_user.groupby("Algorithm", as_index=False)
        .agg(
            **{
                "Users evaluated": ("userId", "nunique"),
                "Precision@K": ("Precision@K", "mean"),
                "Recall@K": ("Recall@K", "mean"),
                "NDCG@K": ("NDCG@K", "mean"),
            }
        )
    )
    metrics.insert(0, "K", n)
    return metrics


def main() -> None:
    args = parse_args()
    if args.top_n < 1 or args.neighbors < 1:
        raise ValueError("--top-n and --neighbors must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ratings, movies = load_movielens(args.data_dir)

    train, test = temporal_split(ratings)
    metrics = evaluate_ranking(train, test, movies, n=args.top_n, neighbors=args.neighbors)
    metrics.to_csv(args.output_dir / "evaluation_metrics.csv", index=False)

    recommender = ItemItemRecommender(n_neighbors=args.neighbors).fit(ratings, movies)
    popularity = (
        ratings.groupby("movieId")
        .agg(Rating_Count=("rating", "size"), Average_Rating=("rating", "mean"))
        .join(movies.drop_duplicates("movieId").set_index("movieId")[["title", "genres"]])
        .sort_values(["Rating_Count", "Average_Rating"], ascending=False)
        .head(20)
        .reset_index()
    )
    popularity.to_csv(args.output_dir / "popular_movies.csv", index=False)

    selected_user = args.user_id
    if selected_user is None:
        selected_user = int(ratings["userId"].value_counts().idxmax())
        print(f"No --user-id supplied; demonstrating recommendations for most active user {selected_user}.")
    recommendations = recommender.recommend(selected_user, args.top_n)
    recommendations.to_csv(args.output_dir / "recommendations.csv", index=False)

    print(f"Loaded {len(ratings):,} ratings from {ratings['userId'].nunique():,} users "
          f"across {ratings['movieId'].nunique():,} movies.")
    print(f"Mean rating: {ratings['rating'].mean():.2f}; sparsity: "
          f"{1 - len(ratings) / (ratings['userId'].nunique() * ratings['movieId'].nunique()):.2%}.")
    print("Temporal holdout ranking metrics (relevant rating >= 4):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nRecommendations for user {selected_user}:")
    print(recommendations.to_string(index=False))
    print(f"\nOutputs saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
