import unittest

import pandas as pd

from recommender import ItemItemRecommender, evaluate_ranking, temporal_split


def sample_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["One", "Two", "Three", "Four", "Five"],
            "genres": ["Drama"] * 5,
        }
    )
    ratings = pd.DataFrame(
        [
            (1, 1, 5.0, 1), (1, 2, 4.5, 2), (1, 3, 1.0, 3), (1, 4, 5.0, 4),
            (2, 1, 5.0, 1), (2, 2, 4.0, 2), (2, 3, 1.0, 3), (2, 5, 4.5, 4),
            (3, 1, 4.5, 1), (3, 2, 4.0, 2), (3, 3, 1.5, 3), (3, 4, 5.0, 4),
        ],
        columns=["userId", "movieId", "rating", "timestamp"],
    )
    return ratings, movies


class RecommenderTests(unittest.TestCase):
    def test_recommendations_exclude_seen_movies_and_include_titles(self) -> None:
        ratings, movies = sample_data()
        recommender = ItemItemRecommender(n_neighbors=4).fit(ratings, movies)
        recommendations = recommender.recommend(1, n=3)
        self.assertEqual(len(recommendations), 1)
        self.assertNotIn(4, recommendations["movieId"].tolist())
        self.assertEqual(recommendations.iloc[0]["title"], "Five")

    def test_unknown_user_gets_popularity_recommendations(self) -> None:
        ratings, movies = sample_data()
        recommender = ItemItemRecommender(n_neighbors=4).fit(ratings, movies)
        recommendations = recommender.recommend(999, n=3)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(recommendations["score"].isna().all())

    def test_popularity_baseline_excludes_seen_movies(self) -> None:
        ratings, movies = sample_data()
        recommender = ItemItemRecommender(n_neighbors=4).fit(ratings, movies)
        recommendations = recommender.recommend_popular(1, n=3)
        self.assertTrue(set(recommendations["movieId"]).isdisjoint({1, 2, 3, 4}))

    def test_temporal_split_holds_out_latest_ratings_per_user(self) -> None:
        ratings, _ = sample_data()
        train, test = temporal_split(ratings)
        self.assertTrue((test.groupby("userId")["timestamp"].min() > train.groupby("userId")["timestamp"].max()).all())
        self.assertEqual(len(train) + len(test), len(ratings))

    def test_evaluation_returns_ranking_metrics(self) -> None:
        ratings, movies = sample_data()
        train, test = temporal_split(ratings, test_fraction=0.25)
        metrics = evaluate_ranking(train, test, movies, n=2, neighbors=4)
        self.assertEqual(metrics["K"].tolist(), [2, 2])
        self.assertEqual(set(metrics["Algorithm"]), {"Item-item", "Popularity baseline"})
        for metric in ("Precision@K", "Recall@K", "NDCG@K"):
            self.assertTrue(metrics[metric].between(0, 1).all())

    def test_invalid_recommendation_count_is_rejected(self) -> None:
        ratings, movies = sample_data()
        recommender = ItemItemRecommender(n_neighbors=4).fit(ratings, movies)
        with self.assertRaises(ValueError):
            recommender.recommend(1, n=0)


if __name__ == "__main__":
    unittest.main()
