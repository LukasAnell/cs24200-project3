from pathlib import Path

from pandas import DataFrame, read_csv

PATH: Path = Path("data/ml-latest-small")


def load_ratings() -> DataFrame:
    return read_csv(PATH / "ratings.csv")


def load_movies() -> DataFrame:
    return read_csv(PATH / "movies.csv")


if __name__ == "__main__":
    ratings: DataFrame = load_ratings()
    movies: DataFrame = load_movies()
    print("ratings:", ratings.shape, list(ratings.columns))
    print("movies:", movies.shape, list(movies.columns))

