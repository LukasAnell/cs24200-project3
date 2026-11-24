from pathlib import Path

from pandas import DataFrame, read_csv

PATH: Path = Path("data/ml-latest-small")
EXPECTED_SHAPE: tuple[int, int] = (610, 9724)


def load_ratings() -> DataFrame:
    return read_csv(PATH / "ratings.csv")


def load_movies() -> DataFrame:
    return read_csv(PATH / "movies.csv")


def check_ratings(ratings: DataFrame) -> None:
    duplicate_pairs: int = int(ratings.duplicated(subset=["userId", "movieId"]).sum())
    zero_ratings: int = int((ratings["rating"] == 0).sum())

    print("duplicate (userId, movieId) pairs:", duplicate_pairs)
    print("ratings equal to 0:", zero_ratings)
    print("rating range:", ratings["rating"].min(), "to", ratings["rating"].max())


def build_matrix(ratings: DataFrame) -> DataFrame:
    matrix: DataFrame = ratings.pivot(
            index="userId", columns="movieId", values="rating"
            ).fillna(0)

    if matrix.shape != EXPECTED_SHAPE:
        raise ValueError(f"matrix shape {matrix.shape}, expected {EXPECTED_SHAPE}")

    return matrix


if __name__ == "__main__":
    ratings: DataFrame = load_ratings()
    movies: DataFrame = load_movies()

    print("ratings:", ratings.shape, list(ratings.columns))
    print("movies:", movies.shape, list(movies.columns))
    check_ratings(ratings)

    matrix: DataFrame = build_matrix(ratings)
    print("matrix:", matrix.shape)

