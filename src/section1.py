from pandas import DataFrame, Series

from process import build_matrix, load_movies, load_ratings


def top_items(matrix: DataFrame, movies: DataFrame, n: int = 3) -> DataFrame:
    counts: Series = (matrix > 0).sum(axis=0)
    top: Series = counts.nlargest(n, keep="all")
    titles: Series = movies.set_index("movieId")["title"]

    return DataFrame({"title": titles.loc[top.index], "n_users": top})


def answer_1a(matrix: DataFrame, movies: DataFrame) -> None:
    print("1(a) top items by number of users who rated them")
    print(top_items(matrix, movies).to_string())


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())
    movies: DataFrame = load_movies()

    answer_1a(matrix, movies)
