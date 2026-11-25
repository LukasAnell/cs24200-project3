from pandas import CategoricalDtype, DataFrame, Index, Series
from sklearn.decomposition import PCA

from process import build_matrix, load_movies, load_ratings

TOP_GENRES: int = 8
OTHER: str = "Other"


def center_items(matrix: DataFrame) -> DataFrame:
    items: DataFrame = matrix.T

    return items - items.mean(axis=0)


def answer_3a(matrix: DataFrame) -> None:
    centered: DataFrame = center_items(matrix)

    print("3(a) transposed, mean-centered matrix (rows = movies, columns = users)")
    print("shape:", centered.shape)
    print("largest absolute column mean:", float(centered.mean(axis=0).abs().max()))


def pca_items(centered: DataFrame, n_components: int = 2) -> tuple[DataFrame, PCA]:
    model: PCA = PCA(n_components=n_components, svd_solver="full")
    scores: DataFrame = DataFrame(
            model.fit_transform(centered.to_numpy()),
            index=centered.index,
            columns=[f"pc{i + 1}" for i in range(n_components)],
            )

    return scores, model


def answer_3b(matrix: DataFrame) -> None:
    scores, model = pca_items(center_items(matrix))

    print("3(b) PCA with 2 components")
    print("scores shape:", scores.shape)
    print("explained variance ratio:", model.explained_variance_ratio_)
    print("total explained by 2 components:", float(model.explained_variance_ratio_.sum()))


def genre_groups(movies: DataFrame, movie_ids: Index) -> Series:
    first: Series = (
            movies.set_index("movieId")["genres"].str.split("|").str[0].loc[movie_ids]
            )

    top: list[str] = list(first.value_counts().head(TOP_GENRES).index)
    grouped: Series = first.where(first.isin(top), OTHER)

    return grouped.astype(CategoricalDtype(top + [OTHER], ordered=True)).rename("genre")


def answer_3c(matrix: DataFrame, movies: DataFrame) -> None:
    scores, model = pca_items(center_items(matrix))
    genres: Series = genre_groups(movies, scores.index)

    print("3(c) first-genre groups of the plotted movies")
    print(genres.value_counts(sort=False).to_string())


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())
    movies: DataFrame = load_movies()

    answer_3a(matrix)

    print()

    answer_3b(matrix)

    print()

    answer_3c(matrix, movies)

