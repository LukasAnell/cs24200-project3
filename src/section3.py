from numpy import cumsum, ndarray, searchsorted
from pandas import CategoricalDtype, DataFrame, Index, Series
from plotnine import (
        aes,
        geom_point,
        ggplot,
        guide_legend,
        guides,
        labs,
        scale_color_brewer,
        theme,
        theme_minimal,
        )
from sklearn.decomposition import PCA

from process import build_matrix, load_movies, load_ratings

TOP_GENRES: int = 8
OTHER: str = "Other"
VARIANCE_TARGETS: list[float] = [0.80, 0.40]


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


def plot_3c(scores: DataFrame, genres: Series, ratios: ndarray) -> None:
    data: DataFrame = scores.join(genres)
    plot: ggplot = (
            ggplot(data, aes(x="pc1", y="pc2", color="genre"))
            + geom_point(alpha=0.4, size=1)
            + scale_color_brewer(type="qual", palette="Set1")
            + guides(color=guide_legend(override_aes={"alpha": 1, "size": 3}))
            + labs(
                title="Movies projected onto the first two principal components",
                x=f"PC1 ({ratios[0]:.1%} of variance)",
                y=f"PC2 ({ratios[1]:.1%} of variance)",
                color="First genre",
                )
            + theme_minimal()
            + theme(figure_size=(10, 6))
            )
    plot.save("figures/pca_movies_by_genre.png", verbose=False)


def answer_3c(matrix: DataFrame, movies: DataFrame) -> None:
    scores, model = pca_items(center_items(matrix))
    genres: Series = genre_groups(movies, scores.index)

    print("3(c) first-genre groups of the plotted movies")
    print(genres.value_counts(sort=False).to_string())

    plot_3c(scores, genres, model.explained_variance_ratio_)


def intrinsic_dim(centered: DataFrame) -> ndarray:
    model: PCA = PCA(n_components=None, svd_solver="full").fit(centered.to_numpy())

    return cumsum(model.explained_variance_ratio_)


def answer_3d(matrix: DataFrame) -> None:
    cumulative: ndarray = intrinsic_dim(center_items(matrix))

    print("3(d) components needed to reach a share of the variance")
    print("components fitted:", len(cumulative))
    for target in VARIANCE_TARGETS:
        count: int = int(searchsorted(cumulative, target) + 1)
        print(f"{target:.0%}: {count} components (cumulative {cumulative[count - 1]:.4f})")
    print("2 components cumulative:", float(cumulative[1]))


def check_pc1_popularity(matrix: DataFrame) -> None:
    scores, _ = pca_items(center_items(matrix))
    n_raters: Series = (matrix > 0).sum(axis=0).loc[scores.index]

    print("rank correlation of each component with number of raters per movie")
    for column in scores.columns:
        print(column, float(scores[column].rank().corr(n_raters.rank())))


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())
    movies: DataFrame = load_movies()

    answer_3a(matrix)

    print()

    answer_3b(matrix)

    print()

    answer_3c(matrix, movies)

    print()

    check_pc1_popularity(matrix)

    print()

    answer_3d(matrix)

