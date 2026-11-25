from pandas import DataFrame, Series, concat
from plotnine import (
        aes,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_x_continuous,
        theme,
        theme_minimal,
        )
from sklearn.cluster import KMeans

from process import build_matrix, load_movies, load_ratings

RANDOM_STATE: int = 1
N_INIT: int = 10
K_VALUES: list[int] = [2, 4, 8, 16, 32, 64, 128]
K_EXPLORE: int = 8
MIN_RATER_PERCENT: int = 10


def inertia_sweep(matrix: DataFrame) -> DataFrame:
    inertias: list[float] = []

    for k in K_VALUES:
        model: KMeans = KMeans(
                n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE
                ).fit(matrix.to_numpy())

        inertias.append(float(model.inertia_))

    return DataFrame({"k": K_VALUES, "inertia": inertias})


def plot_2a(sweep: DataFrame) -> None:
    plot: ggplot = (
            ggplot(sweep, aes(x="k", y="inertia"))
            + geom_line()
            + geom_point()
            + scale_x_continuous(
                trans="log2", breaks=K_VALUES, labels=[str(k) for k in K_VALUES]
                )
            + labs(
                title="k-means inertia by number of clusters",
                x="Number of clusters k (log scale)",
                y="Inertia",
                )
            + theme_minimal()
            + theme(figure_size=(8, 4.5))
            )
    plot.save("figures/inertia_vs_k.png", verbose=False)


def cluster_labels(matrix: DataFrame, k: int) -> Series:
    model: KMeans = KMeans(
            n_clusters=k, n_init=N_INIT, random_state=RANDOM_STATE
            ).fit(matrix.to_numpy())

    return Series(model.labels_, index=matrix.index, name="cluster")


def cluster_top_items(
        matrix: DataFrame, movies: DataFrame, labels: Series, n: int = 3
        ) -> DataFrame:
    titles: Series = movies.set_index("movieId")["title"]
    tables: list[DataFrame] = []

    for cluster in sorted(labels.unique()):
        block: DataFrame = matrix.loc[labels == cluster]
        counts: Series = (block > 0).sum(axis=0)
        means: Series = block.where(block > 0).mean(axis=0)

        candidates: DataFrame = DataFrame({"n_raters": counts, "mean_rating": means})
        enough: Series = counts * 100 >= MIN_RATER_PERCENT * len(block)
        top: DataFrame = (
                candidates[enough]
                .reset_index()
                .sort_values(
                    ["mean_rating", "n_raters", "movieId"],
                    ascending=[False, False, True],
                    )
                .head(n)
                )

        top.insert(0, "cluster_size", len(block))
        top.insert(0, "cluster", cluster)

        top["title"] = top["movieId"].map(titles)
        
        tables.append(top)

    return concat(tables, ignore_index=True)


def cluster_summary(matrix: DataFrame, labels: Series) -> DataFrame:
    n_ratings: Series = (matrix > 0).sum(axis=1)

    return n_ratings.groupby(labels).agg(["size", "min", "median", "max"])


def answer_2c(matrix: DataFrame, movies: DataFrame, k: int) -> None:
    labels: Series = cluster_labels(matrix, k)

    print(f"cluster sizes and ratings per user at k={k}")
    print(cluster_summary(matrix, labels).to_string())

    print(f"2(c) top items per cluster at k={k}")
    print(cluster_top_items(matrix, movies, labels).round(2).to_string(index=False))


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())
    movies: DataFrame = load_movies()

    sweep: DataFrame = inertia_sweep(matrix)
    print(sweep.to_string(index=False))

    plot_2a(sweep)

    answer_2c(matrix, movies, K_EXPLORE)
