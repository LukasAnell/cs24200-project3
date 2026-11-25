from numpy import ndarray
from pandas import DataFrame, Series
from plotnine import (
        aes,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_color_brewer,
        scale_y_log10,
        theme,
        theme_minimal,
        )
from sklearn.decomposition import TruncatedSVD

from process import build_matrix, load_ratings
from section2 import CHOSEN_K, K_VALUES, RANDOM_STATE, cluster_labels, inertia_sweep

SVD_K: int = 128


def svd_fit(matrix: DataFrame, k: int) -> TruncatedSVD:
    model: TruncatedSVD = TruncatedSVD(
            n_components=k, algorithm="arpack", random_state=RANDOM_STATE
            )

    return model.fit(matrix.to_numpy())


def plot_4a(model: TruncatedSVD) -> None:
    values: DataFrame = DataFrame(
            {
                "component": range(1, len(model.singular_values_) + 1),
                "singular_value": model.singular_values_,
                }
            )
    plot: ggplot = (
            ggplot(values, aes(x="component", y="singular_value"))
            + geom_line()
            + geom_point(size=1)
            + scale_y_log10()
            + labs(
                title="Singular values of the user-movie ratings matrix",
                x="Component",
                y="Singular value (log scale)",
                )
            + theme_minimal()
            + theme(figure_size=(8, 4.5))
            )
    plot.save("figures/svd_singular_values.png", verbose=False)


def answer_4a(matrix: DataFrame) -> None:
    model: TruncatedSVD = svd_fit(matrix, SVD_K)

    print(f"4(a) singular values at k={SVD_K}")
    print("count:", len(model.singular_values_))
    print("first five:", model.singular_values_[:5])
    print("last:", float(model.singular_values_[-1]))

    plot_4a(model)


def variance_table(matrix: DataFrame) -> DataFrame:
    shares: list[float] = [
            float(svd_fit(matrix, k).explained_variance_ratio_.sum()) for k in K_VALUES
            ]

    return DataFrame({"k": K_VALUES, "svd_variance": shares})


def answer_4b(matrix: DataFrame) -> None:
    table: DataFrame = variance_table(matrix).merge(inertia_sweep(matrix), on="k")
    values: ndarray = matrix.to_numpy()
    total_ss: float = float(((values - values.mean(axis=0)) ** 2).sum())
    table["kmeans_share"] = 1 - table["inertia"] / total_ss

    print("4(b) explained variance (SVD) next to k-means inertia, by k")
    print(
            table.to_string(
                index=False,
                formatters={
                    "svd_variance": "{:.4f}".format,
                    "inertia": "{:.0f}".format,
                    "kmeans_share": "{:.4f}".format,
                    },
                )
            )
    print("total sum of squares about the column means:", total_ss)


def svd_users(matrix: DataFrame) -> tuple[DataFrame, TruncatedSVD]:
    model: TruncatedSVD = svd_fit(matrix, 2)
    embedding: DataFrame = DataFrame(
            model.transform(matrix.to_numpy()),
            index=matrix.index,
            columns=["svd1", "svd2"],
            )

    return embedding, model


def answer_4c(matrix: DataFrame) -> None:
    embedding, model = svd_users(matrix)

    print("4(c) SVD with 2 components, users transformed")
    print("embedding shape:", embedding.shape)
    print("explained variance ratio sum:", float(model.explained_variance_ratio_.sum()))


def plot_4d(embedding: DataFrame, labels: Series, ratios: ndarray) -> None:
    data: DataFrame = embedding.join(labels.astype(str))
    plot: ggplot = (
            ggplot(data, aes(x="svd1", y="svd2", color="cluster"))
            + geom_point(alpha=0.7, size=2)
            + scale_color_brewer(type="qual", palette="Set1")
            + labs(
                title=f"Users projected onto the first two SVD components (k-means, k={CHOSEN_K})",
                x=f"SVD component 1 ({ratios[0]:.1%} of variance)",
                y=f"SVD component 2 ({ratios[1]:.1%} of variance)",
                color="Cluster",
                )
            + theme_minimal()
            + theme(figure_size=(8, 6))
            )
    plot.save("figures/svd_users_by_cluster.png", verbose=False)


def answer_4d(matrix: DataFrame) -> None:
    embedding, model = svd_users(matrix)
    labels: Series = cluster_labels(matrix, CHOSEN_K)

    print(f"4(d) users by k-means cluster (k={CHOSEN_K}), cluster sizes:")
    print(labels.value_counts().sort_index().to_string())

    plot_4d(embedding, labels, model.explained_variance_ratio_)


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    answer_4a(matrix)

    print()

    answer_4b(matrix)

    print()

    answer_4c(matrix)

    print()

    answer_4d(matrix)
