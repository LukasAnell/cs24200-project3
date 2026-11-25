from numpy import ndarray
from pandas import DataFrame
from plotnine import (
        aes,
        geom_line,
        geom_point,
        ggplot,
        labs,
        scale_y_log10,
        theme,
        theme_minimal,
        )
from sklearn.decomposition import TruncatedSVD

from process import build_matrix, load_ratings
from section2 import K_VALUES, RANDOM_STATE, inertia_sweep

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


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    answer_4a(matrix)

    print()

    answer_4b(matrix)

    print()

    answer_4c(matrix)
