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
from section2 import RANDOM_STATE

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


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    answer_4a(matrix)
