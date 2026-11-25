from pandas import DataFrame
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

from process import build_matrix, load_ratings

RANDOM_STATE: int = 1
N_INIT: int = 10
K_VALUES: list[int] = [2, 4, 8, 16, 32, 64, 128]


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


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    sweep: DataFrame = inertia_sweep(matrix)
    print(sweep.to_string(index=False))

    plot_2a(sweep)
