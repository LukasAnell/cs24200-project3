from pandas import DataFrame
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


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    print(inertia_sweep(matrix).to_string(index=False))
