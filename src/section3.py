from pandas import DataFrame
from sklearn.decomposition import PCA

from process import build_matrix, load_ratings


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


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    answer_3a(matrix)

    print()

    answer_3b(matrix)

