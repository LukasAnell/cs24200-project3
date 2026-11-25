from pandas import DataFrame

from process import build_matrix, load_ratings


def center_items(matrix: DataFrame) -> DataFrame:
    items: DataFrame = matrix.T

    return items - items.mean(axis=0)


def answer_3a(matrix: DataFrame) -> None:
    centered: DataFrame = center_items(matrix)

    print("3(a) transposed, mean-centered matrix (rows = movies, columns = users)")
    print("shape:", centered.shape)
    print("largest absolute column mean:", float(centered.mean(axis=0).abs().max()))


if __name__ == "__main__":
    matrix: DataFrame = build_matrix(load_ratings())

    answer_3a(matrix)

