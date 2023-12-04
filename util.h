#include <omp.h>

#include <cassert>
#include <vector>

using std::vector;
template <typename T>
vector<vector<T>> transposeMatrix(const vector<vector<T>>& matrix) {
	int n = matrix.size();
	int m = matrix[0].size();
	vector transposed(m, vector<T>(n));
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			transposed[j][i] = matrix[i][j];
		}
	}
	return transposed;
}

template <typename T>
vector<vector<T>> multiplyMatrices(const vector<vector<T>>& matrix1,
								   const vector<vector<T>>& matrix2) {
	int rows1 = matrix1.size();
	int cols1 = matrix1[0].size();
	int rows2 = matrix2.size();
	int cols2 = matrix2[0].size();

	assert(cols1 == rows2 && "Wrong matrix sizes");

	vector result(rows1, vector<T>(cols2));

#pragma omp parallel for shared(matrix1, matrix2, result)
	for (int i = 0; i < rows1; ++i) {
		for (int j = 0; j < cols2; ++j) {
			for (int k = 0; k < cols1; ++k) {
				result[i][j] += matrix1[i][k] * matrix2[k][j];
			}
		}
	}

	return result;
}

template <typename T>
std::vector<T> multiplyMatrixByVector(const std::vector<std::vector<T>>& matrix,
									  const std::vector<T>& vector) {
	int rows = matrix.size();
	int cols = matrix[0].size();

	std::vector<T> result(rows);

	for (int i = 0; i < rows; ++i) {
		for (int j = 0; j < cols; ++j) {
			result[i] += matrix[i][j] * vector[j];
		}
	}

	return result;
}