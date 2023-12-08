﻿#include <omp.h>
#include <Windows.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <ostream>
#include <random>
#include <thread>
#include <vector>

using std::cin, std::cout, std::vector;

template <typename T>
std::ostream &operator<<(std::ostream &out, vector<T> &v) {
	for (int i = 0; i < min((int)v.size(), 5); i++) {
		out << v[i] << ' ';
	}
	if (v.size() > 5) {
		out << "...";
	}

	return out;
}

// Решение СЛАУ методом Якоби

double get_diff(vector<double> &a, vector<double> &b) {
	double total = 0;
	for (unsigned int i = 0; i < a.size(); i++) {
		total += pow(a[i] - b[i], 2);
	}
	return sqrt(total);
}

vector<double> jacoby_serial(vector<vector<double>> &a, vector<double> &b, double tolerance) {
	int n = a.size();
	vector<double> x_k(n), x_k1(n);
	do {
		for (int i = 0; i < n; i++) {
			double sum = b[i];
			for (int j = 0; j < n; j++) {
				if (i != j) {
					sum -= x_k[j] * a[i][j];
				}
			}
			x_k1[i] = sum / a[i][i];
		}
		swap(x_k, x_k1);
	} while (get_diff(x_k, x_k1) > tolerance);
	return x_k1;
}

vector<double> jacoby_omp(vector<vector<double>> &a, vector<double> &b, double tolerance) {
	int n = a.size();
	vector<double> x_k(n), x_k1(n);
	do {
#pragma omp parallel for
		for (int i = 0; i < n; i++) {
			double sum = b[i];
			// #pragma omp parallel for reduction(+ : sum)
			for (int j = 0; j < n; j++) {
				if (i != j) {
					sum -= x_k[j] * a[i][j];
				}
			}
			x_k1[i] = sum / a[i][i];
		}
		swap(x_k, x_k1);
	} while (get_diff(x_k, x_k1) > tolerance);
	return x_k1;
}


vector<double> jacoby_winapi(vector<vector<double>> &a, vector<double> &b, double tolerance) {
	int n = a.size();
	vector<double> x_k(n), x_k1(n);
	HANDLE *threads = new HANDLE[n];

	// Да! Действительно можно записать tuple вместо struct!!!
	typedef std::tuple<int, vector<vector<double>> *, vector<double> *, vector<double> *, double, double *> winapi_parameter;
	// Здесь решил не выносить функцию потока, а записать её как лямбда-функцию
	auto lambda_fun = [](LPVOID param) -> DWORD {
		auto data = reinterpret_cast<winapi_parameter *>(param);

		// C++ 20 on it's best
		auto &[idx, a_ref, b_ref, x_k_ref, tol, res] = *data;

		double sum = b_ref->at(idx);
		for (size_t j = 0; j < a_ref->size(); j++) {
			if (idx != j) {
				sum -= x_k_ref->at(j) * a_ref->at(idx)[j];
			}
		}
		*res = sum / a_ref->at(idx)[idx];

		return 0;
		};

	do {
		for (int i = 0; i < n; i++) {
			threads[i] = CreateThread(NULL, 0, lambda_fun, new winapi_parameter(i, &a, &b, &x_k, tolerance, &x_k1[i]), 0, NULL);
		}

		WaitForMultipleObjects(n, threads, TRUE, INFINITE);
		swap(x_k, x_k1);

	} while (get_diff(x_k, x_k1) > tolerance);

	for (int i = 0; i < n; i++) {
		if (threads[i]) {
			CloseHandle(threads[i]);
		}
	}
	delete[] threads;

	return x_k1;
}

int main() {
	setlocale(LC_ALL, "Russian");
	int n = 1e4;
	double tolerance = 1e-4;

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dis(-100, 100);

	vector a(n, vector<double>(n));
	vector b(n, 0.);
	for (int i = 0; i < n; i++) {
		b[i] = dis(gen);
		for (int j = 0; j < n; j++) {
			a[i][j] = dis(gen);
			while (i == j && abs(a[i][j]) < 10) {
				a[i][j] = dis(gen);
			}
			if (i == j) {
				a[i][j] *= n;
			}
		}
	}

	/*vector<vector<double>> a{
	   {4, -1, 0},
	   {-1, 4, -1},
	   {0, -1, 3},
	};
	vector<double> b{5, -7, 6};*/

	auto start_time = std::chrono::high_resolution_clock::now();
	auto serial_result = jacoby_serial(a, b, tolerance);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto serial_duration = end_time - start_time;

	start_time = std::chrono::high_resolution_clock::now();
	auto omp_result = jacoby_omp(a, b, tolerance);
	end_time = std::chrono::high_resolution_clock::now();
	auto omp_duration = end_time - start_time;

	start_time = std::chrono::high_resolution_clock::now();
	auto winapi_result = jacoby_winapi(a, b, tolerance);
	end_time = std::chrono::high_resolution_clock::now();
	auto winapi_duration = end_time - start_time;

	cout << "Последовательное вычисление:\nРезультат: " << serial_result
		<< "\nВремя: " << serial_duration.count() / 1e6 << " мс\n";
	cout << std::endl;
	cout << "Параллельное (openmp) вычисление:\nРезультат: " << omp_result
		<< "\nВремя: " << omp_duration.count() / 1e6 << " мс\n";
	cout << std::endl;
	cout << "Параллельное (winapi) вычисление:\nРезультат: " << winapi_result
		<< "\nВремя: " << winapi_duration.count() / 1e6 << " мс\n";
	cout << std::endl;
	cout << "Выигрыш (openmp) в " << (double)serial_duration.count() / omp_duration.count() << " раз" << std::endl;
	cout << "Выигрыш (winapi) в " << (double)serial_duration.count() / winapi_duration.count() << " раз" << std::endl;
	// std::system("pause");
}