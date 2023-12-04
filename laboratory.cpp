#include <omp.h>

#include <chrono>
#include <cmath>
#include <iostream>

using std::cin, std::cout;

double func_to_integrate(double x) {
	// return 1;
	return std::sin(x) + x;
}

double integrate_serial(double begin, double end, double h, double (*func)(double)) {
	double total = 0;
	int n = (end - begin) / h;
	for (int i = 0; i < n; i++) {
		total += func(begin + i * h + h / 2);
	}
	return total * h;
}

double integrate_parallel(double begin, double end, double h, double (*func)(double)) {
	double total = 0;
	int n = (end - begin) / h;
#pragma omp parallel for reduction(+ : total)
	for (int i = 0; i < n; i++) {
		total += func(begin + i * h + h / 2);
	}
	return total * h;
}

int main() {
	double begin_x = 0, end_x = 1000, h = 1e-5;

	auto start_time = std::chrono::high_resolution_clock::now();
	double serial_result = integrate_serial(begin_x, end_x, h, func_to_integrate);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto serial_duration = end_time - start_time;

	start_time = std::chrono::high_resolution_clock::now();
	double parallel_result = integrate_parallel(begin_x, end_x, h, func_to_integrate);
	end_time = std::chrono::high_resolution_clock::now();
	auto parallel_duration = end_time - start_time;

	cout << "Последовательное вычисление:\nРезультат: " << serial_result
		 << "\nВремя: " << serial_duration.count() / 1e6 << " мс\n";
	cout << std::endl;
	cout << "Параллельное вычисление:\nРезультат: " << parallel_result
		 << "\nВремя: " << parallel_duration.count() / 1e6 << " мс\n";
	cout << std::endl;
	cout << "Выигрыш в " << (double)serial_duration.count() / parallel_duration.count() << " раз"
		 << std::endl;
	// std::system("pause");
}
