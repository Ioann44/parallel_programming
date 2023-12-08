#include <omp.h>
#include <Windows.h>

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

double integrate_parallel_omp(double begin, double end, double h, double (*func)(double)) {
	double total = 0;
	int n = (end - begin) / h;
#pragma omp parallel for reduction(+ : total)
	for (int i = 0; i < n; i++) {
		total += func(begin + i * h + h / 2);
	}
	return total * h;
}

// WinApi версия посложнее, потому снабжена комментариями
// Под параметры функции потока выделен единственный указатель,
// так что упаковка в struct (общепринятый метод), хотя в теории думаю можно вывернуться с tuple
struct ThreadParams {
	double begin;
	double end;
	double h;
	double (*func)(double);
	double result;
};
// Собственно именно эта функция и будет выполняться в потоке
DWORD WINAPI ThreadFunction(LPVOID params) {
	ThreadParams *in = static_cast<ThreadParams *>(params);
	double total = 0;
	int n = (in->end - in->begin) / in->h;
	for (int i = 0; i < n; i++) {
		total += in->func(in->begin + i * in->h + in->h / 2);
	}
	in->result = total * in->h;
	return 0;
}
double integrate_parallel_winapi(double begin, double end, double h, double (*func)(double)) {
	const int numThreads = 8;
	HANDLE threads[numThreads];
	ThreadParams params[numThreads];

	double total = 0;
	int n = (end - begin) / h;
	// Проще округлить размер чанка вверх
	int chunkSize = (n + numThreads - 1) / numThreads;
	double temp_begin = begin;

	for (int i = 0; i < numThreads; ++i) {
		params[i].begin = temp_begin;
		params[i].end = min(temp_begin + chunkSize * h, end); // Чанк округлён вверх, а выходить за пределы не нужно
		params[i].h = h;
		params[i].func = func;

		threads[i] = CreateThread(NULL, 0, ThreadFunction, &params[i], 0, NULL);

		temp_begin = params[i].end;
	}

	// Ждём завершения всех потоков
	WaitForMultipleObjects(numThreads, threads, TRUE, INFINITE);

	for (int i = 0; i < numThreads; ++i) {
		total += params[i].result;
		// Высвобождаются ресурсы
		if (threads[i]) {
			CloseHandle(threads[i]);
		}
	}

	return total;
}

int main() {
	setlocale(LC_ALL, "Russian");
	double begin_x = 0, end_x = 1000, h = 1e-5;

	auto start_time = std::chrono::high_resolution_clock::now();
	double serial_result = integrate_serial(begin_x, end_x, h, func_to_integrate);
	auto end_time = std::chrono::high_resolution_clock::now();
	auto serial_duration = end_time - start_time;

	start_time = std::chrono::high_resolution_clock::now();
	double omp_result = integrate_parallel_omp(begin_x, end_x, h, func_to_integrate);
	end_time = std::chrono::high_resolution_clock::now();
	auto omp_duration = end_time - start_time;

	start_time = std::chrono::high_resolution_clock::now();
	double winapi_result = integrate_parallel_winapi(begin_x, end_x, h, func_to_integrate);
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
	cout << "Выигрыш openmp в " << (double)serial_duration.count() / omp_duration.count() << " раз"
		<< std::endl;
	cout << "Выигрыш winapi в " << (double)serial_duration.count() / winapi_duration.count() << " раз"
		<< std::endl;
	// std::system("pause");
}
