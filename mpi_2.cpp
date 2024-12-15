#include "master_slave_wrapper.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <mutex>
#include <numeric>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

double func(double x) { return std::pow(x, 3) + std::cos(x); }

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    std::mutex mutex_batch, mutex_result;

    double result_integrate{0.};

    constexpr size_t points_count{10000}, batch_size{1002},
        batch_count{static_cast<size_t>(
            std::ceil(static_cast<double>(points_count) / batch_size))};
    constexpr double left_boundary{0.}, right_boundary{1.},
        step{(right_boundary - left_boundary) / (points_count - 1)};

    int batch_i{0};

    assert(batch_count > size && "Check your constants!");
    assert(batch_size > 2 && "Check your constants!");
    static_assert(points_count % batch_size > 2 && "Check your constants!");

    if (rank == 0) {
      BOOST_LOG_TRIVIAL(info) << "Starting execute main process" << std::endl;

      auto listener = [&mutex_batch, &mutex_result, &batch_i, rank,
                       &result_integrate](size_t proccess_i) {
        while (true) {
          mutex_batch.lock();
          if (batch_i >= batch_count) {
            master_slave_wrapper_sync::send_int(-1 /*break process signal*/,
                                                rank, proccess_i);
            mutex_batch.unlock();
            break;
          }
          master_slave_wrapper_sync::send_int(batch_i, rank, proccess_i);
          // Im doing mistake, change batch_i in center of block bad practice
          batch_i++;
          mutex_batch.unlock();

          double result = master_slave_wrapper_sync::receive_double(proccess_i);

          mutex_result.lock();
          result_integrate += result;
          mutex_result.unlock();
        }
      };

      // Start listener threads
      std::vector<std::thread> listener_threads;
      listener_threads.reserve(size - 1);
      std::ranges::for_each(
          std::views::iota(1, size), [&listener, &listener_threads](int i) {
            listener_threads.push_back(std::move(std::thread(listener, i)));
          });
      std::ranges::for_each(listener_threads, [](std::thread &t) { t.join(); });

      BOOST_LOG_TRIVIAL(info)
          << "Result integrate: " << std::format("{}", result_integrate)
          << std::endl;

      BOOST_LOG_TRIVIAL(info) << "End execute main process" << std::endl;

    } else {
      // The good practice was created daemon thread, which was listen a
      // specific data escape, which should signalize abound exit process, but
      // im lazy
      batch_i = master_slave_wrapper_sync::receive_int(0);

      int batch_stop = 0;
      auto generate_batch = [&batch_i, &batch_stop](double x) {
        if (left_boundary + (batch_i * batch_size + x) * step >
                right_boundary &&
            batch_stop == 0) {
          batch_stop = x + 1;
          return right_boundary;
        }
        return left_boundary + (batch_i * batch_size + x) * step;
      };
      while (batch_i != -1) {
        std::vector<double> batch(batch_size + 1);
        // At first fill batch with batch index, after transform it to x
        // values
        std::iota(batch.begin(), batch.end(), 0);
        std::ranges::transform(batch.begin(), batch.end(), batch.begin(),
                               generate_batch);
        if (batch_stop != 0) {
          batch.resize(batch_stop);
        }
        std::vector<double> values(batch.size() - 1);

        for (size_t i = 0; i < values.size(); i++) {
          values[i] = (batch[i + 1] - batch[i]) *
                      (func(batch[i]) + func(batch[i + 1])) / 2.;
        }

        double result = std::reduce(values.begin(), values.end());
        master_slave_wrapper_sync::send_double(result, rank);
        batch_i = master_slave_wrapper_sync::receive_int(0);
        batch_stop = 0;
      }
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}