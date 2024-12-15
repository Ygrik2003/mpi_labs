#include "master_slave_wrapper.h"

#include <algorithm>
#include <chrono>
#include <ranges>
#include <stdexcept>
#include <thread>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  try {
    if (rank == 0) {
      BOOST_LOG_TRIVIAL(info) << "Starting execute main process" << std::endl;

      std::mutex mutex_logging;
      std::vector<std::string> collected_data(size);
      collected_data[0] = "Hello from process 0";

      auto listener = [&mutex_logging, &collected_data](size_t i) {
        std::string delay_info = master_slave_wrapper_sync::receive_string(i);
        // Log delay to receive next data
        mutex_logging.lock();
        BOOST_LOG_TRIVIAL(info)
            << "Data from process " << i << ": " << delay_info << std::endl;
        mutex_logging.unlock();

        // Receive data
        collected_data[i] = master_slave_wrapper_sync::receive_string(i);
        mutex_logging.lock();
        BOOST_LOG_TRIVIAL(info) << "Data from process " << i << ": "
                                << collected_data[i] << std::endl;
        mutex_logging.unlock();
      };
      std::vector<std::thread> listener_threads;
      listener_threads.reserve(size - 1);
      std::ranges::for_each(
          std::views::iota(1, size), [&listener, &listener_threads](int i) {
            listener_threads.push_back(std::move(std::thread(listener, i)));
          });
      std::ranges::for_each(listener_threads, [](std::thread &t) { t.join(); });

    } else {
      srand(rank);
      const size_t time_calculation = rand() % 5000;

      // Emit delay
      master_slave_wrapper_sync::send_string(
          "Wait " + std::to_string(time_calculation) + " milliseconds", rank);
      std::this_thread::sleep_for(std::chrono::milliseconds(time_calculation));

      master_slave_wrapper_sync::send_string(
          "Hello from process " + std::to_string(rank), rank);
    }
  } catch (const std::exception &e) {
    std::cerr << "Error: " << e.what() << std::endl;
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Finalize();
  return 0;
}