
#include <boost/log/trivial.hpp>
#include <mpi.h>

#include <iostream>
#include <string>
#include <vector>

namespace master_slave_wrapper_sync {
constexpr bool debug_mpi = false;

void send_double_vector(const std::vector<double> &value, int current_rank,
                        int receiver_rank = 0) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Sending double vector from " << current_rank << " process to "
        << receiver_rank << " process" << std::endl;
  }
  int size = value.size();
  MPI_Send(&size, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
  MPI_Send(value.data(), size, MPI_DOUBLE, receiver_rank, 1, MPI_COMM_WORLD);
}

[[nodiscard]] std::vector<double> receive_double_vector(int rank) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Receiving double vector from " << rank << " process" << std::endl;
  }

  MPI_Status status;

  int size{0};
  std::vector<double> value;

  MPI_Recv(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
  value.resize(size);

  MPI_Recv(value.data(), size, MPI_DOUBLE, rank, 1, MPI_COMM_WORLD, &status);

  return value;
}

void send_double(const double value, int current_rank, int receiver_rank = 0) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Sending double from " << current_rank << " process to "
        << receiver_rank << " process" << std::endl;
  }

  MPI_Send(&value, 1, MPI_DOUBLE, receiver_rank, 0, MPI_COMM_WORLD);
}

[[nodiscard]] double receive_double(int rank) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Receiving double from " << rank << " process" << std::endl;
  }

  MPI_Status status;

  double value{0.};

  MPI_Recv(&value, 1, MPI_DOUBLE, rank, 0, MPI_COMM_WORLD, &status);
  return value;
}

void send_int_vector(const std::vector<int> &value, int current_rank,
                     int receiver_rank = 0) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Sending int vector from " << current_rank << " process to "
        << receiver_rank << " process" << std::endl;
  }
  int size = value.size();
  MPI_Send(&size, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
  MPI_Send(value.data(), size, MPI_INT, receiver_rank, 1, MPI_COMM_WORLD);
}

[[nodiscard]] std::vector<int> receive_int_vector(int rank) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Receiving int vector from " << rank << " process" << std::endl;
  }

  MPI_Status status;

  int size{0};
  std::vector<int> value;

  MPI_Recv(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
  value.resize(size);

  MPI_Recv(value.data(), size, MPI_INT, rank, 1, MPI_COMM_WORLD, &status);

  return value;
}

void send_int(const int value, int current_rank, int receiver_rank = 0) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Sending int from " << current_rank << " process to "
        << receiver_rank << " process" << std::endl;
  }

  MPI_Send(&value, 1, MPI_INT, receiver_rank, 0, MPI_COMM_WORLD);
}

[[nodiscard]] int receive_int(int rank) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Receiving int from " << rank << " process" << std::endl;
  }

  MPI_Status status;

  int value{0};

  MPI_Recv(&value, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
  return value;
}

/// @brief Send string from current process to target process
/// @param s String for sending
/// @param current_rank Current process
/// @param receiver_rank Target process
void send_string(const std::string &s, int current_rank,
                 int receiver_rank = 0) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Sending string from " << current_rank << " process to "
        << receiver_rank << " process" << std::endl;
  }

  int size = s.length() + 1;
  MPI_Send(&size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  MPI_Send(s.c_str(), size, MPI_CHAR, 0, 1, MPI_COMM_WORLD);
}

/// @brief Async method for receiving message, which was sended with send_string
/// function
/// @param rank Is number of proccess which sended data
/// @return Received string
[[nodiscard]] std::string receive_string(int rank) {
  if constexpr (debug_mpi) {
    BOOST_LOG_TRIVIAL(info)
        << "Receiving string from " << rank << " process" << std::endl;
  }

  MPI_Status status;

  int size{0};
  std::string data;

  MPI_Recv(&size, 1, MPI_INT, rank, 0, MPI_COMM_WORLD, &status);
  data.resize(size);

  MPI_Recv(data.data(), size, MPI_CHAR, rank, 1, MPI_COMM_WORLD, &status);
  return data;
}
}; // namespace master_slave_wrapper_sync

namespace master_slave_wrapper_async {
constexpr bool debug_mpi = false;

}; // namespace master_slave_wrapper_async