#include <omp.h>

#include <boost/log/trivial.hpp>

#include <cmath>
#include <vector>

using namespace std;

int main() {
  BOOST_LOG_TRIVIAL(info) << "Starting execute main process" << std::endl;

  size_t n = 1000;
  double h = 1.0 / (n - 1);
  double tolerance = 1e-12;
  size_t maxIterations = 1000000;

  vector<vector<double>> u(n, vector<double>(n));
  vector<vector<double>> u_new(n, vector<double>(n));

  for (int i = 0; i < n; ++i) {
    u[i][0] = 0.0;
    u[i][n - 1] = 0.0;
    u[0][i] = 0.0;
    u[n - 1][i] = 0.0;
  }

  auto f = [](double x1, double x2) { return std::pow(x1 + x2, 2); };

  double diff = std::numeric_limits<double>::max();
  size_t iteration = 0;

  while (diff > tolerance && iteration < maxIterations) {
#pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; ++i) {
      for (int j = 1; j < n - 1; ++j) {
        u_new[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] +
                              u[i][j + 1] - h * h * f(i * h, j * h));
        diff = abs(u_new[i][j] - u[i][j]);
      }
    }

#pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; ++i) {
      for (int j = 1; j < n - 1; ++j) {
        u[i][j] = u_new[i][j];
      }
    }
    iteration++;
  }

  BOOST_LOG_TRIVIAL(info) << "Iteration: " << iteration << endl;
  BOOST_LOG_TRIVIAL(info) << "Error: " << diff << endl;

  return 0;
}