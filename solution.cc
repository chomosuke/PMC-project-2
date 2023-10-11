#include "mpi.h"
#include <random>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

// It would be cleaner to put seed and Gaussian_point into this class,
// but this allows them to be called like regular C functions.
// Feel free to make the whole code more C++-like.
class unit_normal {
    mt19937 gen;
    normal_distribution<double> d{0, 1};

  public:
    void seed(long int s) { gen.seed(s); }
    double sample() { return d(gen); }
};

class disc_dist {
    mt19937 gen;
    discrete_distribution<> d;

  public:
    disc_dist(vector<double> probs) : d(probs.begin(), probs.end()) {}
    void seed(long int s) { gen.seed(s); }
    int sample() { return d(gen); }
};

class normal_dist {
    vector<double> mean;
    double stddev;

  public:
    normal_dist(vector<double> mean_, double stddev_) {
        mean = mean_;
        stddev = stddev_;
    }

    vector<double> generate(unit_normal* un) {
        vector<double> point;
        for (int i = 0; i < mean.size(); i++) {
            point.push_back(un->sample() * stddev + mean[i]);
        }
        return point;
    }
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s [input_file]\n", argv[0]);
        exit(1);
    }

    // Read number of points and dimension
    FILE* fp = fopen(argv[1], "r");
    int N, D; // number of points and dimensions
    fscanf(fp, "%d%d", &N, &D);

    // It would be cleaner to put this all in a class,
    // but this keeps the skeleton C-friendly.
    int c; // number of GMM components
    fscanf(fp, "%d", &c);

    // Init all GMM distributions
    vector<double> probs;
    vector<normal_dist> dists;
    for (int i = 0; i < c; i++) {
        vector<double> mean;
        for (int d = 0; d < D; d++) {
            double m;
            fscanf(fp, "%lf", &m);
            mean.push_back(m);
        }
        double prob;
        double stddev;
        fscanf(fp, "%lg%lg", &stddev, &prob);
        probs.push_back(prob);
        dists.push_back(normal_dist(mean, stddev));
    }

    MPI_Init(&argc, &argv);
    int rank, k;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &k);

    unit_normal un;
    un.seed(42 + rank);

    disc_dist dd(probs);
    dd.seed(42 + rank);

    srand(42 + rank);

    // Spread among ranks in MPI communicator
    int local_N = N / k;
    if (rank < N % k) {
        local_N++;
    }
    cout << "Generating " << local_N << " points for rank " << rank << " of "
         << k << "." << endl;
    vector<vector<double>> points;
    // TODO maybe parallelize this
    for (int i = 0; i < local_N; i++) {
        points.push_back(dists[dd.sample()].generate(&un));
    }

    // now we init the center
    // find distances in parallel
    // one process choose center and broadcast
    vector<vector<double>> centers;
    // first point can be choosen from the points generated locally
    centers.push_back(points[rand() % points.size()]);
    // throw away all non rank 0 center and make them the same as the rank 0
    // center
    MPI_Bcast(centers[0].data(), D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    cout << "center" << centers[0][0] << " " << centers[0][1] << " "
         << centers[0][2] << endl;

    MPI_Finalize();
}
