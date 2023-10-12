#include "mpi.h"
#include <random>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define ROOT 0

mt19937 gen;

// It would be cleaner to put seed and Gaussian_point into this class,
// but this allows them to be called like regular C functions.
// Feel free to make the whole code more C++-like.
class unit_normal {
    normal_distribution<double> d{0, 1};

  public:
    double sample() { return d(gen); }
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

double get_distance_sq(double* p1, double* p2, int D) {
    double distance = 0;
    for (int i = 0; i < D; i++) {
        double c = p1[i] - p2[i];
        distance += c * c;
    }
    return distance;
}

double get_distance(double* p1, double* p2, int D) {
    return sqrt(get_distance_sq(p1, p2, D));
}

vector<int> k_means(double** means, vector<vector<double>>& points, int k,
                    int D, int rank) {
    int mean_count = k;

    double* sums_data = (double*)malloc(mean_count * D * sizeof(double));
    double** sums = (double**)malloc(mean_count * sizeof(double*));
    int* counts = (int*)malloc(mean_count * sizeof(int));
    double* all_sums_data;
    int* all_counts;
    if (rank == ROOT) {
        all_sums_data = (double*)malloc(mean_count * D * k * sizeof(double));
        all_counts = (int*)malloc(mean_count * k * sizeof(int));
    }

    bool mean_changed = true;
    vector<int> mean_indexes;
    while (mean_changed) {
        // Every node assign its own point to a center
        // TODO parallelize
        mean_indexes.clear();
        for (int i = 0; i < points.size(); i++) {
            int mi = 0;
            double d = get_distance_sq(means[mi], points[i].data(), D);
            for (int j = 1; j < mean_count; j++) {
                double nd = get_distance_sq(means[j], points[i].data(), D);
                if (nd < d) {
                    d = nd;
                    mi = j;
                }
            }
            mean_indexes.push_back(mi);
        }
        // now calculate the sum of each cluster excluding points on other nodes
        for (int i = 0; i < mean_count; i++) {
            counts[i] = 0;
            sums[i] = sums_data + i * D;
            for (int j = 0; j < D; j++) {
                sums[i][j] = 0;
            }
        }
        // TODO parallelize
        for (int i = 0; i < points.size(); i++) {
            int ci = mean_indexes[i];
            for (int j = 0; j < D; j++) {
                sums[ci][j] += points[i][j];
            }
            counts[ci]++;
        }

        // Send sum to ROOT to have all new means calculated
        MPI_Gather(sums_data, mean_count * D, MPI_DOUBLE, all_sums_data,
                   mean_count * D, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Gather(counts, mean_count, MPI_DOUBLE, all_counts, mean_count,
                   MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        // calculate the new mean
        mean_changed = false;
        if (rank == ROOT) {
            for (int i = 0; i < mean_count; i++) {
                int count = 0;
                for (int ni = 0; ni < k; ni++) {
                    // ni is the node index
                    // i is the offset, the cluster index
                    count += all_counts[i + ni * mean_count];
                }
                for (int j = 0; j < D; j++) {
                    double sum;
                    for (int ni = 0; ni < k; ni++) {
                        // i * D + j is the offset
                        // ni * cluster_count * D is the node stride
                        int stride = mean_count * D;
                        sum += all_sums_data[i * D + j + ni * stride];
                    }
                    double m = sum / count;
                    if (means[i][j] != m) {
                        mean_changed = true;
                    }
                    means[i][j] = m;
                }
            }
        }

        if (rank == ROOT) {
            free(all_sums_data);
            free(all_counts);
        }

        // Broadcast new means
        MPI_Bcast(means[0], mean_count * D, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
        MPI_Bcast(&mean_changed, 1, MPI_C_BOOL, ROOT, MPI_COMM_WORLD);
    }

    free(sums_data);
    free(sums);
    free(counts);

    return mean_indexes;
}

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

    gen.seed(42 + rank);

    unit_normal un;

    discrete_distribution dd(probs.begin(), probs.end());

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
        points.push_back(dists[dd(gen)].generate(&un));
    }

    // now we init the center
    // find distances in parallel
    // one process choose center and broadcast
    vector<vector<double>> centers;
    // first point can be choosen from the points generated locally
    centers.push_back(points[rand() % points.size()]);
    // throw away all non rank 0 center and make them the same as the rank 0
    // center
    MPI_Bcast(centers[0].data(), D, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);

    while (centers.size() < k) {
        // calculate the distance
        vector<double> distances;
        // TODO parallelize this
        for (int i = 0; i < points.size(); i++) {
            distances.push_back(
                get_distance_sq(centers[0].data(), points[i].data(), D));
            for (int j = 1; j < centers.size(); j++) {
                distances[i] =
                    min(distances[i], get_distance_sq(centers[j].data(),
                                                      points[i].data(), D));
            }
        }

        // sum the distance together and choose a node to choose the center.
        // TODO parallelize this
        double dist_sum = 0;
        for (int i = 0; i < distances.size(); i++) {
            dist_sum += distances[i];
        }

        // cout << "Node " << rank << " calculated: " << dist_sum << endl;

        // have one node choose a node
        double* all_sum;
        if (rank == ROOT) {
            all_sum = (double*)malloc(k * sizeof(double));
        }
        MPI_Gather(&dist_sum, 1, MPI_DOUBLE, all_sum, 1, MPI_DOUBLE, ROOT,
                   MPI_COMM_WORLD);

        int next_center_node;
        if (rank == ROOT) {

            // cout << "root recieved: ";
            // for (int i = 0; i < k; i++) {
            //     cout << all_sum[i] << " ";
            // }
            // cout << endl;

            discrete_distribution dd(all_sum, all_sum + k);
            next_center_node = dd(gen);
            free(all_sum);
        }
        MPI_Bcast(&next_center_node, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

        // Now choose that center
        vector<double> center;
        if (rank == next_center_node) {
            discrete_distribution dd(distances.begin(), distances.end());
            center = points[dd(gen)];
        } else {
            center = points[0];
        }

        // broadcast the center
        MPI_Bcast(center.data(), D, MPI_DOUBLE, next_center_node,
                  MPI_COMM_WORLD);
        centers.push_back(center);
    }

    // cout << "centers:" << endl;
    // for (int i = 0; i < centers.size(); i++) {
    //     for (int j = 0; j < D; j++) {
    //         cout << centers[i][j] << " ";
    //     }
    //     cout << endl;
    // }

    double* means_data = (double*)malloc(centers.size() * D * sizeof(double));
    double** means = (double**)malloc(centers.size() * sizeof(double*));
    for (int i = 0; i < centers.size(); i++) {
        means[i] = means_data + i * D;
        for (int j = 0; j < D; j++) {
            means[i][j] = centers[i][j];
        }
    }

    k_means(means, points, k, D, rank);

    MPI_Finalize();
}
