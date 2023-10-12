#include "mpi.h"
#include <assert.h>
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
        assert(MPI_SUCCESS == MPI_Gather(sums_data, mean_count * D, MPI_DOUBLE,
                                         all_sums_data, mean_count * D,
                                         MPI_DOUBLE, ROOT, MPI_COMM_WORLD));
        assert(MPI_SUCCESS == MPI_Gather(counts, mean_count, MPI_INT,
                                         all_counts, mean_count, MPI_INT, ROOT,
                                         MPI_COMM_WORLD));
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
                    double sum = 0;
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

        // Broadcast new means
        assert(MPI_SUCCESS == MPI_Bcast(means[0], mean_count * D, MPI_DOUBLE,
                                        ROOT, MPI_COMM_WORLD));
        assert(MPI_SUCCESS ==
               MPI_Bcast(&mean_changed, 1, MPI_C_BOOL, ROOT, MPI_COMM_WORLD));
    }

    if (rank == ROOT) {
        free(all_sums_data);
        free(all_counts);
    }

    free(sums_data);
    free(sums);
    free(counts);

    return mean_indexes;
}

double** localize_clusters(vector<int>& cluster_indices,
                           vector<vector<double>>& points, int k, int D,
                           int rank, int* cluster_size) {
    int num_cluster = k;

    // move each cluster to a node

    // calculate size of each cluster
    int* local_cluster_sizes = (int*)malloc(num_cluster * sizeof(int));
    for (int i = 0; i < num_cluster; i++) {
        local_cluster_sizes[i] = 0;
    }
    for (int i = 0; i < cluster_indices.size(); i++) {
        local_cluster_sizes[cluster_indices[i]]++;
    }
    int* all_cluster_sizes;
    all_cluster_sizes = (int*)malloc(num_cluster * k * sizeof(int));
    assert(MPI_SUCCESS == MPI_Allgather(local_cluster_sizes, num_cluster,
                                        MPI_INT, all_cluster_sizes, num_cluster,
                                        MPI_INT, MPI_COMM_WORLD));
    int* cluster_sizes = (int*)malloc(num_cluster * sizeof(int));
    for (int i = 0; i < num_cluster; i++) {
        int count = 0;
        for (int ni = 0; ni < k; ni++) {
            // ni is the node index
            // i is the offset, the cluster index
            count += all_cluster_sizes[i + ni * num_cluster];
        }
        cluster_sizes[i] = count;
    }

    // group local points into clusters
    vector<double*> local_clusters_pre;
    vector<int> filled_so_far;
    for (int i = 0; i < num_cluster; i++) {
        local_clusters_pre.push_back(
            (double*)malloc(local_cluster_sizes[i] * D * sizeof(double)));
        filled_so_far.push_back(0);
    }
    for (int i = 0; i < cluster_indices.size(); i++) {
        int ci = cluster_indices[i];
        for (int j = 0; j < D; j++) {
            local_clusters_pre[ci][filled_so_far[ci] * D + j] = points[i][j];
        }
        filled_so_far[ci]++;
    }

    // transfer local points to node where the cluster belongs to.
    double** local_cluster =
        (double**)malloc(cluster_sizes[rank] * sizeof(double*));
    double* local_cluster_data =
        (double*)malloc(cluster_sizes[rank] * D * sizeof(double));

    int* counts_recv = (int*)malloc(k * sizeof(int));
    int* displacements = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        // You want to send local_clusters_pre[i] to local_cluster_data
        // fill counts_recv & displacements
        for (int j = 0; j < k; j++) {
            counts_recv[j] = all_cluster_sizes[j * k + i] * D;
        }
        int d = 0;
        for (int j = 0; j < k; j++) {
            displacements[j] = d;
            d += counts_recv[j];
        }
        assert(MPI_SUCCESS ==
               MPI_Gatherv(local_clusters_pre[i], local_cluster_sizes[i] * D,
                           MPI_DOUBLE, local_cluster_data, counts_recv,
                           displacements, MPI_DOUBLE, i, MPI_COMM_WORLD));
    }
    free(counts_recv);
    free(displacements);

    // organize local_cluster
    for (int i = 0; i < cluster_sizes[rank]; i++) {
        local_cluster[i] = local_cluster_data + i * D;
    }

    for (int i = 0; i < num_cluster; i++) {
        free(local_clusters_pre[i]);
    }

    *cluster_size = cluster_sizes[rank];
    return local_cluster;
}

int main(int argc, char** argv) {
    // argc = 2;
    if (argc < 2) {
        fprintf(stderr, "usage: %s [input_file]\n", argv[0]);
        exit(1);
    }

    // Read number of points and dimension
    FILE* fp = fopen(argv[1], "r");
    // FILE* fp = fopen("input.txt", "r");
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

    assert(MPI_SUCCESS == MPI_Init(&argc, &argv));
    int rank, k;
    assert(MPI_SUCCESS == MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    assert(MPI_SUCCESS == MPI_Comm_size(MPI_COMM_WORLD, &k));

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
    assert(MPI_SUCCESS ==
           MPI_Bcast(centers[0].data(), D, MPI_DOUBLE, ROOT, MPI_COMM_WORLD));

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

        // have one node choose a node
        double* all_sum;
        if (rank == ROOT) {
            all_sum = (double*)malloc(k * sizeof(double));
        }
        assert(MPI_SUCCESS == MPI_Gather(&dist_sum, 1, MPI_DOUBLE, all_sum, 1,
                                         MPI_DOUBLE, ROOT, MPI_COMM_WORLD));

        int next_center_node;
        if (rank == ROOT) {
            discrete_distribution dd(all_sum, all_sum + k);
            next_center_node = dd(gen);
            free(all_sum);
        }
        assert(MPI_SUCCESS ==
               MPI_Bcast(&next_center_node, 1, MPI_INT, ROOT, MPI_COMM_WORLD));

        // Now choose that center
        vector<double> center;
        if (rank == next_center_node) {
            discrete_distribution dd(distances.begin(), distances.end());
            center = points[dd(gen)];
        } else {
            center = points[0];
        }

        // broadcast the center
        assert(MPI_SUCCESS == MPI_Bcast(center.data(), D, MPI_DOUBLE,
                                        next_center_node, MPI_COMM_WORLD));
        centers.push_back(center);
    }

    double* means_data = (double*)malloc(centers.size() * D * sizeof(double));
    double** means = (double**)malloc(centers.size() * sizeof(double*));
    for (int i = 0; i < centers.size(); i++) {
        means[i] = means_data + i * D;
        for (int j = 0; j < D; j++) {
            means[i][j] = centers[i][j];
        }
    }

    vector<int> cluster_indices = k_means(means, points, k, D, rank);
    assert(cluster_indices.size() == points.size());

    cout << "means: " << endl;
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < D; j++) {
            cout << means[i][j] << " ";
        }
        cout << endl;
    }
    free(means_data);
    free(means);

    int cluster_size;
    double** local_cluster =
        localize_clusters(cluster_indices, points, k, D, rank, &cluster_size);

    double mean[3] = {0, 0, 0};
    for (int i = 0; i < cluster_size; i++) {
        for (int j = 0; j < D; j++) {
            mean[j] += local_cluster[i][j];
        }
    }
    cout << "mean for rank " << rank;
    for (int j = 0; j < D; j++) {
        mean[j] /= cluster_size;
        cout << " " << mean[j];
    }
    cout << endl;

    free(local_cluster[0]);
    free(local_cluster);

    assert(MPI_SUCCESS == MPI_Finalize());
}
