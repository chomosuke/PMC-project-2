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
    // cout << "Generating " << local_N << " points for rank " << rank << " of "
    //      << k << "." << endl;
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

        // cout << "Node " << rank << " calculated: " << dist_sum << endl;

        // have one node choose a node
        double* all_sum;
        if (rank == ROOT) {
            all_sum = (double*)malloc(k * sizeof(double));
        }
        assert(MPI_SUCCESS == MPI_Gather(&dist_sum, 1, MPI_DOUBLE, all_sum, 1,
                                         MPI_DOUBLE, ROOT, MPI_COMM_WORLD));

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

    // for (int i = 0; i < points.size(); i++) {
    //     cout << "if (rank == " << rank << ") { vector<double> point = {";
    //     for (int j = 0; j < points[i].size(); j++) {
    //         cout << points[i][j] << ",";
    //     }
    //     cout << "}; points.push_back(point);}" << endl;
    // }
    // MPI_Barrier(MPI_COMM_WORLD);
    // if (rank == ROOT) {
    //     for (int i = 0; i < centers.size(); i++) {
    //         cout << "{ vector<double> center = {";
    //         for (int j = 0; j < D; j++) {
    //             cout << centers[i][j] << ",";
    //         }
    //         cout << "}; centers.push_back(center); }" << endl;
    //     }
    // }

    points.clear();
    centers.clear();
if (rank == 1) { vector<double> point = {0.437627,-0.461309,1.42949,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {1.69033,0.606546,-0.042529,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {1.49342,0.913428,-0.151974,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.401872,-0.676448,0.847558,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {-1.03552,0.802107,0.805672,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {-0.519111,-0.602802,1.07313,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {1.27367,-0.2164,-0.345803,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {-0.564612,-0.0469421,0.398866,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {1.16126,-0.0899912,-0.957433,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {1.1617,-0.295409,0.640704,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.956422,-0.308522,0.22578,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.234866,0.721007,1.04114,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.234185,0.498596,-0.433086,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.79427,0.437712,1.08716,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.53603,-0.40207,0.893108,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.379998,0.749575,0.037738,}; points.push_back(point);}
if (rank == 1) { vector<double> point = {0.27467,-0.138112,1.3226,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.0907966,-0.332523,-0.580055,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-0.898982,-1.16526,0.964221,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.274307,-0.608763,1.07754,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {1.41912,0.559555,-0.772087,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-0.583517,0.015329,-0.285858,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-0.00753908,-0.215867,0.622569,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-0.235338,0.623613,0.789214,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.517017,-0.0461052,0.2208,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {1.13351,-0.334442,0.0433022,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-1.74634,0.299006,-0.0672895,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.493062,1.19308,-0.886285,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.248514,0.952064,0.431532,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.0266555,0.126029,0.493548,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {1.53327,0.991902,0.127066,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {-0.0919352,0.111465,0.991082,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {1.37508,-0.329687,-0.29244,}; points.push_back(point);}
if (rank == 2) { vector<double> point = {0.528223,0.333859,0.546994,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.832453,1.78745,1.80279,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.156959,0.506663,-0.21795,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {1.23693,0.232857,0.204187,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.238215,0.396373,0.0517061,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.541075,0.762534,0.4523,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.528475,0.571934,0.0758415,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.306265,0.97852,0.298993,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {1.3317,0.429207,-0.420707,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.480709,0.818336,1.04378,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.177734,-0.176596,-1.46683,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.0876226,0.817862,0.779631,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.46783,0.562034,1.76266,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-0.25545,0.144566,0.569997,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {-1.12965,0.909742,0.210276,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.0141931,0.247405,0.605205,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {1.46701,0.323114,-0.800702,}; points.push_back(point);}
if (rank == 3) { vector<double> point = {0.875875,0.115605,-0.249681,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.554065,0.244569,-0.462333,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.245276,0.537032,-0.605456,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {-0.972347,0.45628,2.85552,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.698644,-0.290408,0.35489,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {-1.34684,-2.32478,0.749704,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {-1.42057,1.84907,-1.78566,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.106131,1.09734,0.85404,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {1.37894,-0.51494,0.0262021,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {-0.813055,-1.11539,0.583302,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.353138,0.0528083,-0.326091,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.347769,-0.653631,1.41274,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.475728,-1.46705,0.317618,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {1.27122,0.822338,0.157255,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.516025,0.772641,-1.03566,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {0.861207,-0.83256,-0.329725,}; points.push_back(point);}
if (rank == 4) { vector<double> point = {1.27705,-0.155654,-0.679069,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.0961046,-0.0257084,0.92043,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.66008,0.635923,-1.27469,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.160775,-0.578884,0.439316,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.759999,0.0920385,1.92735,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-0.104456,0.327652,1.80847,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.0827594,1.03441,0.131258,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-0.0465181,-0.461334,1.03624,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-0.243312,0.296525,0.896709,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-1.84472,0.285053,0.581208,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-0.716746,-0.124192,-0.788083,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {1.20402,-0.533894,-0.534646,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.443984,0.434707,1.42032,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.568857,-0.320578,-0.58955,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.854125,0.0383558,0.210573,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {0.346937,-0.892734,-0.447448,}; points.push_back(point);}
if (rank == 5) { vector<double> point = {-0.559655,-0.365537,1.19781,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.271872,-0.307715,0.401097,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {2.14166,-2.01096,-0.492803,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-0.482619,0.164165,0.233095,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.117995,0.0222218,-0.427793,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-0.251044,-0.163867,-1.47633,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.74349,0.56717,0.947627,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.8533,-0.0149193,0.0475629,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.664654,0.779193,-1.1011,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.815915,0.187782,-0.596579,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-0.204526,-0.68897,0.823442,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-0.0891671,-0.521948,0.0863469,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.324199,0.785532,2.65601,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.236597,0.329595,2.17037,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.535493,-0.754979,1.36075,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-1.87568,-1.36678,0.636305,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {-0.45336,0.34922,1.0868,}; points.push_back(point);}
if (rank == 0) { vector<double> point = {0.551832,-0.0559939,0.734471,}; points.push_back(point);}
{ vector<double> center = {0.535493,-0.754979,1.36075,}; centers.push_back(center); }
{ vector<double> center = {1.27122,0.822338,0.157255,}; centers.push_back(center); }
{ vector<double> center = {-0.832453,1.78745,1.80279,}; centers.push_back(center); }
{ vector<double> center = {1.27367,-0.2164,-0.345803,}; centers.push_back(center); }
{ vector<double> center = {-0.177734,-0.176596,-1.46683,}; centers.push_back(center); }
{ vector<double> center = {0.0907966,-0.332523,-0.580055,}; centers.push_back(center); }

    double* means_data = (double*)malloc(centers.size() * D * sizeof(double));
    double** means = (double**)malloc(centers.size() * sizeof(double*));
    for (int i = 0; i < centers.size(); i++) {
        means[i] = means_data + i * D;
        for (int j = 0; j < D; j++) {
            means[i][j] = centers[i][j];
        }
    }

    k_means(means, points, k, D, rank);

    cout << "means: " << endl;
    for (int i = 0; i < centers.size(); i++) {
        for (int j = 0; j < D; j++) {
            cout << means[i][j] << " ";
        }
        cout << endl;
    }

    free(means_data);
    free(means);

    assert(MPI_SUCCESS == MPI_Finalize());
}
