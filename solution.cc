#include "mpi.h"
#include <assert.h>
#include <fstream>
#include <iostream>
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

double** init_points(int size, int D) {
    double* points_data = (double*)malloc(size * D * sizeof(double));
    double** points = (double**)malloc(size * sizeof(double*));
    for (int i = 0; i < size; i++) {
        points[i] = points_data + i * D;
    }
    return points;
}

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
        memset(counts, 0, mean_count * sizeof(int));
        memset(sums_data, 0, mean_count * D * sizeof(double));
        for (int i = 0; i < mean_count; i++) {
            sums[i] = sums_data + i * D;
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
    memset(local_cluster_sizes, 0, num_cluster * sizeof(int));
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
        // TODO Change this to Alltoallv
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

double get_distance_to_hyperplane(double* hyperplane, double* point, int D) {
    double n = 0;
    for (int i = 0; i < D; i++) {
        n += hyperplane[i] * point[i];
    }
    n += hyperplane[D];
    double d = 0;
    for (int i = 0; i < D; i++) {
        d += hyperplane[D] * hyperplane[D];
    }
    return n / sqrt(d);
}

typedef struct node {
    double* center_of_mass;
    int mass;
    double* b1;
    // b2 should be larger
    double* b2;
    int num_children;
    struct node** children;
} node;

void encode(node* root, vector<uint8_t>& bytes, int D) {
    bytes.insert(bytes.end(), (uint8_t*)root->center_of_mass,
                 (uint8_t*)(root->center_of_mass + D));
    bytes.insert(bytes.end(), (uint8_t*)&root->mass,
                 (uint8_t*)(&root->mass + 1));
    bytes.insert(bytes.end(), (uint8_t*)root->b1, (uint8_t*)(root->b1 + D));
    bytes.insert(bytes.end(), (uint8_t*)root->b2, (uint8_t*)(root->b2 + D));
    bytes.insert(bytes.end(), (uint8_t*)&root->num_children,
                 (uint8_t*)(&root->num_children + 1));
    for (int i = 0; i < root->num_children; i++) {
        encode(root->children[i], bytes, D);
    }
    free(root->center_of_mass);
    free(root->children);
    free(root);
}

node* decode(uint8_t* bytes, int* size_read, int D) {
    node* root = (node*)malloc(sizeof(node));
    root->center_of_mass = (double*)malloc(D * sizeof(double));
    memcpy(root->center_of_mass, bytes + *size_read, D * sizeof(double));
    *size_read += D * sizeof(double);
    memcpy(&root->mass, bytes + *size_read, sizeof(int));
    *size_read += sizeof(int);
    root->b1 = (double*)malloc(D * sizeof(double));
    memcpy(root->b1, bytes + *size_read, D * sizeof(double));
    *size_read += D * sizeof(double);
    root->b2 = (double*)malloc(D * sizeof(double));
    memcpy(root->b2, bytes + *size_read, D * sizeof(double));
    *size_read += D * sizeof(double);
    memcpy(&root->num_children, bytes + *size_read, sizeof(int));
    *size_read += sizeof(int);
    root->children = (node**)malloc(root->num_children * sizeof(node*));
    for (int i = 0; i < root->num_children; i++) {
        root->children[i] = decode(bytes, size_read, D);
    }
    return root;
}
// TODO test encode and decode

node* construct(vector<double*>& points, vector<double>& b1, vector<double>& b2,
                int D) {
    node* root = (node*)malloc(sizeof(node));
    root->center_of_mass = (double*)malloc(D * sizeof(double));
    root->b1 = (double*)malloc(D * sizeof(double));
    memcpy(root->b1, b1.data(), D * sizeof(double));
    root->b2 = (double*)malloc(D * sizeof(double));
    memcpy(root->b2, b2.data(), D * sizeof(double));
    if (points.size() == 1) {
        root->num_children = 0;
        root->mass = 1;
        memcpy(root->center_of_mass, points[0], D * sizeof(double));
    } else {
        vector<node*> children;
        unsigned long long permutation = 0;
        while (permutation < pow(2, D)) {
            // construct b1 & b2
            vector<double> nb1, nb2;
            for (int i = 0; i < D; i++) {
                if ((permutation >> i) & 1) {
                    nb1.push_back(b1[i]);
                    nb2.push_back((b1[i] + b2[i]) / 2);
                } else {
                    nb1.push_back((b1[i] + b2[i]) / 2);
                    nb2.push_back(b2[i]);
                }
            }
            vector<double*> npoints;
            for (int i = 0; i < points.size(); i++) {
                bool within = true;
                for (int j = 0; j < D; j++) {
                    if (points[i][j] < nb1[j] || points[i][j] >= nb2[j]) {
                        within = false;
                        break;
                    }
                }
                if (within) {
                    npoints.push_back(points[i]);
                }
            }
            if (npoints.size() > 0) {
                children.push_back(construct(npoints, nb1, nb2, D));
            }

            permutation++;
        }
        root->num_children = children.size();
        root->children = (node**)malloc(root->num_children * sizeof(node*));
        memcpy(root->children, children.data(),
               root->num_children * sizeof(node*));

        root->center_of_mass = (double*)malloc(D * sizeof(double));
        memset(root->center_of_mass, 0, D * sizeof(double));
        root->mass = 0;
        for (int i = 0; i < root->num_children; i++) {
            for (int j = 0; j < D; j++) {
                root->center_of_mass[j] += root->children[i]->center_of_mass[j];
            }
            root->mass += root->children[i]->mass;
        }
        for (int j = 0; j < D; j++) {
            root->center_of_mass[j] /= root->num_children;
        }
    }
    return root;
}

#define THETA 0.1
node* get_partial(node* full, double* hyperplane, int D) {
    node* partial = (node*)malloc(sizeof(node));

    partial->center_of_mass = (double*)malloc(D * sizeof(double));
    memcpy(partial->center_of_mass, full->center_of_mass, D * sizeof(double));
    partial->b1 = (double*)malloc(D * sizeof(double));
    memcpy(partial->b1, full->b1, D * sizeof(double));
    partial->b2 = (double*)malloc(D * sizeof(double));
    memcpy(partial->b2, full->b2, D * sizeof(double));
    partial->mass = full->mass;

    if (abs(full->b2[0] - full->b1[0]) /
            get_distance_to_hyperplane(hyperplane, full->center_of_mass, D) <
        THETA) {
        // Do not traverse further
        partial->num_children = 0;
        partial->children = NULL;
    } else {
        // Do traverse further
        partial->num_children = full->num_children;
        partial->children =
            (node**)malloc(partial->num_children * sizeof(node*));
        for (int i = 0; i < full->num_children; i++) {
            partial->children[i] =
                get_partial(full->children[i], hyperplane, D);
        }
    }

    return partial;
}

uint8_t* alltoallv(uint8_t* buf_send, int* counts_send,
                   /*out*/ int* counts_recv, int k) {
    assert(MPI_SUCCESS == MPI_Alltoall(counts_send, 1, MPI_INT, counts_recv, 1,
                                       MPI_INT, MPI_COMM_WORLD));
    int* displacements_send = (int*)malloc(k * sizeof(int));
    int total_send = 0;
    for (int i = 0; i < k; i++) {
        displacements_send[i] = total_send;
        total_send += counts_send[i];
    }

    int total_recv = 0;
    int* displacements_recv = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        displacements_recv[i] = total_recv;
        total_recv += counts_recv[i];
    }
    uint8_t* buf_recv = (uint8_t*)malloc(total_recv * sizeof(uint8_t));
    assert(MPI_SUCCESS ==
           MPI_Alltoallv(buf_send, counts_send, displacements_send, MPI_UINT8_T,
                         buf_recv, counts_recv, displacements_recv, MPI_UINT8_T,
                         MPI_COMM_WORLD));
    free(displacements_send);
    free(displacements_recv);
    return buf_recv;
}

#define G 9.8
vector<double> get_acc(double* point, node* root, int D) {
    vector<double> acc;
    double d = get_distance(point, root->center_of_mass, D);
    if (abs(root->b2[0] - root->b1[0]) / d < THETA) {
        // use this center of mass
        for (int i = 0; i < D; i++) {
            // TODO verify this
            acc.push_back(G * root->mass *
                          (root->center_of_mass[i] - point[i]) / d / d / d);
        }
    } else {
        for (int i = 0; i < D; i++) {
            acc.push_back(0);
        }
        for (int i = 0; i < root->num_children; i++) {
            vector<double> acc_i = get_acc(point, root->children[i], D);
            for (int j = 0; j < D; j++) {
                acc[j] += acc_i[j];
            }
        }
    }
    return acc;
}

vector<double> get_acc(double* point, vector<node*> trees, int D) {
    vector<double> acc;
    for (int i = 0; i < D; i++) {
        acc.push_back(0);
    }
    for (int i = 0; i < trees.size(); i++) {
        vector<double> acc_i = get_acc(point, trees[i], D);
        for (int j = 0; j < D; j++) {
            acc[j] += acc_i[j];
        }
    }
    return acc;
}

#define DT 0.1
void simulate(int local_cluster_size, double** local_cluster,
              double** velocities, int k, int D, int rank) {
    // We want to reduce the communication overhead.
    // We can do that by sending other nodes center of mass directly.
    //
    // Simple method: If a cluster is close, send them all our points, otherwise
    // send them center of gravity only.
    //
    // Complex method: take the closest point, and give it all the center of
    // masses according to the Barnes-Hut method
    //
    // Which one is faster?
    //
    // Build the tree in both methods will make things faster.
    //
    // The second method have less communication overhead, However the problem
    // is that the point closest to the center of mass isn't the point closest
    // to some other points
    //
    // To solve this, instead of using the closest point, we use the hyperplane
    // that seperate the two clusters.
    // If two cluster do not touch, then we choose a hyperplane that's
    // perpendicular to the line from the closest point to the center of mass
    // In fact even if they do touch, we choose the hyperplane in the same way
    // anyways.
    //
    // Steps:
    // - 1 Construct the tree.
    // - 2 Choose the closest point.
    // - 3 Calculate the hyperplane.
    // - 4 Send the partial tree to the corrisponding cluster
    // - 5 move the points.

    int num_means = k;

    // Construct the tree
    // Datastructure for the tree, is just a regular tree
    // Use encode / decode t2 send
    vector<double*> points;
    vector<double> b1;
    vector<double> b2;
    for (int j = 0; j < D; j++) {
        b1.push_back(local_cluster[0][j]);
        b2.push_back(local_cluster[0][j]);
    }
    for (int i = 0; i < local_cluster_size; i++) {
        points.push_back(local_cluster[i]);
        for (int j = 0; j < D; j++) {
            b1[j] = min(b1[j], local_cluster[i][j]);
            b2[j] = max(b2[j], local_cluster[i][j]);
        }
    }
    double max_diff = 0;
    for (int j = 0; j < D; j++) {
        max_diff = max(max_diff, b2[j] - b1[j]);
    }
    for (int j = 0; j < D; j++) {
        double cur_diff = b2[j] - b1[j];
        b2[j] += (max_diff - cur_diff) / 2;
        b1[j] = b2[j] - max_diff;
    }
    node* all_root = construct(points, b1, b2, D);

    double** means = init_points(num_means, D);
    assert(MPI_SUCCESS == MPI_Allgather(all_root->center_of_mass, D, MPI_DOUBLE,
                                        means[0], D, MPI_DOUBLE,
                                        MPI_COMM_WORLD));

    // Choose the closest point:
    double* closest_points_data =
        (double*)malloc(num_means * D * sizeof(double));
    for (int i = 0; i < num_means; i++) {
        if (i == rank) {
            memset(closest_points_data + D * i, 0, D * sizeof(double));
            continue;
        }
        double* mean = means[i];
        double* closest_point = (double*)malloc(D * sizeof(double));
        memcpy(closest_point, local_cluster[0], D * sizeof(double));
        double distance = get_distance_sq(closest_point, mean, D);
        for (int j = 1; j < local_cluster_size; j++) {
            double nd = get_distance_sq(local_cluster[j], mean, D);
            if (nd < distance) {
                distance = nd;
                memcpy(closest_point, local_cluster[j], D * sizeof(double));
            }
        }
        memcpy(closest_points_data + D * i, closest_point, D * sizeof(double));
    }
    double* other_closest_points =
        (double*)malloc(num_means * D * sizeof(double));
    assert(MPI_SUCCESS == MPI_Alltoall(closest_points_data, D, MPI_DOUBLE,
                                       other_closest_points, D, MPI_DOUBLE,
                                       MPI_COMM_WORLD));
    free(closest_points_data);
    closest_points_data = other_closest_points;
    double** closest_points = (double**)malloc(num_means * sizeof(double*));
    for (int i = 0; i < num_means; i++) {
        closest_points[i] = closest_points_data + i * D;
    }

    // closest_points[i] is the closest point in cluster[i] compared to the
    // center of gravity in local cluster

    // Calculate the hyperplane
    double* hyperplanes_data =
        (double*)malloc(num_means * (D + 1) * sizeof(double));
    double** hyperplanes = (double**)malloc(num_means * sizeof(double));
    // hyperplane[i] is defined by hyperplane[i][0:D] . x + hyperplane[i][D] = 0
    for (int i = 0; i < num_means; i++) {
        hyperplanes[i] = hyperplanes_data + i * (D + 1);
    }
    for (int i = 0; i < num_means; i++) {
        if (i == rank) {
            continue;
        }
        // v.x = 0, any point on the hyperplane is orthogonal to v. We just need
        // to solve for d in v.x + d = 0.
        // In this case v is mean - closest_points
        for (int j = 0; j < D; j++) {
            hyperplanes[i][j] = means[rank][j] - closest_points[i][j];
        }
        // Solve for d
        double d;
        for (int j = 0; j < D; j++) {
            d -= hyperplanes[i][j] * closest_points[i][j];
        }
        hyperplanes[i][D] = d;
    }

    // Construct the partial tree for each rank
    vector<node*> partials;
    for (int i = 0; i < k; i++) {
        if (i == rank) {
            partials.push_back(NULL);
            continue;
        }
        node* partial = get_partial(all_root, hyperplanes[i], D);
        partials.push_back(partial);
    }

    // Send the partial tree to each node
    vector<uint8_t> bytes;
    vector<int> lengths;
    int prev_len = 0;
    for (int i = 0; i < k; i++) {
        if (i == rank) {
            lengths.push_back(0);
            continue;
        }
        encode(partials[i], bytes, D);
        lengths.push_back(bytes.size() - prev_len);
        prev_len = bytes.size();
    }
    int* counts_recv = (int*)malloc(k * sizeof(int));
    uint8_t* recv = alltoallv(bytes.data(), lengths.data(), counts_recv, k);
    int size_read = 0;
    vector<node*> trees;
    for (int i = 0; i < k; i++) {
        if (i == rank) {
            trees.push_back(all_root);
            continue;
        }
        trees.push_back(decode(recv, &size_read, D));
    }
    free(recv);
    free(counts_recv);

    // Move
    // TODO parallelize
    for (int i = 0; i < k; i++) {
        double* point = local_cluster[i];
        vector<double> acc = get_acc(point, trees, D);
        for (int j = 0; j < D; j++) {
            point[j] += velocities[i][j] * DT;
            velocities[i][j] += acc[j] * DT;
        }
    }

    free(means[0]);
    free(means);
    free(hyperplanes_data);
    free(hyperplanes);
    free(closest_points_data);
    free(closest_points);
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
    if (local_N <= 0) {
        fprintf(stderr,
                "This program isn't designed for less points than nodes!");
        exit(1);
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

    free(means_data);
    free(means);

    int local_cluster_size;
    double** local_cluster = localize_clusters(cluster_indices, points, k, D,
                                               rank, &local_cluster_size);

    cout << rank << " with size " << local_cluster_size << endl;

    double** velocities = init_points(local_cluster_size, D);

    for (int i = 0; i < 100; i++) {
        simulate(local_cluster_size, local_cluster, velocities, k, D, rank);

        char fname[20];
        sprintf(fname, "points/%d.csv", i);
        ofstream PointsFile(fname);
        for (int j = 0; j < local_cluster_size; j++) {
            PointsFile << local_cluster[j][0];
            for (int l = 1; l < D; l++) {
                PointsFile << "," << local_cluster[j];
            }
            PointsFile << endl;
        }
        PointsFile.close();
    }

    free(velocities[0]);
    free(velocities);
    free(local_cluster[0]);
    free(local_cluster);

    assert(MPI_SUCCESS == MPI_Finalize());
}
