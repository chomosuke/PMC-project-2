#include <stdio.h>
#include <stdlib.h>
#include <random>

// It would be cleaner to put seed and Gaussian_point into this class,
// but this allows them to be called like regular C functions.
// Feel free to make the whole code more C++-like.
class unit_normal {
    std::mt19937 gen;
    std::normal_distribution<double> d{0,1};
    public :
    void seed (long int s) { gen.seed (s); }
    double sample () { return d(gen); }
};

void
seed (unit_normal *un, long int s) {
    un->seed (s);
}

double *
Gaussian_point (double *out, unit_normal *un, int D, double *mean, double sd) {
    for (int i = 0; i < D; i++)
        out[i] = un->sample () * sd + mean[i];
    return out;
}



int
main (int argc, char *argv[])
{
    int N, D, c;        // number of points, dimensions and GMM components

    if (argc < 2) {
        fprintf (stderr, "usage: %s [input_file]\n", argv[0]);
        exit (1);
    }

    // Read number of points and dimension
    FILE *fp = fopen (argv[1], "r");
    fscanf (fp, "%d%d", &N, &D);

    N /= 1;     // Spread among ranks in MPI communicator

    // It would be cleaner to put this all in a class,
    // but this keeps the skeleton C-friendly.
    fscanf (fp, "%d", &c);

    double *gmm_mean_data = (double*)malloc(c*D * sizeof(*gmm_mean_data));
    double **gmm_means  = (double**)malloc(c * sizeof(*gmm_means));
    double *gmm_stddevs = (double*)malloc(c * sizeof(*gmm_stddevs));
    double *gmm_probs   = (double*)malloc(c * sizeof(*gmm_probs));

    for (int i = 0; i < c; i++) {
        gmm_means[i] = &(gmm_mean_data[i * D]);
        for (int d = 0; d < D; d++)
            fscanf (fp, "%lf", &gmm_means[i][d]);
        fscanf (fp, "%lg%lg", &gmm_stddevs[i], &gmm_probs[i]);
    }

    unit_normal un;

    // Example use
    seed (&un, 6);
    double *vec = (double*)malloc (D * sizeof(*vec));
    Gaussian_point (vec, &un, D, gmm_means[c-1], gmm_stddevs[c-1]);
    for (int i = 0; i < D; i++) {
        printf ("%lf ", vec[i]);
    }
    printf ("\n");
}
