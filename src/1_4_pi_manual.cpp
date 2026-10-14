/*

This program will numerically compute the integral of

                  4/(1+x*x)

from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.

History: Written by Tim Mattson, 11/1999.
         Modified/extended by Jonathan Rouzaud-Cornabas, 10/2022
*/

#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <fstream>
#include <vector>
#include <omp.h>

static long long num_steps = 100000000;
double step;

int main (int argc, char** argv)
{
    int nb_threads = omp_get_max_threads();

    // Read command line arguments.
    for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 ) || ( strcmp( argv[ i ], "-num_steps" ) == 0 ) ) {
            num_steps = atoll( argv[ ++i ] );
            printf( "  User num_steps is %lld\n", num_steps );
        } else if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-num_threads" ) == 0 ) ) {
            nb_threads = atoi( argv[ ++i ] );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Pi Options:\n" );
            printf( "  -num_steps (-N) <int>:      Number of steps to compute Pi (by default 100000000)\n" );
            printf( "  -num_threads (-C) <int>:    Number of OpenMP threads\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
    }

    omp_set_num_threads(nb_threads);

    double pi, sum = 0.0;

    step = 1.0/(double) num_steps;

    // One slot per thread: each thread writes only to its own slot,
    // so no synchronization is needed during the parallel loop.
    std::vector<double> partial_sums(nb_threads, 0.0);

    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // Split [1, num_steps] into nb_threads contiguous chunks.
        long long chunk = num_steps / nb_threads;
        long long start = (long long)tid * chunk + 1;
        long long stop;

        if (tid == nb_threads - 1) {
            // Last thread absorbs the remainder of the integer division.
            stop = num_steps;
        } else {
            stop = start + chunk - 1;
        }

        double local_sum = 0.0;
        for (long long i = start; i <= stop; i++) {
            double x = (i - 0.5) * step;
            local_sum += 4.0 / (1.0 + x * x);
        }

        partial_sums[tid] = local_sum;
    }

    for (int t = 0; t < nb_threads; t++) {
        sum += partial_sums[t];
    }

    pi = step * sum;

    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
              1.0e-6 * ( end.tv_usec - begin.tv_usec );

    printf("\n pi with %lld steps is %lf in %lf seconds (threads=%d)\n ", num_steps, pi, time, nb_threads);

    std::ofstream csv("stats.csv", std::ios::app);
    csv << "manual," << nb_threads << "," << num_steps << "," << time << "\n";
    csv.close();
}
