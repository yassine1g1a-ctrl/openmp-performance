/*
**  PROGRAM: Matrix Multiply
**
**  PURPOSE: This is a simple matrix multiply program. 
**           It will compute the product
**
**                C  = A * B
**
**           A and B are set to constant matrices so we
**           can make a quick test of the multiplication.
**
**
**  HISTORY: Written by Tim Mattson, Nov 1999.
*            Modified and extended by Jonathan Rouzaud-Cornabas, Oct 2022
*/


#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>
#include <omp.h>
#include <fstream>
#include <cmath>

#define AVAL 3.14
#define BVAL 5.42
#define TOL  1e-9

void matmul(double *A, double *B, double *C, int Ndim, int Pdim, int Mdim)
{
    int i,j,k;
    double tmp;
    for (i=0; i<Ndim; i++){
        for (j=0; j<Mdim; j++){
            tmp = 0.0;
            for(k=0;k<Pdim;k++){
                tmp += *(A+(i*Ndim+k)) *  *(B+(k*Pdim+j));
            }
            *(C+(i*Ndim+j)) = tmp;
        }
    }
}

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	int nbthreads = omp_get_max_threads();       // nombre de threads
	int i,j;
	double *in[8];                                 // A, B, C, D, E, F, G, H
	double *AB, *CD, *EF, *GH, *ABCD, *EFGH, *R;
	double cval, err, maxrel;
    double dN, dM, dP, mflops;



    
    // Read command line arguments.
      for ( int i = 0; i < argc; i++ ) {
        if ( ( strcmp( argv[ i ], "-N" ) == 0 )) {
            Ndim = atoi( argv[ ++i ] );
            printf( "  User N is %d\n", Ndim );
        } else if ( ( strcmp( argv[ i ], "-M" ) == 0 )) {
            Mdim = atoi( argv[ ++i ] );
            printf( "  User M is %d\n", Mdim );
        } else if ( ( strcmp( argv[ i ], "-P" ) == 0 )) {
            Pdim = atoi( argv[ ++i ] );
            printf( "  User P is %d\n", Pdim );
        } else if ( ( strcmp( argv[ i ], "-C" ) == 0 ) || ( strcmp( argv[ i ], "-nbthreads" ) == 0 ) ) {
            nbthreads = atoi( argv[ ++i ] );
            omp_set_num_threads( nbthreads );
            printf( "  User nbthreads is %d\n", nbthreads );
        } else if ( ( strcmp( argv[ i ], "-h" ) == 0 ) || ( strcmp( argv[ i ], "-help" ) == 0 ) ) {
            printf( "  Matrix multiplication Options:\n" );
            printf( "  -N <int>:              Size of the dimension N (by default 1000)\n" );
            printf( "  -M <int>:              Size of the dimension M (by default 1000)\n" );
            printf( "  -P <int>:              Size of the dimension P (by default 1000)\n" );
            printf( "  -C <int>:              Number of threads (default: max available)\n" );
            printf( "  -help (-h):            print this message\n\n" );
            exit( 1 );
        }
      }
      
	
    for (int m=0; m<8; m++) {
		in[m] = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
		for (long x=0; x<(long)Ndim*Ndim; x++)
			in[m][x] = AVAL;                   /* Initialize matrices */
	}
	AB   = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	CD   = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	EF   = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	GH   = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	ABCD = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	EFGH = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));
	R    = (double *)malloc((size_t)Ndim*Ndim*sizeof(double));


	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
    #pragma omp parallel
    #pragma omp single
    {
        /* niveau 1 : 4 produits independants */
        #pragma omp task
        matmul(in[0], in[1], AB, Ndim, Pdim, Mdim);
        #pragma omp task
        matmul(in[2], in[3], CD, Ndim, Pdim, Mdim);
        #pragma omp task
        matmul(in[4], in[5], EF, Ndim, Pdim, Mdim);
        #pragma omp task
        matmul(in[6], in[7], GH, Ndim, Pdim, Mdim);
        #pragma omp taskwait

        /* niveau 2 : 2 produits independants */
        #pragma omp task
        matmul(AB, CD, ABCD, Ndim, Pdim, Mdim);
        #pragma omp task
        matmul(EF, GH, EFGH, Ndim, Pdim, Mdim);
        #pragma omp taskwait

        /* niveau 3 : le produit final */
        matmul(ABCD, EFGH, R, Ndim, Pdim, Mdim);
	}
	/* Check the answer */


    gettimeofday( &end, NULL );

    // Calculate time.
    double time = 1.0 * ( end.tv_sec - begin.tv_sec ) +
                1.0e-6 * ( end.tv_usec - begin.tv_usec );
                
	printf(" N %d M %d P %d multiplication in %f seconds \n", Ndim, Mdim, Pdim, time);

      dN = (double)Ndim;
      dM = (double)Mdim;
      dP = (double)Pdim;
      mflops = 7.0 * 2.0 * dN * dN * dN/(1000000.0* time);
 
	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);
	std::ofstream("stats_part5.csv", std::ios::app) << "chain," << nbthreads << "," << Ndim << "," << Mdim << "," << Pdim << "," << time << "," << mflops << "\n";

	cval = pow((double)Ndim, 7) * pow(AVAL, 8);
	maxrel = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = fabs( *(R+i*Ndim+j) - cval ) / cval;
			if (err > maxrel) maxrel = err;
		}
	}

	if (maxrel > TOL)
		printf("\n Errors in multiplication");
	else
		printf("\n Hey, it worked");

	printf("\n all done \n");
}
