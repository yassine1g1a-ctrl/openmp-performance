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
#include "half.hpp"
using half_float::half;


#define AVAL 3.14f
#define BVAL 5.42f
#define TOL  0.05

int main(int argc, char **argv)
{
    int Ndim = 1000, Pdim = 1000, Mdim = 1000;   /* A[N][P], B[P][M], C[N][M] */
	int nbthreads = omp_get_max_threads();       // nombre de threads
	int i,j,k;
	half *A, *B, *C;
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
      
	
      A = (half *)malloc(Ndim*Pdim*sizeof(half));
      B = (half *)malloc(Pdim*Mdim*sizeof(half));
      C = (half *)malloc(Ndim*Mdim*sizeof(half));

	/* Initialize matrices */

	for (i=0; i<Ndim; i++)
		for (j=0; j<Pdim; j++)
			*(A+(i*Ndim+j)) = half(AVAL);

	for (i=0; i<Pdim; i++)
		for (j=0; j<Mdim; j++)
			*(B+(i*Pdim+j)) = half(BVAL);

	for (i=0; i<Ndim; i++)
		for (j=0; j<Mdim; j++)
			*(C+(i*Ndim+j)) = half(0.0f);

	/* Do the matrix product */
    
    // Timer products.
    struct timeval begin, end;

    gettimeofday( &begin, NULL );
    
    #pragma omp parallel for
    for (int i=0; i<Ndim; i++){
		for (int j=0; j<Mdim; j++){
			half tmp = half(0.0f);
			for(int k=0;k<Pdim;k++){
				/* C(i,j) = sum(over k) A(i,k) * B(k,j) */
				tmp += *(A+(i*Ndim+k)) *  *(B+(k*Pdim+j));
			}
			*(C+(i*Ndim+j)) = tmp;
		}
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
      mflops = 2.0 * dN * dN * dN/(1000000.0* time);
 
	printf(" N %d M %d P %d multiplication at %f mflops\n", Ndim, Mdim, Pdim, mflops);
	std::ofstream("stats_part4.csv", std::ios::app) << "half," << nbthreads << "," << Ndim << "," << Mdim << "," << Pdim << "," << time << "," << mflops << "\n";

	cval = (double)Pdim * 3.14 * 5.42;
	maxrel = 0.0;
	for (i=0; i<Ndim; i++){
		for (j=0; j<Mdim; j++){
			err = fabs( (double)(float)*(C+i*Ndim+j) - cval ) / cval;
			if (err > maxrel) maxrel = err;
		}
	}

	if (maxrel > TOL)
		printf("\n Errors in multiplication");
	else
		printf("\n Hey, it worked");


	printf("\n all done \n");
}
