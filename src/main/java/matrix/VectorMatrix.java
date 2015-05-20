package matrix;

import mpi.Intracomm;
import mpi.MPI;

import java.util.Arrays;
import java.util.Random;

public class VectorMatrix {

	public static void main( final String... args ) {
		MPI.Init( args );
		final Intracomm comm = MPI.COMM_WORLD;
		final int rank = comm.Rank();
		final int size = comm.Size();

		// Size of matrix
		final int n = 2 * size;
		final int m = 5;

		// Set seed to ensure same values
		final Random random = new Random( 2015 );

		// Generate vector x
		final double[] x = new double[ m ];
		for ( int i = 0; i < m; i++ ) {
			x[ i ] = random.nextDouble();
		}

		// Matrix a
		final double[] a = new double[ m * n ];

		// Generate matrix a
		if ( 0 == rank ) {
			for ( int i = 0; i < m * n; i++ ) {
				a[ i ] = random.nextDouble();
			}

			System.out.print( "A: " );
			System.out.println( Arrays.toString( a ) );
			System.out.print( "x: " );
			System.out.println( Arrays.toString( x ) );
		}

		// Number of rows we will do
		int local_n = n / size;

		// Scatter and get our peace
		final double[] local_a = new double[ m * local_n ];
		comm.Scatter( a, 0, local_n * m, MPI.DOUBLE, local_a, 0, m * n / size, MPI.DOUBLE, 0 );

		// Calculate ax
		double[] local_ax = multiply( local_a, x );

		// Gather and get solution
		double[] ax = new double[ n ];
		comm.Gather( local_ax, 0, local_n, MPI.DOUBLE, ax, 0, local_n, MPI.DOUBLE, 0 );

		if ( 0 == rank ) {
			System.out.print( "Ax: " );
			System.out.println( Arrays.toString( ax ) );
		}

		MPI.Finalize();
	}

	private static double[] multiply( final double[] a, final double[] x ) {
		// Calculate m and n
		final int m = x.length;
		final int n = a.length / m;

		// Throw exception if shape is wrong
		if ( 0 != a.length - n * m ) {
			throw new IllegalArgumentException( "Invalid shape" );
		}

		final double[] ax = new double[ n ];

		for ( int i = 0; i < n; i++ ) {
			for ( int j = 0; j < m; j++ ) {
				ax[ i ] += a[ i * m + j ] * x[ j ];
			}
		}

		return ax;
	}
}
