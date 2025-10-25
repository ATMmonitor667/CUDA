//nvcc -O2 pointindexing_template.cu -o pointindexing -I /home/microway/cuda-samples/Common/
//./pointindexing  100 2
//./pointindexing  10000000 10

#include <helper_functions.h>
#include <helper_cuda.h>

#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/partition.h>
#include <thrust/scan.h>
#include <thrust/copy.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <assert.h>
#include <iostream>
#include <iterator>
#include <sys/time.h>
#include <time.h>

typedef unsigned short ushort;
typedef unsigned char uchar;

using namespace std;

float calc_time(char *msg,timeval t0, timeval t1)
{
 	long d = t1.tv_sec*1000000+t1.tv_usec - t0.tv_sec * 1000000-t0.tv_usec;
 	float t=(float)d/1000;
 	if(msg!=NULL)
 		printf("%s ...%10.3f\n",msg,t);
 	return t;
}

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

struct point2d
{
    ushort x,y;
    __host__ __device__
    point2d() {}

    __host__ __device__
    point2d(ushort _x, ushort _y) : x(_x), y(_y) {}

};

struct xytor
{

  int lev;

  __host__ __device__
  xytor(int lev): lev(lev) {}

    __host__ __device__
    uint operator()(point2d p )
    {
	    ushort x = p.x;
	    ushort y = p.y;
	    {
		ushort a=x>>(16-lev);
		ushort b=y>>(16-lev);
		return b*(1<<lev)+a;
	    }
    }
};


int main(int argc, char *argv[])
{
    if (argc!=3)
    {
        printf("USAGE: %s #points lev(0<lev<16)\n", argv[0]);
        exit(1);
    }

    int num_points = atoi(argv[1]);
    if(num_points<0) num_points=10000;
    printf("num_points=%d\n",num_points);

    int run_lev=atoi(argv[2]);
    if(run_lev<0) run_lev=10;
    printf("run_lev=%d\n",run_lev);

    vector<unsigned int> da(num_points);
    timeval s0, s1,s2,s3,s4,s5,s6,s7;

    //allocate host memory
    point2d * h_points=new point2d[num_points];
    uint *h_cellids=new uint[num_points];

    point2d * dptr_points=NULL;
    uint *dptr_cellids=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_points,num_points* sizeof(point2d)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_cellids,num_points* sizeof(uint)));
    assert(dptr_points!=NULL&&dptr_cellids!=NULL);

    //generate points
    gettimeofday(&s0, NULL);
    for(int i=0;i<num_points;i++)
    {
    	point2d p;
    	p.x=random()%65536;
    	p.y=random()%65536;
    	h_points[i]=p;
    }
    gettimeofday(&s1, NULL);
    calc_time("generating random points\n",s0,s1);

    // copy point data from CPU host to GPU device
    HANDLE_ERROR( cudaMemcpy( dptr_points, h_points, num_points * sizeof(point2d), cudaMemcpyHostToDevice ) );
    gettimeofday(&s2, NULL);
    calc_time("trnasferring data to GPU\n",s1,s2);

    thrust::device_ptr<point2d> d_points=thrust::device_pointer_cast(dptr_points);
    thrust::device_ptr<uint> d_cellids =thrust::device_pointer_cast(dptr_cellids);

    //====================================================================================================
    //YOUR WORK below: Step 1- transform point coordinates to cell identifiers; pay attention to functor xytor
    //thrust::transform(...);
    thrust::transform(d_points, d_points + num_points, d_cellids, xytor(run_lev));
    cudaDeviceSynchronize();
    gettimeofday(&s3, NULL);
    calc_time("transforming..............\n",s2,s3);

  thrust::stable_sort_by_key(
    d_cellids,
    d_cellids + num_points,
    d_points
);
    cudaDeviceSynchronize();
    gettimeofday(&s4, NULL);
    calc_time("sorting..............\n",s3,s4);

    uint *dptr_PKey=NULL;
    uint *dptr_PLen=NULL;
    uint *dptr_PPos=NULL;
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PKey,num_points* sizeof(uint)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PLen,num_points* sizeof(uint)));
    HANDLE_ERROR( cudaMalloc( (void**)&dptr_PPos,num_points* sizeof(uint)));
    assert(dptr_PKey!=NULL&&dptr_PLen!=NULL&&dptr_PPos!=NULL);
    thrust::device_ptr<uint> d_PKey =thrust::device_pointer_cast(dptr_PKey);
    thrust::device_ptr<uint> d_PLen=thrust::device_pointer_cast(dptr_PLen);
    thrust::device_ptr<uint> d_PPos=thrust::device_pointer_cast(dptr_PPos);

    //YOUR WORK below: Step 3- reduce by key
    //use  d_cellids as the first input vector and thrust::constant_iterator<int>(1) as the second input
    size_t num_cells=0;//num_cells is initialized to 0 just to make the template compile; it should be updated next
    // num_cells = thrust::reduce_by_key(...).first - d_PKey
    num_cells = thrust::reduce_by_key(d_cellids, d_cellids + num_points, thrust::constant_iterator<int>(1), d_PKey, d_PLen);
    size_t num_cells = new_end.first - d_PKey;
    cudaDeviceSynchronize();
    gettimeofday(&s5, NULL);
    calc_time("reducing.......\n",s4,s5);

    //YOUR WORK below: Step 4-  exclusive scan using d_PLen as the input and d_PPos as the output
    //thrust::exclusive_scan(...)
    thrust::exclusive_scan(d_PLen, d_PLen + num_cells, d_PPos);
    cudaDeviceSynchronize();
    gettimeofday(&s6, NULL);
    calc_time("scan.......\n",s5,s6);
    //====================================================================================================
    //transferring data back to CPU
    uint *h_PKey=new uint[num_cells];
    uint *h_PLen=new uint[num_cells];
    uint *h_PPos=new uint[num_cells];
    HANDLE_ERROR( cudaMemcpy( h_points, dptr_points, num_points * sizeof(point2d), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( h_cellids, dptr_cellids, num_points * sizeof(uint), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( h_PKey, dptr_PKey, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( h_PLen, dptr_PLen, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) );
    HANDLE_ERROR( cudaMemcpy( h_PPos, dptr_PPos, num_cells * sizeof(uint), cudaMemcpyDeviceToHost) );
    gettimeofday(&s7, NULL);
    calc_time("transferring back to CPU.......\n",s6,s7);

    //you would have to override the output opertor of point2d to output points to std::cout
    //thrust::copy(h_points, h_points+num_cells, std::ostream_iterator<point2d>(std::cout, " "));

    //alternatively, you can access h_points array and print out x/y
    int point_out=(num_points>50)?50:num_points;
    for(int i=0;i<point_out;i++)
    {
     	point2d p=h_points[i];
     	printf("(%d,%d)",p.x,p.y);
     }
    printf("\n");

    cout<<"cell identifiers:";
    thrust::copy(h_cellids, h_cellids+point_out, std::ostream_iterator<uint>(std::cout, " "));
    cout<<endl;
    int cell_out=(num_cells>20)?20:num_cells;
    cout<<"unique cell identifiers:";
    thrust::copy(h_PKey, h_PKey+cell_out, std::ostream_iterator<uint>(std::cout, " "));
    cout<<endl;
    cout<<"number of points in cells:";
    thrust::copy(h_PLen, h_PLen+cell_out, std::ostream_iterator<uint>(std::cout, " "));
    cout<<endl;
    cout<<"starting point position in cells:";
    thrust::copy(h_PPos, h_PPos+cell_out, std::ostream_iterator<uint>(std::cout, " "));
    cout<<endl;

     //clean up
    cudaFree(dptr_points);
    cudaFree(dptr_cellids);
    cudaFree(dptr_PKey);
    cudaFree(dptr_PLen);
    cudaFree(dptr_PPos);
    delete[] h_points;
    delete[] h_cellids;
    delete[] h_PKey;
    delete[] h_PLen;
    delete[] h_PPos;
}
