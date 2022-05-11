kernel void hist_image(global const uchar* A, global int* H) //histagram code.
{ 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//cumlative histagram
kernel void Cumlative_hs(global int* H, global int* CUMLATIVEH)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id+1; i< N && id; i++) 
	{
		atomic_add(&CUMLATIVEH[i], H[id]);
	}
}

//Look-up table.
kernel void LUT(global int* CUMLATIVEH, global int* LUT)
{
	int id = get_global_id(0);
	LUT[id] = CUMLATIVEH[id] *(double)255 / CUMLATIVEH[255];

}
//ouptus the image. copys images from A to Out
kernel void Back_Projecttion(global uchar* A, global int* LUT, global uchar* Out)
{
	int id = get_global_id(0);
	Out[id] = LUT[A[id]];
}

//!* END OF CODE *!


//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
//kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2) {
//	int id = get_global_id(0);
//	int lid = get_local_id(0);
//	int N = get_local_size(0);
//	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
//	scratch_1[lid] = A[id];

//	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

//	for (int i = 1; i < N; i *= 2) {
//		if (lid >= i)
//			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
//		else
//			scratch_2[lid] = scratch_1[lid];

//		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
//		scratch_3 = scratch_2;
//		scratch_2 = scratch_1;
//		scratch_1 = scratch_3;
//	}

	//copy the cache to output array
//	B[id] = scratch_1[lid];
//}

//Blelloch basic exclusive scan
//kernel void scan_bl(global int* A) {
//	int id = get_global_id(0);
//	int N = get_global_size(0);
//	int t;

	//up-sweep
//	for (int stride = 1; stride < N; stride *= 2) {
//		if (((id + 1) % (stride*2)) == 0)
//			A[id] += A[id - stride];

//		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
//	}

	//down-sweep
//	if (id == 0)
//		A[N-1] = 0;//exclusive scan
//
//	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
//
//	for (int stride = N/2; stride > 0; stride /= 2) {
//		if (((id + 1) % (stride*2)) == 0) {
//			t = A[id];
//			A[id] += A[id - stride]; //reduce 
//			A[id - stride] = t;		 //move
//		}

//		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
//	}
//}

//calculates the block sums
//kernel void block_sum(global const int* A, global int* B, int local_size) {
//	int id = get_global_id(0);
//	B[id] = A[(id+1)*local_size-1];
//}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
//kernel void scan_add_atomic(global int* A, global int* B) {
//	int id = get_global_id(0);
//	int N = get_global_size(0);
//	for (int i = id+1; i < N && id < N; i++)
//		atomic_add(&B[i], A[id]);
//}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
//kernel void scan_add_adjust(global int* A, global const int* B) {
//	int id = get_global_id(0);
//	int gid = get_group_id(0);
//	A[id] += B[gid];
//}