





extern "C" __global__
void convolution3dneuron(float *input, float*weights, float *output, float *bias
int ix,int iy, int iz,
int px, int py, int sx, int sy, int offset, int neuron){
int zoffset= iz/offset;
int y = -py+(sy*blockIdx.y);
int x = -px+(sx*blockIdx.x);
int ystep = threadIdx.y +y;
int xstep = threadIdx.x +x;
int neuronsectionx:=blockDim.y*blockIdx.x;
int neuronsectiony:=blockIdx.y;
int xtile = neuronsectionx+xstep;
int ytile = neuronsectiony+ystep;

extern __shared__ float sharedmemory[];

float *sharedweight=sharedmemory;
float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z*zoffset

    int neuronsection = neuron*neuronsize;
//    int outputsection = neuron*gridDim.x*gridDim.y*gridDim.z;
    sharedweight[rowsection+colsection+depsection+i]= weights[];
// __syncthreads();
if( ystep < inputy && ystep >= 0 && xstep < inputx && xstep>=0){
  
    sectionedinput[neuronsectionx+neuronsectiony + threadIdx.z]=input[(xtile)+(ytile)+threadIdx.z];
   
}else{
    sectionedinput[neuronsectionx+neuronsectiony + threadIdx.z]=0.0
}
__syncthreads();
float adder=sectionedinput[neuronsectionx+neuronsectiony + threadIdx.z]*sharedweight[neuronsectionx+neuronsectiony + threadIdx.z]; 
atomicAdd(&output[(neuron*gridDim.x*gridDim.y)+(blockIdx.x*gridDim.y)+(blockIdx.y)],adder);
__syncthreads();

}