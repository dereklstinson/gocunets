#define _3DMEMBLOCK(x,y,z) (x)*(y)*(z) 
#define _2DMEMLOCATION(x,y) (x)*(y)
#define SPLOC(p,s,i) -(p)+((s)*(i))




extern "C" __global__
void Forward3dneuron(float *input, float*weights, float *output, float *bias,
int ix,int iy, int iz,
int px, int py,
int sx, int sy, 
int offset,int totaloffset,
int neuron, int numofneurons){

//int y = SPLOC(py,sy,blockIdx.y)
int y = -py+(sy*blockIdx.y);
int x = -px+(sx*blockIdx.x);
int ystep = threadIdx.y +y;
int xstep = threadIdx.x +x;
//Since the convolution will be done in parts on the depth side we have to multiply the depth of the input.
int neuronsectionx=blockDim.y*iz*blockIdx.x;
int neuronsectiony=blockIdx.y*iz;
int neuronsectionz=blockIdx.z+(offset*blockDim.z);//This is the section neuron we are calculating in the depth side
int xtile = neuronsectionx+xstep;
int ytile = neuronsectiony+ystep;
int ztile = neuronsectionz; // since there is no slide or padding on the depth side we don't have to add a step, but since we are doing this in parts on the host side we have to specify where at in memory we are exactly


int xsection = iz*blockDim.y*threadIdx.x;
int ysection = iz*threadIdx.y;
int zsection = blockDim.z*offset;

extern __shared__ float sharedmemory[];
float *sharedweight=sharedmemory;
float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z;
    
 
    //putting the weights into shared memory so all of the weights in the zsection will be mapped to a single thread. 
sharedweight[xsection+ysection+threadIdx.z]= weights[xsection+ysection+zsection];
if( ystep < iy && ystep >= 0 && xstep < ix && xstep>=0){    
  //  sectionedinput[neuronsectionx+neuronsectiony + threadIdx.z]=input[(xtile)+(ytile)+threadIdx.z];
  //since the gridDims are equaling the output of the neuron. it is going to be a one to one multiply with the weights
  sectionedinput[xsection+ysection+threadIdx.z]=input[xtile+ytile+ztile];
}else{
    sectionedinput[xsection+xsection + threadIdx.z]=0.0;
}
__syncthreads();
float adder=0.0;
adder+=sectionedinput[xsection+ysection + threadIdx.z]*sharedweight[xsection+ysection+ threadIdx.z];
//atomicAdd(&adder,); 
__syncthreads();
if (offset==totaloffset-1){
    adder+=bias[neuron];
}
int outputsectionx = numofneurons* gridDim.y*blockIdx.x;
int outputsectiony = numofneurons* blockIdx.y;
atomicAdd(&output[outputsectionx+outputsectiony+neuron],adder);
__syncthreads();

}






extern "C" __global__
void Backward3dneuron(float *input, float*weights,float *gradadds, float *grads, float *plgrads,  float *biasgradadds,
    int ix,int iy, int iz,
    int px, int py, 
    int sx, int sy, 
    int offset, int totaloffset,
    int neuron, int numofneurons){
    int y = -py+(sy*blockIdx.y);
    int x = -px+(sx*blockIdx.x);
    int ystep = threadIdx.y +y;
    int xstep = threadIdx.x +x;
    //Since the convolution will be done in parts on the depth side we have to multiply the depth of the input.
    int neuronsectionx=blockDim.y*iz*blockIdx.x;
    int neuronsectiony=blockIdx.y*iz;
    int neuronsectionz=blockIdx.z+(offset*blockDim.z);//This is the section neuron we are calculating in the depth side
    int xtile = neuronsectionx+xstep;
    int ytile = neuronsectiony+ystep;
    int ztile = neuronsectionz; // since there is no slide or padding on the depth side we don't have to add a step, but since we are doing this in parts on the host side we have to specify where at in memory we are exactly
    
    int xsection = iz*blockDim.y*threadIdx.x;
    int ysection = iz*threadIdx.y;
    int zsection = blockDim.z*offset;
    int gradsectionx = numofneurons* gridDim.y*blockIdx.x;
    int gradsectiony = numofneurons* blockIdx.y;


    float grad= grads[gradsectionx+gradsectiony+neuron];   
    if( ystep < iy && ystep >= 0 && xstep < ix && xstep>=0){    
        atomicAdd(&gradadds[xsection+ysection+zsection],input[xtile+ytile+ztile]*grad);
        atomicAdd(&plgrads[xtile+ytile+ztile],weights[xsection+ysection+zsection]*grad);
    }
    __syncthreads();
    biasgradadds[neuron]+=grad;


    }