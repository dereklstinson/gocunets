#define KH 5
#define KL 5
#define KD 32
#define IH 32
#define IL 32
#define ID 32
#define PH 0
#define PL 0
#define SH 1
#define SL 1
#define FCIN 512
#define FCW   512
#define FCOUT 512

/*
For debugging purposes

//for Maxwell RUN:
//nvcc --gpu-architecture=compute_50 --gpu-code=compute_50 --ptx kernels32.cu
//for Pascal RUN:
//nvcc --gpu-architecture=compute_61 --gpu-code=compute_61 --ptx kernels32.cu
//for TX2 RUN:
//nvcc --gpu-architecture=compute_62 --gpu-code=compute_62 --ptx kernels32.cu
//for Volta RUN:
//nvcc --gpu-architecture=compute_70 --gpu-code=compute_70 --ptx kernels32.cu
*/




/*
extern "C" __global__
void convolution(float *input, float *kernel, float *output){
    int il=threadIdx.x;
    int ih=threadIdx.y;
    int id=threadIdx.z;
    int bl=blockIdx.x;
    int bh=blockIdx.y;
__shared__ float []newout;
}
*/

/*
*********************************************************************
function name: convoludion1d
description: dot product of two matrix (not only square)
parameters: 
            input - Pointer to 1d array
            weights - Pointer to 2d array of weights for a layer
            output - pointer for 1d array of output for layer
            px     - is padding
            stride - is the stride
            inputsize - is the length of the input
            weightperneuron - is the number of weights per neuron
            
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further speedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
/*
extern "C" __global__
void convoludion1d(float *input, float *weights, float *output, int px, int stridex,int inputsize, int weightperneuron){
   int col =blockIdx.x*blockDim.x+threadIdx.x;// number of blocks * the size of the blocks + the index inside the block
   int stridecol = blockIdx.x*blockDim.x+threadIdx.x; //this should be the sectioned output 
   int row = blockIdx.y*blockDim.y+threadIdx.y;
   


}
*/
/*
*********************************************************************
function name: convoludion2d
description: dot product of two matrix (not only square)
parameters: 
            input      - Pointer to 2d array
            weights    - Pointer to 2d array of weights for a layer
            output     - pointer for 2d array of output for layer
            px         - is the padding for x dim
            py         - is the padding for y dim
            sx         - is the stride for x dim
            sy         - is the stride for y dim
            gridDim.y  - is the output dim y
            gridDim.x  - is the output dim x
            blockDim.y - is the kernel dim y
            blockDim.x - is the kernel dim x
           
            inputsize - is the length of the input
            weightperneuron - is the number of weights per neuron
            

   
return: none
*********************************************************************
*/

extern "C" __global__
void convolution2d(float *input, float *weights, float *output, int inputx, int inputy, int px, int py, int sx, int sy){
    int row = -py+(sy*blockIdx.y);
    int col = -px+(sx*blockIdx.x);
    int rowstep = threadIdx.y +row;
    int colstep = threadIdx.x +col;
      extern __shared__ float sharedmemory[];
    float *sharedweight=sharedmemory;
    float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y;
    sharedweight[threadIdx.y*blockDim.x+threadIdx.x]=weights[threadIdx.y*blockDim.x+threadIdx.x];
    __syncthreads();
    //float adder=0.0;
    if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0){
        sectionedinput[threadIdx.y*blockDim.x+threadIdx.x]=input[rowstep*gridDim.y+colstep];
        }else{
       sectionedinput[threadIdx.y*blockDim.x+threadIdx.x]=0.0;
   }
  __syncthreads();
   float adder=sectionedinput[threadIdx.y*blockDim.x+threadIdx.x]* sharedweight[threadIdx.y*blockDim.x+threadIdx.x];
    atomicAdd(&output[blockIdx.y*gridDim.x+blockIdx.x],adder);
    __syncthreads();
}

extern "C" __global__
void convolution3d(float *input,float *weights, float*output,
int inputx, int inputy, int inputz, 
int px, int py, int pz,
int sx, int sy, int sz){

    int row = -py+(sy*blockIdx.y);
    int col = -px+(sx*blockIdx.x);
    int dep = -pz+(sz*blockIdx.z);
    int rowstep = threadIdx.y +row;
    int colstep = threadIdx.x +col;
    int depstep = threadIdx.z +dep;
    int rowsection = threadIdx.y*blockDim.x*blockDim.z;
    int colsection = threadIdx.x*blockDim.z;
    int depsection = threadIdx.z;
    int rowtile = rowstep*gridDim.y*gridDim.z;
    int coltile = colstep*gridDim.z;
    int deptile= depstep;
    extern __shared__ float sharedmemory[];
    float *sharedweight=sharedmemory;
    float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z;
    sharedweight[rowsection+colsection+depsection]= weights[rowsection+colsection+depsection];
    __syncthreads();

    if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && depstep<inputz && depstep>=0 ){
       sectionedinput[rowsection+colsection+depsection]=input[(rowtile)+(coltile)+deptile];
    }else{
       sectionedinput[rowsection+colsection+depsection]=0.0;
    }
   __syncthreads();

   float adder=sectionedinput[rowsection+colsection+depsection]*sharedweight[rowsection+colsection+depsection]; 
   atomicAdd(&output[(blockIdx.y*gridDim.x*gridDim.z)+(blockIdx.x*gridDim.z)+blockIdx.z],adder);
   __syncthreads();
}

extern "C" __global__
void makezero(float *output){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int dep = blockIdx.z * blockDim.z + threadIdx.z;
output[row+col+dep]=0.0;

}
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
extern "C" __global__
void backprop4dlayer(float *input, float *weights, float*erroradder, float *grads, float*returngrads,
    int inputx, int inputy, int inputz, 
    int px, int py, int pz,
    int sx, int sy, int sz,
    int depth, int neurons){
        int row = -py+(sy*blockIdx.y);
        int col = -px+(sx*blockIdx.x);
        int dep = -pz+(sz*blockIdx.z);
        int rowstep = threadIdx.y +row;
        int colstep = threadIdx.x +col;
        int depstep = threadIdx.z +dep;
        int neuronsize = blockDim.y*blockDim.x*blockDim.z*depth;
        int rowsection = threadIdx.y*blockDim.x*blockDim.z*depth;
        int colsection = threadIdx.x*blockDim.z*depth;
        int depsection = threadIdx.z*depth;
        int rowtile = rowstep*gridDim.y*gridDim.z*depth;
        int coltile = colstep*gridDim.z*depth;
        int deptile= depstep*depth;
        extern __shared__ float sharedmemory[];
        float *sharedweight=sharedmemory;
        float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z*depth;  
        for(int neuron=0;neuron<neurons;neuron++){
            int neuronsection = neuron*neuronsize;
            int outputsection = neuron*gridDim.x*gridDim.y*gridDim.z;
        for (int i=0;i<depth;i++){
            sharedweight[rowsection+colsection+depsection+i]= weights[(neuronsection)+rowsection+colsection+depsection+i];
        }
       
        __syncthreads();
   
        if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && depstep<inputz && depstep>=0 ){
            for (int i=0;i<depth;i++){
            sectionedinput[rowsection+colsection+depsection+i]=input[(rowtile)+(coltile)+deptile+i];
            }
       
          
        }else{
            for (int i = 0;i<depth;i++){
           sectionedinput[rowsection+colsection+depsection+i]=0.0;
            }
        }
       __syncthreads();
       float adder=0.0;
       for (int i = 0;i<depth;i++){
      adder+=sectionedinput[rowsection+colsection+depsection+i]*sharedweight[rowsection+colsection+depsection+i]; 
       }
       atomicAdd(&returngrads[outputsection+(blockIdx.y*gridDim.x*gridDim.z)+(blockIdx.x*gridDim.z)+blockIdx.z],adder);
       __syncthreads();
    }

}
extern "C" __global__ 
void convolution4dneuron(float *input,float *weights,  float *biases, float *output,
    int inputx, int inputy, int inputz, 
    int px, int py, int pz,
    int sx, int sy, int sz,
    int depth, int neuron){
    
        int row = -py+(sy*blockIdx.y);
        int col = -px+(sx*blockIdx.x);
        int dep = -pz+(sz*blockIdx.z);
        int rowstep = threadIdx.y +row;
        int colstep = threadIdx.x +col;
        int depstep = threadIdx.z +dep;
        int neuronsize = blockDim.y*blockDim.x*blockDim.z*depth;
        int rowsection = threadIdx.y*blockDim.x*blockDim.z*depth;
        int colsection = threadIdx.x*blockDim.z*depth;
        int depsection = threadIdx.z*depth;
        int rowtile = rowstep*gridDim.y*gridDim.z*depth;
        int coltile = colstep*gridDim.z*depth;
        int deptile= depstep*depth;
        extern __shared__ float sharedmemory[];
        float *sharedweight=sharedmemory;
        float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z*depth;  
       
            int neuronsection = neuron*neuronsize;
        //    int outputsection = neuron*gridDim.x*gridDim.y*gridDim.z;
            float bias=biases[neuron];
        for (int i=0;i<depth;i++){
            sharedweight[rowsection+colsection+depsection+i]= weights[(neuronsection)+rowsection+colsection+depsection+i];
        }
       // __syncthreads();
        if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && depstep<inputz && depstep>=0 ){
            for (int i=0;i<depth;i++){
            sectionedinput[rowsection+colsection+depsection+i]=input[(rowtile)+(coltile)+deptile+i];
            }
        }else{
            for (int i = 0;i<depth;i++){
           sectionedinput[rowsection+colsection+depsection+i]=0.0;
            }
        }
       __syncthreads();
       float adder=0.0;
       for (int i = 0;i<depth;i++){
      adder+=sectionedinput[rowsection+colsection+depsection+i]*sharedweight[rowsection+colsection+depsection+i]; 
       }
      adder+=bias;

       atomicAdd(&output[(blockIdx.y*gridDim.x*gridDim.z*gridDim.y)+(blockIdx.x*gridDim.z*gridDim.y)+(blockIdx.z*gridDim.y)+neuron],adder);
       __syncthreads();

    }
    /*
//important: for this function in order to optimize it the memory reads need to be in depth major. 
Or a output per neuron major.  Which is different from a cpu. Where it would be a depth last.
also gridDim.x needs to be the total number of kernel sections 
*/
extern "C" __global__
void unroll4dinput(float *input, float*output, 
    int inputx, int inputy, int inputz, 
    int px, int py, int pz,
    int sx, int sy, int sz, 
    int depth){
        int row = -py+(sy*blockIdx.y);
        int col = -px+(sx*blockIdx.x);
        int wid = -pz+(sz*blockIdx.z);
        int sharedsize= blockDim.x*blockDim.y*blockDim.z*depth; //size of the unroll per kernel 4d kernel this should be the size of the shared memory
        int rowstep = threadIdx.y +row;
        int colstep = threadIdx.x +col;
        int widstep = threadIdx.z +wid;
        int rowsection = threadIdx.y*blockDim.x*blockDim.z;
        int colsection = threadIdx.x*blockDim.z;
        int widsection = threadIdx.z;
        int rowtile = rowstep*gridDim.y*gridDim.z;
        int coltile = colstep*gridDim.z;
        int widtile= widstep;
        int chunk = blockDim.x*blockDim.y*blockDim.z;
        extern __shared__ float sharedmemory[];
      
        
        if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && widstep<inputz && widstep>=0 ){
            for (int i=0;i<depth;i++){
            sharedmemory[(i*chunk)+rowsection+colsection+widsection]=input[(i*chunk)+(rowtile)+(coltile)+widtile];
            }
        }else{
            for (int i = 0;i<depth;i++){
                sharedmemory[(i*chunk)+rowsection+colsection+widsection]=0.0;
            }
        }
       __syncthreads();
       int unrolledsection=(blockIdx.y*gridDim.x*gridDim.z*gridDim.y)+(blockIdx.x*gridDim.z*gridDim.y)+(blockIdx.z*gridDim.y);
       for (int i=0;i<depth;i++){
        output[(unrolledsection*sharedsize)+(i*chunk)+rowsection+colsection+widsection]= sharedmemory[(i*chunk)+rowsection+colsection+widsection];
       }
       
    }

/*
//Version 2 might behave better than the other one. I will have to do a test case.  
Threads have to be multiples of 32. Only one thread dim will be used and that is for the depth of the output per neuron.
the x,y,z are going to be for loops per thread.    
*/
extern "C" __global__
void unroll4dinput_v2(float *input, float *output,
    int inputx, int inputy, int inputz, 
    int px, int py, int pz,
    int sx, int sy, int sz, 
    int kx, int ky, int kz){
    int row = -py+(sy*blockIdx.y);
    int col = -px+(sx*blockIdx.x);
    int wid = -pz+(sz*blockIdx.z);
    int chunk = kx*ky*kz;
    int sharedsize= chunk*blockDim.x; //size of the unroll per kernel 4d kernel this should be the size of the shared memory
  
   
  
    extern __shared__ float sharedmemory[];
    
    for (int i = 0;i<ky;i++){
        int rowstep = ky +row;
        for (int j=0;j<kx;j++){
            int colstep = kx +col;
            for (int k=0;k<kz;k++){
                int widstep = kz +wid;
                int rowsection = (i*kx*kz);
                int colsection = (j*kz);
                if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && widstep<inputz && widstep>=0 ){
              
                    sharedmemory[(threadIdx.x*chunk)+rowsection+colsection+k]=input[(threadIdx.x*chunk)+(rowstep*gridDim.y*gridDim.z)+(colstep*gridDim.z)+widstep];
                    
                }else{
                 
                    sharedmemory[(threadIdx.x*chunk)+rowsection+colsection+k]=0.0;
                   
                }  
            }
        }
    }
__syncthreads();

int unrolledsection=(blockIdx.y*gridDim.x*gridDim.z*gridDim.y)+(blockIdx.x*gridDim.z*gridDim.y)+(blockIdx.z*gridDim.y);
//this is most likely not going to work.  but I will test it out.
for (int i=0;i<chunk;i++){
 output[(unrolledsection*sharedsize)+threadIdx.x*chunk+i]= sharedmemory[(threadIdx.x*chunk)+i];
}
  
   }


extern "C" __global__
void convolution4dlayer(float *input,float *weights,  float *biases, float *output,
    int inputx, int inputy, int inputz, 
    int px, int py, int pz,
    int sx, int sy, int sz,
    int depth, int neurons){
    
        int row = -py+(sy*blockIdx.y);
        int col = -px+(sx*blockIdx.x);
        int dep = -pz+(sz*blockIdx.z);
        int rowstep = threadIdx.y +row;
        int colstep = threadIdx.x +col;
        int depstep = threadIdx.z +dep;
        int neuronsize = blockDim.y*blockDim.x*blockDim.z*depth;
        int rowsection = threadIdx.y*blockDim.x*blockDim.z*depth;
        int colsection = threadIdx.x*blockDim.z*depth;
        int depsection = threadIdx.z*depth;
        int rowtile = rowstep*gridDim.y*gridDim.z*depth;
        int coltile = colstep*gridDim.z*depth;
        int deptile= depstep*depth;
        extern __shared__ float sharedmemory[];
        float *sharedweight=sharedmemory;
        float  *sectionedinput =sharedmemory +blockDim.x*blockDim.y*blockDim.z*depth;  
        for(int neuron=0;neuron<neurons;neuron++){
            int neuronsection = neuron*neuronsize;
       //     int outputsection = neuron*gridDim.x*gridDim.y*gridDim.z;
            float bias=biases[neuron];
        for (int i=0;i<depth;i++){
            sharedweight[rowsection+colsection+depsection+i]= weights[(neuronsection)+rowsection+colsection+depsection+i];
        }
      
        if( rowstep < inputy && rowstep >= 0 && colstep < inputx && colstep>=0 && depstep<inputz && depstep>=0 ){
            for (int i=0;i<depth;i++){
            sectionedinput[rowsection+colsection+depsection+i]=input[(rowtile)+(coltile)+deptile+i];
            }
        }else{
            for (int i = 0;i<depth;i++){
           sectionedinput[rowsection+colsection+depsection+i]=0.0;
            }
        }
       __syncthreads();
       float adder=0.0;
       for (int i = 0;i<depth;i++){
      adder+=sectionedinput[rowsection+colsection+depsection+i]*sharedweight[rowsection+colsection+depsection+i]; 
       }
      adder+=bias;
       atomicAdd(&output[(blockIdx.y*gridDim.x*gridDim.z*gridDim.y)+(blockIdx.x*gridDim.z*gridDim.y)+(blockIdx.z*gridDim.y)+neuron],adder);
       __syncthreads();
    }
    }



extern "C" __global__
void fcnnlayer(float *input,float *weights, float *output){
    int NeuronIndex  = blockIdx.x;
    int WeightIndex = threadIdx.x;
    int InputIndex   = threadIdx.y;
    
    __shared__ float sharedinput[FCIN];
    __shared__ float sharedweights[FCW];
    __shared__ float sharedoutput[FCOUT];
    sharedoutput[NeuronIndex]=0.0;
    if (InputIndex<FCIN){
        sharedinput[InputIndex]=input[InputIndex];
        sharedweights[NeuronIndex]=weights[NeuronIndex*FCW+WeightIndex];
    }
    __syncthreads();
    if (NeuronIndex<FCOUT && InputIndex<FCIN && WeightIndex<FCW){
        int temp = sharedinput[InputIndex]*sharedweights[WeightIndex];
        atomicAdd(&sharedoutput[NeuronIndex],temp);
    }
    __syncthreads();
    output[NeuronIndex]=sharedoutput[NeuronIndex];
    __syncthreads();
}
/*
*********************************************************************
function name: gpu_matrix_mult
description: dot product of two matrix (not only square)
parameters: 
            &a GPU device pointer to a m X n matrix (A)
            &b GPU device pointer to a n X k matrix (B)
            &c GPU device output purpose pointer to a m X k matrix (C) 
            to store the result
Note:
    grid and block should be configured as:
        dim3 dimGrid((k + BLOCK_SIZE - 1) / BLOCK_SIZE, (m + BLOCK_SIZE - 1) / BLOCK_SIZE);
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    further speedup can be obtained by using shared memory to decrease global memory access times
return: none
*********************************************************************
*/
extern "C" __global__
void gpu_matrix_mult(float *a,float *b, float *c, int m, int n, int k){ 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;
    if( col < k && row < m){
        for(int i = 0; i < n; i++){
            sum += a[row * n + i] * b[i * k + col];
        }
        c[row * k + col] = sum;
    }
} 



extern "C" __global__
void test(float *Din1, float *Din2, float *Dout){
    int idx = threadIdx.x;
        Dout[idx]=Din1[idx]*Din2[idx];
}




/*
__global__ void UnRoll(float *Dinput,float *Dout,int it,int,ih, int il, int kt, int kh, int kl, int st, int sh, int sl){
int ut=(it+(2*pt));
int uh=(ih+(2*ph));
int ul=(il+(2*pl));
int sc=(it-kt+(2*pt))/st)+1;
int sc=(ih-kh+(2*ph))/sh)+1;
int sc=(i-kl+(2*pl))/sl)+1;

}

__global__ void ConvUnroll1Const(float* device_x, float* device_a, int itime,int,iheight, int ktime, int ilength, int kheight, int klength, int stime, int sheight, int slength) {
    __shared__ float  inputImg1[25][32];
    int thIdx = threadIdx.x;
    int thIdy = threadIdx.y;
    int imgIndex = blockIdx.x;
    
    // one block will be responsible for whole output image
    // so each block need to iterate 24 * 24 / 32 = 18 times, where (24 * 24) is output image size
    for (int i = 0; i < 18; i++) {
        float acc = 0.0f;
        // load image data 
        if (thIdy < 25) { // X_unroll out_H = 25
            int feaIdxY = thIdy / 5; // position in [5*5] mask
            int feaIdxX = thIdy % 5;
            int inputFeaStartPixelY = (i * 32 + thIdx) / 24; // top-left position in input image
            int inputFeaStartPixelX = (i * 32 + thIdx) % 24;;
            int inputIndex = imgIndex * 28 * 28 + (inputFeaStartPixelY + feaIdxY) * 28 + inputFeaStartPixelX + feaIdxX; // pixel index to be loaded by this thread
            inputImg1[thIdy][thIdx] = device_x[inputIndex];
        }
        __syncthreads();
        
        // matrix multiplication
        for (int j = 0; j < 25; j++) {
            float m1 = ConvMask1[j * 32 + thIdy];
            acc += m1 * inputImg1[j][thIdx];
        }
        
        // layout transformation: 24*24*32 -> 32*48*12
        int oriOutPixX = (i * 32 + thIdx) % 24; // original position in output map
        int oriOutPixY = (i * 32 + thIdx) / 24;
        int newPixX = oriOutPixX / 2; // new position in output map
        int newPixY = oriOutPixY / 2 * 4 + oriOutPixY % 2 * 2 + oriOutPixX % 2;
        // new 1-d index in output map
        int outIndex = imgIndex * 32 * 48 * 12 + thIdy * 48 * 12 + newPixY * 12 + newPixX;
        
        device_a[outIndex] = acc;
        __syncthreads();
    }

    */
