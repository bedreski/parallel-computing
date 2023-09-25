
//NVDIA GPU Gems 3: https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda
//Cuda Toolkit Documentation: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#introduction

/*Modificações pra resolver o Shared Memory Bank Conflict (SMBC)
O padrão de acesso à memória faz com que o SMBC aconteça. 
A memória compartilhada tratada nesse tipo de algoritmo é composta de vários "bancos"
Se várias threads no mesmo warp acessam o mesmo banco, há conflito. 
Conflitos desse tipo geram serialização dos múltiplos acessos ao banco de memória. 
Isso significa que um acesso à memória compartilhada com um conflito de banco de memória
de grau 'n', precisa de 'n' vezes mais ciclos para processar, comparado a um acesso sem conflito.

Conflitos de banco de memória são evitáveis aqui ao ter cuidado com o acesso à memória de vetores
__shared__ 
É adicionado ao índice o valor do índice dividido pelo número de bancos de memória compartilhada. 
*/
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \     ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

__global__ void prescan(float *g_odata, float *g_idata, int n) { 

    extern __shared__ float temp[];  // allocated on invocation 
    int thid = threadIdx.x; 
    int offset = 1;
    //Bloco A 
    //temp[2*thid] = g_idata[2*thid]; // load input into shared memory 
    //temp[2*thid+1] = g_idata[2*thid+1]; 
    //---- Resolver SMBC -----
    int ai = thid;
    int bi = thid + (n/2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    temp[ai + bankOffsetA] = g_idata[ai];
    temp[bi + bankOffsetB] = g_idata[bi];
    //---------- 

    for (int d = n>>1; d > 0; d >>= 1)   // build sum in place up the tree 
    { 
        __syncthreads();

        if (thid < d)    
        { 
            //Bloco B 
            int ai = offset*(2*thid+1)-1;     
            int bi = offset*(2*thid+2)-1;
            ai += ai / NUM_BANKS; 
            bi += bi / NUM_BANKS; 
            temp[bi] += temp[ai];    
        }
        offset *= 2; 
    } 

    //Bloco C 
    if (thid == 0) 
    { 
        //temp[n - 1] = 0; 
        //---- Resolver SMBC ----
        temp[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;
        //-----------

    } // clear the last element 

    for (int d = 1; d < n; d *= 2) // traverse down tree & build scan 
    {
        offset >>= 1;
        __syncthreads();
        if (thid < d)      
        {
            //Bloco D 
            int ai = offset*(2*thid+1)-1;
            int bi = offset*(2*thid+2)-1;
            float t = temp[ai];
            temp[ai] = temp[bi];
            temp[bi] += t;
        } 
    }

    __syncthreads();

    //Bloco E 
    //g_odata[2*thid] = temp[2*thid]; // write results to device memory
    //g_odata[2*thid+1] = temp[2*thid+1];

    //---- Resolver SMBC ----
    g_odata[ai] = temp[ai + bankOffsetA];
    g_odata[bi] = temp[bi + bankOffsetB];
    //---------
}

