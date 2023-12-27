/*Assignment 2 CPT316 2023/2024
  Group members:
  1. MOHAMAD NAZMI BIN HASHIM
  2. MIOR MUHAMMAD IRFAN BIN MIOR LATFEE
  3. OOI JING MIN
  4. MUNIRAH BINTI SHAMSUDIN
  5. SHONEERA SIMIN */

/*This code demonstrates how the linear seacrh algorithm works in C programming.
 It shows the step by step process of searching for an element efficiently using 
 CUDA for parallel processing */

 /* In order to run program, first open terminal and enter "nvcc -o LinearCUDA LinearCUDA.cu" and wait untill program build succesfully 
  then enter "./LinearCUDA_executable" untill appear the output*/


#include <stdio.h>
#include <cuda_runtime.h>

//Define the size of the array
const int N = 1000;  

//Define the number of threads per block
const int threadsPerBlock = 256;

//CUDA kernel for linear search
__global__ void linearSearch(int *array, int size, int target, int *result, int *comparisons) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x; //Calculate the global thread ID


//Shared memory for counting comparisons within a block
    __shared__ int comparisonsInBlock[threadsPerBlock];
    comparisonsInBlock[threadIdx.x] = 0;
    __syncthreads();

// Go through the array, each thread covering a chunk.
    for (int i = tid; i < size; i += blockDim.x * gridDim.x) {
        comparisonsInBlock[threadIdx.x]++;


//Check if the current element is the target
        if (array[i] == target) {
            atomicMin(result, i); //update the result with the minimum index
            break;
        }
    }

   // Synchronize threads within the block
    __syncthreads();

    //Reduce the comparisons within the block using parallel reduction
        for (int i = blockDim.x / 2 ; i > 0; i >>= 1) {
            if (threadIdx.x < i){
                comparisonsInBlock[threadIdx.x] += comparisonsInBlock[threadIdx.x +i];
            }
            __syncthreads();
        }
        
        //Store the total comparisons made by the block in the global comparisons array
        if (threadIdx.x == 0){
            atomicAdd(comparisons, comparisonsInBlock[0]);
            }
        }
        
    


int main() {
    int array[N]; //Host array

    // Initializing the array with provided values
    int values[] = {402,279,570,263,1,327,368,106,979,43,755,197,316,935,880,609,638,136,304,874,
				706,646,386,920,207,916,332,790,340,201,998,4,421,514,967,381,487,955,35,972,
				665,618,251,49,217,339,977,624,42,680,437,116,250,756,894,651,56,86,996,578,
				767,778,46,751,896,422,913,229,151,181,269,902,41,29,871,990,577,355,167,298,
				522,561,889,54,804,758,74,730,431,870,39,394,899,841,206,299,324,444,556,711,
				814,529,256,847,348,728,759,781,66,509,642,412,469,485,863,840,92,500,721,
				312,231,142,189,930,573,667,133,799,457,993,305,707,227,30,315,950,365,361,
				398,983,585,436,484,352,892,809,945,617,539,829,209,400,376,843,956,984,180,
				821,134,866,635,525,550,356,951,833,526,712,604,210,137,183,908,586,171,185,
				426,867,792,492,741,966,985,555,27,447,322,541,615,351,258,347,765,454,995,
				784,314,724,620,72,218,595,936,371,79,820,396,268,143,810,391,278,265,283,
				669,295,650,239,407,731,366,553,68,557,670,748,390,657,634,244,722,220,84,
				413,115,308,64,506,815,460,653,952,385,301,286,132,455,715,7,587,173,141,
				643,576,659,632,948,303,750,882,536,932,452,668,338,739,904,120,364,90,335,
				466,369,528,816,453,848,248,636,419,117,619,976,399,684,257,21,280,318,405,
				681,838,476,8,796,637,247,885,289,912,872,672,325,271,393,9,845,879,795,
				524,655,424,472,411,255,943,743,320,433,233,331,395,346,149,552,740,927,530,
				991,988,468,164,918,842,113,357,818,588,462,473,729,921,517,184,450,534,590,
				152,662,290,545,420,580,478,328,567,98,262,687,87,898,73,144,613,474,805,
				78,714,384,516,766,981,451,67,273,794,254,434,456,603,130,873,776,959,512,
				886,589,108,946,464,104,282,812,647,195,965,645,834,639,709,688,10,718,949,
				608,238,5,701,95,862,960,612,97,856,397,193,440,568,232,423,746,25,496,45,
				807,849,491,785,236,353,549,131,978,240,44,159,14,274,924,367,439,674,406,
				372,836,700,135,508,11,883,264,467,121,441,380,676,275,914,876,868,139,337,
				859,215,270,678,489,222,532,970,621,28,827,228,782,24,861,546,890,683,170,
				122,363,962,582,165,813,600,999,786,693,465,602,58,291,533,691,100,162,689,
				774,415,177,26,75,853,775,80,176,656,544,831,572,99,633,837,761,939,33,563,
				994,644,666,203,844,266,986,370,211,18,675,202,374,717,293,190,915,658,903,
				281,566,851,392,521,225,738,565,900,277,234,76,129,138,708,235,543,569,850,
				923,187,93,910,3,47,341,243,610,542,52,245,22,793,562,737,212,81,309,128,
				798,777,188,389,118,628,435,663,779,772,819,146,160,686,857,249,166,446,
				974,968,221,768,208,287,887,350,285,77,591,826,172,192,695,622,699,940,300,
				537,260,625,145,571,288,640,559,954,60,345,442,548,551,501,224,519,623,443,
				200,62,771,302,223,330,497,362,428,418,458,616,574,791,199,448,416,919,607,
				179,127,888,958,753,38,878,219,518,377,584,124,971,163,980,174,579,875,147,
				306,803,931,881,832,901,156,928,581,82,83,808,311,96,493,50,723,705,410,69,
				704,463,2,679,749,957,61,858,59,70,716,213,150,403,449,745,109,860,272,252,
				358,754,155,158,702,854,55,488,592,975,114,911,720,822,802,835,246,153,375,
				694,925,15,763,725,824,502,823,953,787,760,196,432,438,593,477,71,230,32,
				547,430,893,126,520,140,558,597,641,94,963,191,654,614,409,909,855,483,696,
				205,757,404,169,764,649,503,905,735,780,682,214,294,846,112,261,175,742,934,
				505,388,329,554,538,540,154,296,63,611,1000,527,482,354,869,897,719,583,504,
				673,157,732,349,697,360,627,510,864,471,490,85,333,997,964,630,23,652,111,
				237,733,865,16,989,703,101,20,40,182,982,34,713,877,664,373,297,57,783,789,
				507,51,86,461,495,383,830,961,941,321,479,747,427,414,408,926,241,198,148,
				475,929,91,53,685,31,922,769,313,907,65,267,216,852,969,629,727,942,884,459,
				378,103,336,736,6,307,598,379,253,226,161,359,677,292,973,828,342,105,531,
				564,734,710,276,594,692,125,168,906,752,19,601,494,401,797,762,343,891,417,
				744,107,13,800,12,310,671,344,204,425,36,596,470,334,839,992,600,631,242,
				660,513,319,599,917,773,606,194,937,110,481,511,817,575,382,811,523,726,933,
				499,690,284,801,535,788,648,445,178,123,17,825,661,626,88,37,486,102,770,
				947,323,515,317,806,326,944,987,895,119,480,259,387,48,938,698,560,498,429,89};

    for (int i = 0; i < N; ++i) {
        array[i] = values[i];
    }

   //Device arrays for CUDA
    int *dev_array;
    int *dev_result;
    int *dev_comparisons;

    //Allocate memory on the GPU
    cudaMalloc((void**)&dev_array, N * sizeof(int));
    cudaMalloc((void**)&dev_result, sizeof(int));
    cudaMalloc((void**)&dev_comparisons, sizeof(int));

    // Copy array from host to device
    cudaMemcpy(dev_array, array, N * sizeof(int), cudaMemcpyHostToDevice);

    // Maximum number of targets the user can input
    const int maxTargets = 5;

    // Variables for user input
    int numTargets;
    int targets[maxTargets];

    
    printf("Linear Search Implementation in CUDA\n");
    printf("====================================\n\n");

    //Prompt user the number of elements to compare up to maxTargets
    printf("Enter the number of element to compare(up to %d): ", maxTargets);
    scanf("%d", &numTargets);

    // Check if the input is valid
    if (numTargets <= 0 || numTargets > maxTargets) {
        printf("Invalid number of element.\n");
        return 1;
    }

    // Prompt the user for each target value
    for (int i = 0; i < numTargets; ++i) {
        printf("Enter the number %d: ", i + 1);
        scanf("%d", &targets[i]);
    }
printf("\n\n");

    // Loop over each target
    for (int targetIndex = 0; targetIndex < numTargets; ++targetIndex) {
        int target = targets[targetIndex];

        // Set initial result to a value larger than any possible index
        cudaMemcpy(dev_result, &N, sizeof(int), cudaMemcpyHostToDevice);

        // Set initial comparisons count to 0
        int comparisons = 0;
        cudaMemcpy(dev_comparisons, &comparisons, sizeof(int), cudaMemcpyHostToDevice);

        // Create CUDA events for timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        // Record start time
        cudaEventRecord(start);

        // Launch kernel with dynamic parallelism
        int gridSize = (N + threadsPerBlock - 1) / threadsPerBlock;
        linearSearch<<<gridSize, threadsPerBlock>>>(dev_array, N, target, dev_result, dev_comparisons);

        // Record end time
        cudaEventRecord(stop);

        // Copy back the result and comparisons count
        int result;
        cudaMemcpy(&result, dev_result, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&comparisons, dev_comparisons, sizeof(int), cudaMemcpyDeviceToHost);

        // Ensure that all CUDA threads have completed
        cudaDeviceSynchronize();

        // Check for CUDA errors
        cudaError_t cudaError =cudaGetLastError();
        if(cudaError != cudaSuccess){
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(cudaError));
            return 1;
        }


        // Print result for each target
        if (result < N && result >= 0) {
            printf("Element number %d found at index %d\n", target, result);
        } else {
            printf("The Target's Element %d is not found.\n", target);
        }

        // Calculate and print execution time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        printf("Execution Time: %f ms\n", milliseconds);

        printf("Number of Comparisons made to find the the key value: %d\n", comparisons);
        printf("\n");

        // Destroy CUDA events
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    // Free allocated memory on the GPU
    cudaFree(dev_array);
    cudaFree(dev_result);
    cudaFree(dev_comparisons);

    return 0;
}