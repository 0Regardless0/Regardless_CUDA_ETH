#include <cuda.h>
#include <curand_kernel.h>

#define ADDRESS_LENGTH 20
#define PRIVATE_KEY_LENGTH 32

// Custom comparison function for device code
__device__ int compareBytes(const unsigned char* a, const unsigned char* b, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

// Binary search on sorted address array
__device__ int binarySearch(const unsigned char* sortedData, int dataSize, const unsigned char* target)
{
    int left = 0;
    int right = dataSize - 1;
    while (left <= right)
    {
        int mid = left + (right - left) / 2;
        int cmp = compareBytes(sortedData + mid * ADDRESS_LENGTH, target, ADDRESS_LENGTH);
        if (cmp == 0) return mid;
        if (cmp < 0) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

extern "C" __global__ void generateAndCompare(
    unsigned char* addresses,
    unsigned char* privateKeys, // Added to store private keys
    unsigned char** sortedFiles, int* fileSizes,
    unsigned int* foundCount, unsigned char* foundAddresses,
    unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    curandState state;
    curand_init(seed + idx, 0, 0, &state);

    // Generate private key (32 bytes)
    unsigned char privateKey[PRIVATE_KEY_LENGTH];
    for (int i = 0; i < PRIVATE_KEY_LENGTH; i++)
        privateKey[i] = (unsigned char)(curand(&state) & 0xFF);

    // Store private key in output buffer
    int privOffset = idx * PRIVATE_KEY_LENGTH;
    memcpy(privateKeys + privOffset, privateKey, PRIVATE_KEY_LENGTH);

    // Dummy address derivation (for comparison only)
    unsigned char address[ADDRESS_LENGTH];
    for (int i = 0; i < ADDRESS_LENGTH; i++)
        address[i] = privateKey[i + 12]; // Simplified; host will recompute correctly

    // Copy to output buffer
    int addrOffset = idx * ADDRESS_LENGTH;
    memcpy(addresses + addrOffset, address, ADDRESS_LENGTH);

    // Compare with sorted files
    int fileIdx = address[0];
    if (fileSizes[fileIdx] > 0)
    {
        int found = binarySearch(sortedFiles[fileIdx], fileSizes[fileIdx], address);
        if (found >= 0)
        {
            unsigned int pos = atomicAdd(foundCount, 1);
            memcpy(foundAddresses + pos * ADDRESS_LENGTH, address, ADDRESS_LENGTH);
        }
    }
}