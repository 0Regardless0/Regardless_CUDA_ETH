using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Diagnostics; // For Stopwatch
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using Nethereum.Web3.Accounts; // For address derivation
using Nethereum.Hex.HexConvertors.Extensions; // For hex conversion

namespace EthereumAddressGenerator
{
    class Program
    {
        private static CudaContext cudaContext;
        private static CudaKernel generateAndCompareKernel;

        private const int ADDRESS_LENGTH = 20;
        private const int PRIVATE_KEY_LENGTH = 32;
        private const int FILES_COUNT = 256;
        private const int THREADS_PER_BLOCK = 256;
        private const int BLOCKS_PER_GRID = 1024;

        private static CudaDeviceVariable<byte> d_addresses;
        private static CudaDeviceVariable<byte> d_privateKeys;
        private static CudaDeviceVariable<byte>[] d_sortedFiles;
        private static CudaDeviceVariable<uint> d_foundCount;
        private static CudaDeviceVariable<byte> d_foundAddresses;

        static void Main(string[] args)
        {
            bool enableTesting = args.Length > 0 && args[0].ToLower() == "--test";
            Console.WriteLine($"Running {(enableTesting ? "with" : "without")} testing output");

            try
            {
                InitializeCuda();
                LoadSortedFiles();
                LoadCudaKernel();
                RunGenerationAndComparison(enableTesting);
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
            }
            finally
            {
                Cleanup();
            }
        }

        static void InitializeCuda()
        {
            cudaContext = new CudaContext(0);
            Console.WriteLine($"CUDA Initialized on {cudaContext.GetDeviceName()}");

            int totalThreads = BLOCKS_PER_GRID * THREADS_PER_BLOCK;
            d_addresses = new CudaDeviceVariable<byte>(totalThreads * ADDRESS_LENGTH);
            d_privateKeys = new CudaDeviceVariable<byte>(totalThreads * PRIVATE_KEY_LENGTH);
            d_sortedFiles = new CudaDeviceVariable<byte>[FILES_COUNT];
            d_foundCount = new CudaDeviceVariable<uint>(1);
            d_foundAddresses = new CudaDeviceVariable<byte>(totalThreads * ADDRESS_LENGTH);
        }

        static void LoadSortedFiles()
        {
            string basePath = @"C:\chunks";
            for (int i = 0; i < FILES_COUNT; i++)
            {
                string fileName = Path.Combine(basePath, $"{i:X2}.bin");
                if (!File.Exists(fileName))
                {
                    throw new FileNotFoundException($"File {fileName} not found.");
                }

                byte[] fileData = File.ReadAllBytes(fileName);
                d_sortedFiles[i] = new CudaDeviceVariable<byte>(fileData.Length);
                d_sortedFiles[i].CopyToDevice(fileData);
                Console.WriteLine($"Loaded {fileName} ({fileData.Length / ADDRESS_LENGTH} addresses)");
            }
        }

        static void LoadCudaKernel()
        {
            string ptxPath = "generateAndCompare.ptx";
            if (!File.Exists(ptxPath))
            {
                throw new FileNotFoundException($"PTX file {ptxPath} not found. Please compile generateAndCompare.cu with nvcc.");
            }

            byte[] ptxData = File.ReadAllBytes(ptxPath);
            generateAndCompareKernel = cudaContext.LoadKernelPTX(ptxData, "generateAndCompare");
            generateAndCompareKernel.BlockDimensions = new dim3(THREADS_PER_BLOCK, 1, 1);
            generateAndCompareKernel.GridDimensions = new dim3(BLOCKS_PER_GRID, 1, 1);
            Console.WriteLine("CUDA Kernel Loaded Successfully");
        }

        static void RunGenerationAndComparison(bool enableTesting)
        {
            int[] fileSizes = new int[FILES_COUNT];
            for (int i = 0; i < FILES_COUNT; i++)
                fileSizes[i] = (int)(d_sortedFiles[i].Size / ADDRESS_LENGTH);

            CudaDeviceVariable<int> d_fileSizes = new CudaDeviceVariable<int>(FILES_COUNT);
            d_fileSizes.CopyToDevice(fileSizes);

            CUdeviceptr[] filePointers = d_sortedFiles.Select(f => f.DevicePointer).ToArray();
            CudaDeviceVariable<CUdeviceptr> d_filePointers = new CudaDeviceVariable<CUdeviceptr>(FILES_COUNT);
            d_filePointers.CopyToDevice(filePointers);

            ulong seed = (ulong)DateTime.Now.Ticks;
            int iteration = 0;
            Stopwatch stopwatch = new Stopwatch(); // For timing

            while (true)
            {
                stopwatch.Restart(); // Start timing

                uint zero = 0;
                d_foundCount.CopyToDevice(zero);

                generateAndCompareKernel.Run(
                    d_addresses.DevicePointer,
                    d_privateKeys.DevicePointer,
                    d_filePointers.DevicePointer,
                    d_fileSizes.DevicePointer,
                    d_foundCount.DevicePointer,
                    d_foundAddresses.DevicePointer,
                    seed + (ulong)iteration
                );

                if (enableTesting)
                {
                    byte[] allAddresses = new byte[BLOCKS_PER_GRID * THREADS_PER_BLOCK * ADDRESS_LENGTH];
                    byte[] allPrivateKeys = new byte[BLOCKS_PER_GRID * THREADS_PER_BLOCK * PRIVATE_KEY_LENGTH];
                    d_addresses.CopyToHost(allAddresses);
                    d_privateKeys.CopyToHost(allPrivateKeys);

                    int displayLimit = Math.Min(5, BLOCKS_PER_GRID * THREADS_PER_BLOCK);
                    Console.WriteLine($"--- Testing Output for Iteration {iteration + 1} ---");
                    for (int i = 0; i < displayLimit; i++)
                    {
                        byte[] privKeyBytes = allPrivateKeys.Skip(i * PRIVATE_KEY_LENGTH).Take(PRIVATE_KEY_LENGTH).ToArray();
                        string privKeyHex = BitConverter.ToString(privKeyBytes).Replace("-", "").ToLower();

                        var account = new Account(privKeyHex);
                        string realAddrHex = account.Address.ToLower();

                        Console.WriteLine($"Private Key: {privKeyHex}");
                        Console.WriteLine($"Address: {realAddrHex}");
                        Console.WriteLine();
                    }
                }

                uint foundCount = d_foundCount[0];
                if (foundCount > 0)
                {
                    byte[] foundAddresses = new byte[foundCount * ADDRESS_LENGTH];
                    d_foundAddresses.CopyToHost(foundAddresses, 0, 0, foundCount * ADDRESS_LENGTH);

                    byte[] allPrivateKeys = new byte[BLOCKS_PER_GRID * THREADS_PER_BLOCK * PRIVATE_KEY_LENGTH];
                    d_privateKeys.CopyToHost(allPrivateKeys);

                    for (int i = 0; i < foundCount; i++)
                    {
                        byte[] addrBytes = foundAddresses.Skip(i * ADDRESS_LENGTH).Take(ADDRESS_LENGTH).ToArray();
                        string addrHex = "0x" + BitConverter.ToString(addrBytes).Replace("-", "").ToLower();

                        byte[] privKeyBytes = allPrivateKeys.Skip(i * PRIVATE_KEY_LENGTH).Take(PRIVATE_KEY_LENGTH).ToArray();
                        string privKeyHex = BitConverter.ToString(privKeyBytes).Replace("-", "").ToLower();
                        var account = new Account(privKeyHex);
                        string realAddrHex = account.Address.ToLower();

                        if (realAddrHex == addrHex)
                        {
                            Console.WriteLine($"Match found: Address={realAddrHex}, PrivateKey={privKeyHex}");
                            File.AppendAllText("found.txt", $"Address={realAddrHex}, PrivateKey={privKeyHex}\n");
                        }
                    }
                }

                //stopwatch.Stop(); // Stop timing
                //double elapsedSeconds = stopwatch.Elapsed.TotalSeconds;
                //double keysPerSecond = (BLOCKS_PER_GRID * THREADS_PER_BLOCK) / elapsedSeconds;
                //string speedFormatted = keysPerSecond >= 1000000 ? $"{keysPerSecond / 1000000:F2}M KPS" : $"{keysPerSecond / 1000:F2}K KPS";

                //iteration++;
                //Console.WriteLine($"Iteration {iteration}: Processed {BLOCKS_PER_GRID * THREADS_PER_BLOCK} addresses, Speed: {speedFormatted}");

                //if (iteration % 1000 == 0) seed = (ulong)DateTime.Now.Ticks;
                //Thread.Sleep(1);
            }
        }

        static void Cleanup()
        {
            if (d_addresses != null) d_addresses.Dispose();
            if (d_privateKeys != null) d_privateKeys.Dispose();
            if (d_sortedFiles != null)
                foreach (var file in d_sortedFiles) file?.Dispose();
            if (d_foundCount != null) d_foundCount.Dispose();
            if (d_foundAddresses != null) d_foundAddresses.Dispose();
            if (cudaContext != null) cudaContext.Dispose();
        }
    }
}