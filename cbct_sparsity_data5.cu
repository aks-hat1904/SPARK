#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <chrono>
#include <random>
#include <fstream>
#include <iostream>
#include <iomanip>

// CUDA compatibility macros
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Structure to hold geometry parameters
struct CTGeometry {
    int detector_rows;
    int detector_cols;
    int num_angles;
    int image_size_x;
    int image_size_y;
    int image_size_z;
    float source_to_origin;
    float origin_to_detector;
    float detector_spacing_h;
    float detector_spacing_v;
    float pixel_size;
};

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            return -1; \
        } \
    } while(0)

// Optimized CUDA kernel for Siddon's ray tracing algorithm
__global__ void siddon_ray_trace_kernel_optimized(
    const CTGeometry* __restrict__ geom,
    const float* __restrict__ angles,
    unsigned long long* __restrict__ ray_intersections,
    long long total_rays,
    long long ray_offset = 0  // For chunked processing
) {
    // Use 32-bit indexing for better performance when possible
    const long long ray_idx = (long long)blockIdx.x * blockDim.x + threadIdx.x + ray_offset;
    
    if (ray_idx >= total_rays + ray_offset) return;
    
    // Cache geometry values in registers
    const int detector_rows = geom->detector_rows;
    const int detector_cols = geom->detector_cols;
    const int num_angles = geom->num_angles;
    const float pixel_size = geom->pixel_size;
    const float source_to_origin = geom->source_to_origin;
    const float origin_to_detector = geom->origin_to_detector;
    const float detector_spacing_h = geom->detector_spacing_h;
    const float detector_spacing_v = geom->detector_spacing_v;
    
    // Calculate indices with optimized division
    const long long rays_per_angle = (long long)detector_rows * detector_cols;
    const int angle_idx = (int)(ray_idx / rays_per_angle);
    const long long detector_idx = ray_idx % rays_per_angle;
    const int detector_row = (int)(detector_idx / detector_cols);
    const int detector_col = (int)(detector_idx % detector_cols);
    
    // Early bounds checking
    if (angle_idx >= num_angles) {
        ray_intersections[ray_idx - ray_offset] = 0;
        return;
    }
    
    // Use fast math functions
    float angle = angles[angle_idx];
    float cos_angle, sin_angle;
    __sincosf(angle, &sin_angle, &cos_angle);  // Compute sin and cos simultaneously
    
    // Source position
    const float source_x = -source_to_origin * cos_angle;
    const float source_y = -source_to_origin * sin_angle;
    const float source_z = 0.0f;
    
    // Detector position calculations with fewer operations
    const float detector_center_x = origin_to_detector * cos_angle;
    const float detector_center_y = origin_to_detector * sin_angle;
    
    const float detector_u = (detector_col - detector_cols * 0.5f + 0.5f) * detector_spacing_h;
    const float detector_v = (detector_row - detector_rows * 0.5f + 0.5f) * detector_spacing_v;
    
    // Detector pixel world coordinates
    const float detector_x = detector_center_x - detector_u * sin_angle;
    const float detector_y = detector_center_y + detector_u * cos_angle;
    const float detector_z = detector_v;
    
    // Ray direction
    float ray_dx = detector_x - source_x;
    float ray_dy = detector_y - source_y;
    float ray_dz = detector_z - source_z;
    
    // Fast normalization using rsqrtf
    const float inv_ray_length = rsqrtf(ray_dx * ray_dx + ray_dy * ray_dy + ray_dz * ray_dz);
    if (!isfinite(inv_ray_length)) {
        ray_intersections[ray_idx - ray_offset] = 0;
        return;
    }
    
    ray_dx *= inv_ray_length;
    ray_dy *= inv_ray_length;
    ray_dz *= inv_ray_length;
    
    // Pre-compute volume bounds
    const float half_size_x = geom->image_size_x * pixel_size * 0.5f;
    const float half_size_y = geom->image_size_y * pixel_size * 0.5f;
    const float half_size_z = geom->image_size_z * pixel_size * 0.5f;
    
    const float min_x = -half_size_x;
    const float max_x = half_size_x;
    const float min_y = -half_size_y;
    const float max_y = half_size_y;
    const float min_z = -half_size_z;
    const float max_z = half_size_z;
    
    // Optimized epsilon handling
    const float epsilon = 1e-10f;
    const float large_val = 1e30f;
    
    // Calculate entry and exit parameters with branchless operations where possible
    float t_min_x, t_max_x;
    if (fabsf(ray_dx) < epsilon) {
        if (source_x < min_x || source_x > max_x) {
            ray_intersections[ray_idx - ray_offset] = 0;
            return;
        }
        t_min_x = -large_val;
        t_max_x = large_val;
    } else {
        const float inv_dx = 1.0f / ray_dx;
        t_min_x = (min_x - source_x) * inv_dx;
        t_max_x = (max_x - source_x) * inv_dx;
        if (t_min_x > t_max_x) {
            float temp = t_min_x; t_min_x = t_max_x; t_max_x = temp;
        }
    }
    
    float t_min_y, t_max_y;
    if (fabsf(ray_dy) < epsilon) {
        if (source_y < min_y || source_y > max_y) {
            ray_intersections[ray_idx - ray_offset] = 0;
            return;
        }
        t_min_y = -large_val;
        t_max_y = large_val;
    } else {
        const float inv_dy = 1.0f / ray_dy;
        t_min_y = (min_y - source_y) * inv_dy;
        t_max_y = (max_y - source_y) * inv_dy;
        if (t_min_y > t_max_y) {
            float temp = t_min_y; t_min_y = t_max_y; t_max_y = temp;
        }
    }
    
    float t_min_z, t_max_z;
    if (fabsf(ray_dz) < epsilon) {
        if (source_z < min_z || source_z > max_z) {
            ray_intersections[ray_idx - ray_offset] = 0;
            return;
        }
        t_min_z = -large_val;
        t_max_z = large_val;
    } else {
        const float inv_dz = 1.0f / ray_dz;
        t_min_z = (min_z - source_z) * inv_dz;
        t_max_z = (max_z - source_z) * inv_dz;
        if (t_min_z > t_max_z) {
            float temp = t_min_z; t_min_z = t_max_z; t_max_z = temp;
        }
    }
    
    const float t_min = fmaxf(fmaxf(t_min_x, t_min_y), fmaxf(t_min_z, 0.0f));
    const float t_max = fminf(fminf(t_max_x, t_max_y), t_max_z);
    
    if (t_min > t_max) {
        ray_intersections[ray_idx - ray_offset] = 0;
        return;
    }
    
    // Pre-compute stepping parameters
    const float inv_pixel_size = 1.0f / pixel_size;
    const float alpha_x = (fabsf(ray_dx) > epsilon) ? pixel_size / fabsf(ray_dx) : large_val;
    const float alpha_y = (fabsf(ray_dy) > epsilon) ? pixel_size / fabsf(ray_dy) : large_val;
    const float alpha_z = (fabsf(ray_dz) > epsilon) ? pixel_size / fabsf(ray_dz) : large_val;
    
    // Starting point
    const float start_x = source_x + t_min * ray_dx;
    const float start_y = source_y + t_min * ray_dy;
    const float start_z = source_z + t_min * ray_dz;
    
    // Calculate starting voxel indices with optimized floor
    int i = __float2int_rd((start_x - min_x) * inv_pixel_size);
    int j = __float2int_rd((start_y - min_y) * inv_pixel_size);
    int k = __float2int_rd((start_z - min_z) * inv_pixel_size);
    
    // Clamp to valid range
    i = max(0, min(i, geom->image_size_x - 1));
    j = max(0, min(j, geom->image_size_y - 1));
    k = max(0, min(k, geom->image_size_z - 1));
    
    // Direction increments
    const int i_inc = (ray_dx > 0) ? 1 : ((ray_dx < 0) ? -1 : 0);
    const int j_inc = (ray_dy > 0) ? 1 : ((ray_dy < 0) ? -1 : 0);
    const int k_inc = (ray_dz > 0) ? 1 : ((ray_dz < 0) ? -1 : 0);
    
    // Calculate next intersection parameters
    float alpha_x_next = (fabsf(ray_dx) > epsilon) ?
        ((ray_dx > 0) ? (min_x + (i + 1) * pixel_size - start_x) / ray_dx :
                       (min_x + i * pixel_size - start_x) / ray_dx) : large_val;
    
    float alpha_y_next = (fabsf(ray_dy) > epsilon) ?
        ((ray_dy > 0) ? (min_y + (j + 1) * pixel_size - start_y) / ray_dy :
                       (min_y + j * pixel_size - start_y) / ray_dy) : large_val;
    
    float alpha_z_next = (fabsf(ray_dz) > epsilon) ?
        ((ray_dz > 0) ? (min_z + (k + 1) * pixel_size - start_z) / ray_dz :
                       (min_z + k * pixel_size - start_z) / ray_dz) : large_val;
    
    const float alpha_max = t_max - t_min;
    
    // Traverse with optimized loop
    unsigned int intersections = 0;
    const int max_iterations = geom->image_size_x + geom->image_size_y + geom->image_size_z;
    
    #pragma unroll 4
    while (i >= 0 && i < geom->image_size_x && 
           j >= 0 && j < geom->image_size_y && 
           k >= 0 && k < geom->image_size_z &&
           intersections < max_iterations) {
        
        intersections++;
        
        // Branchless minimum finding and stepping
        bool x_is_min = (alpha_x_next <= alpha_y_next) && (alpha_x_next <= alpha_z_next);
        bool y_is_min = (alpha_y_next <= alpha_z_next) && !x_is_min;
        
        if (x_is_min) {
            if (alpha_x_next > alpha_max) break;
            i += i_inc;
            if (fabsf(ray_dx) > epsilon) alpha_x_next += alpha_x;
        } else if (y_is_min) {
            if (alpha_y_next > alpha_max) break;
            j += j_inc;
            if (fabsf(ray_dy) > epsilon) alpha_y_next += alpha_y;
        } else {
            if (alpha_z_next > alpha_max) break;
            k += k_inc;
            if (fabsf(ray_dz) > epsilon) alpha_z_next += alpha_z;
        }
    }
    
    ray_intersections[ray_idx - ray_offset] = (unsigned long long)intersections;
}

// Optimized reduction using CUB library for better performance
#include <cub/cub.cuh>

// Optimized host function with better memory management and streaming
double calculate_sparsity_optimized(const CTGeometry& geom, bool verbose = true) {
    // Parameter validation
    if (geom.detector_rows <= 0 || geom.detector_cols <= 0 || geom.num_angles <= 0 ||
        geom.image_size_x <= 0 || geom.image_size_y <= 0 || geom.image_size_z <= 0) {
        if (verbose) printf("Error: Invalid geometry parameters\n");
        return -1;
    }
    
    const long long total_rays = (long long)geom.detector_rows * geom.detector_cols * geom.num_angles;
    const long long total_voxels = (long long)geom.image_size_x * geom.image_size_y * geom.image_size_z;
    const long long matrix_size = total_rays * total_voxels;
    
    // Get device properties for optimal configuration
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    // Calculate optimal chunk size based on available memory
    size_t free_mem, total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    
    // Use 80% of free memory for ray intersections
    const size_t max_rays_per_chunk = (free_mem * 0.8) / sizeof(unsigned long long);
    const long long chunk_size = min(total_rays, (long long)max_rays_per_chunk);
    
    if (verbose) {
        printf("GPU: %s (SM %d.%d, %d MPs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
        printf("Matrix: %lld x %lld = %lld elements\n", total_rays, total_voxels, matrix_size);
        printf("Chunk size: %lld rays (%.1f%% of total)\n", chunk_size, 100.0 * chunk_size / total_rays);
    }
    
    // Pre-compute angles
    std::vector<float> angles(geom.num_angles);
    for (int i = 0; i < geom.num_angles; i++) {
        angles[i] = 2.0f * M_PI * i / geom.num_angles;
    }
    
    // Allocate device memory with error checking
    CTGeometry* d_geom;
    float* d_angles;
    unsigned long long* d_ray_intersections;
    
    CUDA_CHECK(cudaMalloc(&d_geom, sizeof(CTGeometry)));
    CUDA_CHECK(cudaMalloc(&d_angles, geom.num_angles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ray_intersections, chunk_size * sizeof(unsigned long long)));
    
    // Copy constant data to device
    CUDA_CHECK(cudaMemcpy(d_geom, &geom, sizeof(CTGeometry), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_angles, angles.data(), geom.num_angles * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create CUDA streams for overlapping computation
    const int num_streams = 2;
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    unsigned long long total_intersections = 0;
    
    // Optimal block size based on occupancy
    int block_size = 0;
    int min_grid_size = 0;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, siddon_ray_trace_kernel_optimized, 0, 0);
    
    // Round to nearest power of 2 for better performance
    block_size = 1 << (int)log2f((float)block_size);
    block_size = min(block_size, 512); // Cap at 512 for better register usage
    
    if (verbose) {
        printf("Using block size: %d threads\n", block_size);
    }
    
    // Process rays in chunks with streaming
    for (long long processed_rays = 0; processed_rays < total_rays; processed_rays += chunk_size) {
        const long long current_chunk_size = min(chunk_size, total_rays - processed_rays);
        const int stream_idx = (processed_rays / chunk_size) % num_streams;
        
        // Launch kernel with optimal configuration
        const int grid_size = (int)((current_chunk_size + block_size - 1) / block_size);
        
        siddon_ray_trace_kernel_optimized<<<grid_size, block_size, 0, streams[stream_idx]>>>(
            d_geom, d_angles, d_ray_intersections, total_rays, processed_rays
        );
        
        // Use CUB for efficient reduction
        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;
        unsigned long long* d_sum;
        cudaMalloc(&d_sum, sizeof(unsigned long long));
        
        // Determine temporary device storage requirements
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_ray_intersections, d_sum, current_chunk_size, streams[stream_idx]);
        
        // Allocate temporary storage
        cudaMalloc(&d_temp_storage, temp_storage_bytes);
        
        // Run sum-reduction
        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_ray_intersections, d_sum, current_chunk_size, streams[stream_idx]);
        
        // Copy result back
        unsigned long long chunk_sum;
        cudaMemcpyAsync(&chunk_sum, d_sum, sizeof(unsigned long long), cudaMemcpyDeviceToHost, streams[stream_idx]);
        
        // Synchronize stream before using result
        cudaStreamSynchronize(streams[stream_idx]);
        
        total_intersections += chunk_sum;
        
        // Cleanup temporary memory
        cudaFree(d_temp_storage);
        cudaFree(d_sum);
        
        if (verbose && ((processed_rays + current_chunk_size) % (total_rays / 10) == 0 || processed_rays + current_chunk_size == total_rays)) {
            printf("Progress: %lld / %lld rays (%.1f%%)\n", 
                   processed_rays + current_chunk_size, total_rays, 
                   100.0 * (processed_rays + current_chunk_size) / total_rays);
        }
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (verbose) {
        printf("Computation completed in %ld ms\n", duration.count());
        printf("Total non-zero elements: %llu\n", total_intersections);
        printf("Performance: %.2f Mrays/sec\n", (double)total_rays / duration.count() * 1000.0 / 1e6);
    }
    
    // Calculate sparsity
    const double sparsity = 1.0 - (double)total_intersections / (double)matrix_size;
    
    // Cleanup
    for (int i = 0; i < num_streams; i++) {
        cudaStreamDestroy(streams[i]);
    }
    cudaFree(d_geom);
    cudaFree(d_angles);
    cudaFree(d_ray_intersections);
    
    return sparsity;
}

// Additional optimization: Precomputed lookup tables for common angles
class OptimizedRayTracer {
private:
    float* d_sin_table;
    float* d_cos_table;
    int num_angles;
    
public:
    OptimizedRayTracer(int angles) : num_angles(angles) {
        // Precompute sin/cos lookup tables
        std::vector<float> sin_table(angles), cos_table(angles);
        for (int i = 0; i < angles; i++) {
            float angle = 2.0f * M_PI * i / angles;
            sin_table[i] = sinf(angle);
            cos_table[i] = cosf(angle);
        }
        
        cudaMalloc(&d_sin_table, angles * sizeof(float));
        cudaMalloc(&d_cos_table, angles * sizeof(float));
        cudaMemcpy(d_sin_table, sin_table.data(), angles * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cos_table, cos_table.data(), angles * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ~OptimizedRayTracer() {
        cudaFree(d_sin_table);
        cudaFree(d_cos_table);
    }
    
    // Use precomputed tables in kernel (modify kernel to accept these as parameters)
    double calculateSparsity(const CTGeometry& geom, bool verbose = true) {
        // Implementation using precomputed tables...
        return calculate_sparsity_optimized(geom, verbose);
    }
};

// Function to generate random but physically realistic CT geometry parameters
CTGeometry generate_random_geometry(std::mt19937& rng) {
    CTGeometry geom;
    
    // Define more conservative ranges for CT parameters to avoid memory issues
    std::uniform_int_distribution<int> detector_size_dist(512, 1024);    // Smaller detector sizes
    std::uniform_int_distribution<int> num_angles_dist(360, 2400);      // Fewer angles
    std::uniform_int_distribution<int> image_size_dist(128, 512);      // Smaller image sizes
    std::uniform_real_distribution<float> source_dist(10.0f, 150.0f);  // mm
    std::uniform_real_distribution<float> detector_dist(500.0f, 1000.0f);  // mm
    std::uniform_real_distribution<float> spacing_dist(0.1f, 0.5f);       // mm
    std::uniform_real_distribution<float> pixel_dist(0.1f, 0.5f);         // mm
    
    // Generate detector parameters
    geom.detector_rows = detector_size_dist(rng);
    geom.detector_cols = detector_size_dist(rng);
    geom.num_angles = num_angles_dist(rng);
    
    // Generate image parameters
    geom.image_size_x = image_size_dist(rng);
    geom.image_size_y = image_size_dist(rng);
    geom.image_size_z = image_size_dist(rng);
    
    // Generate geometry parameters
    geom.source_to_origin = source_dist(rng);
    geom.origin_to_detector = detector_dist(rng);
    geom.detector_spacing_h = spacing_dist(rng);
    geom.detector_spacing_v = spacing_dist(rng);
    geom.pixel_size = pixel_dist(rng);
    
    return geom;
}

// Function to generate dataset
void generate_dataset(int num_samples, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        printf("Error: Could not open file %s for writing\n", filename.c_str());
        return;
    }
    
    // Write CSV header
    file << "detector_rows,detector_cols,num_angles,image_size_x,image_size_y,image_size_z,";
    file << "source_to_origin,origin_to_detector,detector_spacing_h,detector_spacing_v,pixel_size,sparsity\n";
    
    // Initialize random number generator
    std::random_device rd;
    std::mt19937 rng(rd());
    
    printf("Generating dataset with %d samples...\n", num_samples);
    printf("Progress: ");
    
    int successful_samples = 0;
    for (int i = 0; i < num_samples && successful_samples < num_samples; i++) {
        // Generate random geometry
        CTGeometry geom = generate_random_geometry(rng);
        
        // Calculate sparsity (non-verbose mode for dataset generation)
        double sparsity = calculate_sparsity_optimized(geom, false);
        
        if (sparsity < 0) {
            printf("Error calculating sparsity for sample %d, retrying...\n", i + 1);
            continue;
        }
        
        // Write to CSV file
        file << geom.detector_rows << "," << geom.detector_cols << "," << geom.num_angles << ",";
        file << geom.image_size_x << "," << geom.image_size_y << "," << geom.image_size_z << ",";
        file << geom.source_to_origin << "," << geom.origin_to_detector << ",";
        file << geom.detector_spacing_h << "," << geom.detector_spacing_v << ",";
        file << geom.pixel_size << "," << std::fixed << std::setprecision(8) << sparsity << "\n";
        
        successful_samples++;
        
        // Progress indicator
        if (successful_samples % (num_samples / 10) == 0 || successful_samples == num_samples) {
            printf("%.0f%% ", 100.0 * successful_samples / num_samples);
            fflush(stdout);
        }
    }
    
    printf("\nDataset generation completed!\n");
    printf("Successfully generated %d samples\n", successful_samples);
    printf("Results saved to: %s\n", filename.c_str());
    file.close();
}

int main() {
    // Check CUDA device
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        printf("No CUDA devices found!\n");
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using GPU: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("Global memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Check if compute capability is supported
    if (prop.major < 3) {
        printf("Warning: GPU compute capability %d.%d may not be fully supported\n", prop.major, prop.minor);
        printf("Minimum recommended compute capability is 3.0\n");
    }
    printf("\n");
    
    // Ask user for choice
    int choice;
    printf("Choose an option:\n");
    printf("1. Run single example calculation\n");
    printf("2. Generate dataset\n");
    printf("Enter your choice (1 or 2): ");
    std::cin >> choice;
    
    if (choice == 1) {
        // Original example calculation with smaller, safer parameters
        CTGeometry geom;
        geom.detector_rows = 64;
        geom.detector_cols = 64;
        geom.num_angles = 180;
        geom.image_size_x = 64;
        geom.image_size_y = 64;
        geom.image_size_z = 64;
        geom.source_to_origin = 570.0f;  // mm
        geom.origin_to_detector = 430.0f; // mm
        geom.detector_spacing_h = 1.0f;   // mm
        geom.detector_spacing_v = 1.0f;   // mm
        geom.pixel_size = 1.0f;           // mm
        
        printf("=== CT Geometry Parameters ===\n");
        printf("Detector: %d x %d pixels\n", geom.detector_rows, geom.detector_cols);
        printf("Number of angles: %d\n", geom.num_angles);
        printf("Image size: %d x %d x %d voxels\n", 
               geom.image_size_x, geom.image_size_y, geom.image_size_z);
        printf("Source to origin: %.1f mm\n", geom.source_to_origin);
        printf("Origin to detector: %.1f mm\n", geom.origin_to_detector);
        printf("Detector spacing: %.1f x %.1f mm\n", 
               geom.detector_spacing_h, geom.detector_spacing_v);
        printf("Pixel size: %.1f mm\n", geom.pixel_size);
        printf("\n");
        
        // Calculate sparsity
        double sparsity = calculate_sparsity_optimized(geom);
        
        if (sparsity >= 0) {
            printf("\n=== Results ===\n");
            printf("System matrix sparsity: %.6f (%.2f%%)\n", sparsity, sparsity * 100.0);
            printf("Density: %.6f (%.2f%%)\n", 1.0 - sparsity, (1.0 - sparsity) * 100.0);
        }
    }
    else if (choice == 2) {
        // Dataset generation
        int num_samples;
        printf("Enter the number of data points to generate: ");
        std::cin >> num_samples;
        
        if (num_samples <= 0) {
            printf("Invalid number of samples. Please enter a positive integer.\n");
            return -1;
        }
        
        std::string filename = "cbct_sparsity_dataset.csv";
        printf("Generating dataset with %d samples...\n", num_samples);
        printf("Output file: %s\n\n", filename.c_str());
        
        auto start_time = std::chrono::high_resolution_clock::now();
        generate_dataset(num_samples, filename);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        printf("Total time: %ld seconds\n", duration.count());
    }
    else {
        printf("Invalid choice. Please run the program again and select 1 or 2.\n");
        return -1;
    }
    
    return 0;
}