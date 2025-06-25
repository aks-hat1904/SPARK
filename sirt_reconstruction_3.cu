#include <cuda_runtime.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct ProjectionData {
    float* h_data;
    float* d_data;
    int angles, rows, cols;
    size_t total_elements, total_bytes;
};

struct ReconstructionVolume {
    float* h_volume;
    float* d_volume;
    int nx, ny, nz;
    float voxel_size;
    size_t total_elements, total_bytes;
};

struct GeometryParams {
    float source_origin_distance;    // 160 mm
    float origin_detector_distance;  // 40 mm
    float detector_pixel_size;       // 0.048 mm
};

// Element-wise array subtraction
__global__ void subtractArraysKernel(float* result, const float* a, const float* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        result[idx] = a[idx] - b[idx];
    }
}

// Volume update with non-negativity constraint
__global__ void updateVolumeKernel(float* volume, const float* correction, int n, float relaxation_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        volume[idx] += relaxation_factor * correction[idx];
        if (volume[idx] < 0.0f) volume[idx] = 0.0f; // Non-negativity constraint
    }
}

// 2. ADD VOXEL SENSITIVITY COMPUTATION KERNEL:
__global__ void computeVoxelSensitivityKernel(
    float* voxel_weights, const GeometryParams geom, 
    int nx, int ny, int nz, float voxel_size,
    int proj_rows, int proj_cols, float angle_rad) {
    
    int vox_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vox_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vox_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vox_x >= nx || vox_y >= ny || vox_z >= nz) return;
    
    // Same voxel world coordinate calculation as backProjectKernel
    float world_x = (vox_x - nx*0.5f + 0.5f) * voxel_size;
    float world_y = (vox_y - ny*0.5f + 0.5f) * voxel_size;
    float world_z = (vox_z - nz*0.5f + 0.5f) * voxel_size;
    
    float src_x = geom.source_origin_distance * cosf(angle_rad);
    float src_y = geom.source_origin_distance * sinf(angle_rad);
    float src_z = 0.0f;
    
    float ray_x = world_x - src_x;
    float ray_y = world_y - src_y;
    float ray_z = world_z - src_z;
    
    float ray_length = sqrtf(ray_x*ray_x + ray_y*ray_y + ray_z*ray_z);
    float ray_weight = 1.0f / (ray_length * ray_length); // Distance weighting
    
    // Same detector projection logic as backProjectKernel
    float det_normal_x = cosf(angle_rad);
    float det_normal_y = sinf(angle_rad);
    
    float det_center_x = -geom.origin_detector_distance * cosf(angle_rad);
    float det_center_y = -geom.origin_detector_distance * sinf(angle_rad);
    float det_center_z = 0.0f;
    
    float denom = ray_x * det_normal_x + ray_y * det_normal_y;
    if (fabsf(denom) < 1e-8f) return;
    
    float t = ((det_center_x - src_x) * det_normal_x + 
               (det_center_y - src_y) * det_normal_y) / denom;
    if (t <= 0) return;
    
    float intersect_x = src_x + t * ray_x;
    float intersect_y = src_y + t * ray_y;
    float intersect_z = src_z + t * ray_z;
    
    float det_u_vec_x = -sinf(angle_rad);
    float det_u_vec_y = cosf(angle_rad);
    
    float u = (intersect_x - det_center_x) * det_u_vec_x + 
              (intersect_y - det_center_y) * det_u_vec_y;
    float v = intersect_z - det_center_z;
    
    float det_col_f = u / geom.detector_pixel_size + proj_cols*0.5f - 0.5f;
    float det_row_f = v / geom.detector_pixel_size + proj_rows*0.5f - 0.5f;
    
    // Check if projection is within detector bounds
    if (det_col_f >= 0 && det_col_f < proj_cols-1 && 
        det_row_f >= 0 && det_row_f < proj_rows-1) {
        
        size_t vol_idx = (size_t)vox_z * nx * ny + (size_t)vox_y * nx + vox_x;
        atomicAdd(&voxel_weights[vol_idx], ray_weight);
    }
}

// 6. ADD NORMALIZATION KERNEL:
__global__ void normalizeVoxelCorrectionKernel(
    float* correction, const float* voxel_weights, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n && voxel_weights[idx] > 1e-8f) {
        correction[idx] /= voxel_weights[idx];
    }
}

// Trilinear interpolation
__device__ float trilinearInterpolation(
    const float* volume, int nx, int ny, int nz,
    float fx, float fy, float fz) {
    
    int x0 = floorf(fx), x1 = x0 + 1;
    int y0 = floorf(fy), y1 = y0 + 1;
    int z0 = floorf(fz), z1 = z0 + 1;
    
    if (x0 < 0 || x1 >= nx || y0 < 0 || y1 >= ny || z0 < 0 || z1 >= nz)
        return 0.0f;
    
    float xd = fx - x0, yd = fy - y0, zd = fz - z0;
    
    // Get 8 corner values
    size_t idx000 = (size_t)z0 * nx * ny + (size_t)y0 * nx + x0;
    size_t idx001 = idx000 + 1;
    size_t idx010 = idx000 + nx;
    size_t idx011 = idx010 + 1;
    size_t idx100 = idx000 + nx * ny;
    size_t idx101 = idx100 + 1;
    size_t idx110 = idx100 + nx;
    size_t idx111 = idx110 + 1;
    
    float v000 = volume[idx000], v001 = volume[idx001];
    float v010 = volume[idx010], v011 = volume[idx011];
    float v100 = volume[idx100], v101 = volume[idx101];
    float v110 = volume[idx110], v111 = volume[idx111];
    
    // Trilinear interpolation
    float v00 = v000 * (1 - xd) + v001 * xd;
    float v01 = v010 * (1 - xd) + v011 * xd;
    float v10 = v100 * (1 - xd) + v101 * xd;
    float v11 = v110 * (1 - xd) + v111 * xd;
    
    float v0 = v00 * (1 - yd) + v01 * yd;
    float v1 = v10 * (1 - yd) + v11 * yd;
    
    return v0 * (1 - zd) + v1 * zd;
}

// Ray-volume intersection
__device__ bool rayVolumeIntersection(
    float src_x, float src_y, float src_z,
    float ray_dx, float ray_dy, float ray_dz,
    int nx, int ny, int nz, float voxel_size,
    float* t_min, float* t_max) {
    
    float vol_min_x = -nx * voxel_size * 0.5f;
    float vol_max_x = nx * voxel_size * 0.5f;
    float vol_min_y = -ny * voxel_size * 0.5f;
    float vol_max_y = ny * voxel_size * 0.5f;
    float vol_min_z = -nz * voxel_size * 0.5f;
    float vol_max_z = nz * voxel_size * 0.5f;
    
    float t_min_x = (vol_min_x - src_x) / ray_dx;
    float t_max_x = (vol_max_x - src_x) / ray_dx;
    if (t_min_x > t_max_x) { float temp = t_min_x; t_min_x = t_max_x; t_max_x = temp; }
    
    float t_min_y = (vol_min_y - src_y) / ray_dy;
    float t_max_y = (vol_max_y - src_y) / ray_dy;
    if (t_min_y > t_max_y) { float temp = t_min_y; t_min_y = t_max_y; t_max_y = temp; }
    
    float t_min_z = (vol_min_z - src_z) / ray_dz;
    float t_max_z = (vol_max_z - src_z) / ray_dz;
    if (t_min_z > t_max_z) { float temp = t_min_z; t_min_z = t_max_z; t_max_z = temp; }
    
    *t_min = fmaxf(fmaxf(t_min_x, t_min_y), t_min_z);
    *t_max = fminf(fminf(t_max_x, t_max_y), t_max_z);
    
    return *t_min <= *t_max && *t_max > 0;
}

// 3. ADD RAY LENGTH COMPUTATION KERNEL:
__global__ void computeRayLengthsKernel(
    float* ray_lengths, const GeometryParams geom, 
    int nx, int ny, int nz, float voxel_size,
    int proj_rows, int proj_cols, float angle_rad) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= proj_cols || row >= proj_rows) return;
    
    // Same ray setup as forwardProjectKernel
    float det_u = (col - proj_cols * 0.5f + 0.5f) * geom.detector_pixel_size;
    float det_v = (row - proj_rows * 0.5f + 0.5f) * geom.detector_pixel_size;
    
    float src_x = geom.source_origin_distance * cosf(angle_rad);
    float src_y = geom.source_origin_distance * sinf(angle_rad);
    float src_z = 0.0f;
    
    float det_center_x = -geom.origin_detector_distance * cosf(angle_rad);
    float det_center_y = -geom.origin_detector_distance * sinf(angle_rad);
    float det_center_z = 0.0f;
    
    float det_u_vec_x = -sinf(angle_rad);
    float det_u_vec_y = cosf(angle_rad);
    float det_v_vec_z = 1.0f;
    
    float det_pos_x = det_center_x + det_u * det_u_vec_x;
    float det_pos_y = det_center_y + det_u * det_u_vec_y;
    float det_pos_z = det_center_z + det_v * det_v_vec_z;
    
    float ray_dx = det_pos_x - src_x;
    float ray_dy = det_pos_y - src_y;
    float ray_dz = det_pos_z - src_z;
    
    float ray_length = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
    ray_dx /= ray_length;
    ray_dy /= ray_length;
    ray_dz /= ray_length;
    
    float t_min, t_max;
    if (rayVolumeIntersection(src_x, src_y, src_z, ray_dx, ray_dy, ray_dz,
                              nx, ny, nz, voxel_size, &t_min, &t_max)) {
        t_min = fmaxf(t_min, 0.0f);
        ray_lengths[row * proj_cols + col] = t_max - t_min; // Ray segment length through volume
    } else {
        ray_lengths[row * proj_cols + col] = 0.0f;
    }
}

// Forward projection kernel
__global__ void forwardProjectKernel(
    const float* volume, float* projections,
    const GeometryParams geom, int nx, int ny, int nz, float voxel_size,
    int proj_rows, int proj_cols, float angle_rad) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col >= proj_cols || row >= proj_rows) return;
    
    // Detector pixel position
    float det_u = (col - proj_cols * 0.5f + 0.5f) * geom.detector_pixel_size;
    float det_v = (row - proj_rows * 0.5f + 0.5f) * geom.detector_pixel_size;
    
    // Source position
    float src_x = geom.source_origin_distance * cosf(angle_rad);
    float src_y = geom.source_origin_distance * sinf(angle_rad);
    float src_z = 0.0f;
    
    // Detector center
    float det_center_x = -geom.origin_detector_distance * cosf(angle_rad);
    float det_center_y = -geom.origin_detector_distance * sinf(angle_rad);
    float det_center_z = 0.0f;
    
    // Detector coordinate system
    float det_u_vec_x = -sinf(angle_rad);
    float det_u_vec_y = cosf(angle_rad);
    float det_v_vec_z = 1.0f;
    
    // Detector pixel world position
    float det_pos_x = det_center_x + det_u * det_u_vec_x;
    float det_pos_y = det_center_y + det_u * det_u_vec_y;
    float det_pos_z = det_center_z + det_v * det_v_vec_z;
    
    // Ray direction
    float ray_dx = det_pos_x - src_x;
    float ray_dy = det_pos_y - src_y;
    float ray_dz = det_pos_z - src_z;
    
    float ray_length = sqrtf(ray_dx*ray_dx + ray_dy*ray_dy + ray_dz*ray_dz);
    ray_dx /= ray_length;
    ray_dy /= ray_length;
    ray_dz /= ray_length;
    
    // Find volume intersection
    float t_min, t_max;
    if (!rayVolumeIntersection(src_x, src_y, src_z, ray_dx, ray_dy, ray_dz,
                              nx, ny, nz, voxel_size, &t_min, &t_max)) {
        projections[row * proj_cols + col] = 0.0f;
        return;
    }
    
    t_min = fmaxf(t_min, 0.0f);
    
    // Ray tracing with adaptive sampling
    float step_size = voxel_size * 0.25f;
    float ray_segment = t_max - t_min;
    int num_steps = (int)ceilf(ray_segment / step_size);
    
    float accumulated_value = 0.0f;
    if (num_steps > 0) {
        step_size = ray_segment / num_steps;
        
        for (int step = 0; step <= num_steps; step++) {
            float t = t_min + step * step_size;
            if (t > t_max) break;
            
            float pos_x = src_x + t * ray_dx;
            float pos_y = src_y + t * ray_dy;
            float pos_z = src_z + t * ray_dz;
            
            // Convert to voxel coordinates
            float vox_x = pos_x / voxel_size + nx * 0.5f;
            float vox_y = pos_y / voxel_size + ny * 0.5f;
            float vox_z = pos_z / voxel_size + nz * 0.5f;
            
            accumulated_value += trilinearInterpolation(volume, nx, ny, nz, vox_x, vox_y, vox_z);
        }
    }
    
    float ray_segment_length = t_max - t_min;
    projections[row * proj_cols + col] = accumulated_value * step_size;
}

// Back projection kernel
__global__ void backProjectKernel(
    float* volume, const float* projection_diff,
    const GeometryParams geom, int nx, int ny, int nz, float voxel_size,
    int proj_rows, int proj_cols, float angle_rad) {
    
    int vox_x = blockIdx.x * blockDim.x + threadIdx.x;
    int vox_y = blockIdx.y * blockDim.y + threadIdx.y;
    int vox_z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (vox_x >= nx || vox_y >= ny || vox_z >= nz) return;
    
    // Voxel world coordinates
    float world_x = (vox_x - nx*0.5f + 0.5f) * voxel_size;
    float world_y = (vox_y - ny*0.5f + 0.5f) * voxel_size;
    float world_z = (vox_z - nz*0.5f + 0.5f) * voxel_size;
    
    // Source position
    float src_x = geom.source_origin_distance * cosf(angle_rad);
    float src_y = geom.source_origin_distance * sinf(angle_rad);
    float src_z = 0.0f;
    
    // Ray from source to voxel
    float ray_x = world_x - src_x;
    float ray_y = world_y - src_y;
    float ray_z = world_z - src_z;
    
    // Distance weighting for cone beam
    float ray_length_sq = ray_x*ray_x + ray_y*ray_y + ray_z*ray_z;
    float distance_weight = 1.0f / ray_length_sq;

    // ADD THIS: Compute ray direction for intersection calculation
    float ray_length = sqrtf(ray_length_sq);
    float ray_dx = ray_x / ray_length;
    float ray_dy = ray_y / ray_length;
    float ray_dz = ray_z / ray_length;
    
    // ADD THIS: Compute ray-volume intersection to get segment length
    float t_min, t_max;
    if (!rayVolumeIntersection(src_x, src_y, src_z, ray_dx, ray_dy, ray_dz,
                              nx, ny, nz, voxel_size, &t_min, &t_max)) {
        return; // Ray doesn't intersect volume
    }
    
    t_min = fmaxf(t_min, 0.0f);
    float ray_segment_length = t_max - t_min;
    
    // If ray segment is too small, skip
    if (ray_segment_length < 1e-6f) return;
    
    // Detector plane normal
    float det_normal_x = cosf(angle_rad);
    float det_normal_y = sinf(angle_rad);
    
    // Detector center
    float det_center_x = -geom.origin_detector_distance * cosf(angle_rad);
    float det_center_y = -geom.origin_detector_distance * sinf(angle_rad);
    float det_center_z = 0.0f;
    
    // Ray-plane intersection
    float denom = ray_x * det_normal_x + ray_y * det_normal_y;
    if (fabsf(denom) < 1e-8f) return;
    
    float t = ((det_center_x - src_x) * det_normal_x + 
               (det_center_y - src_y) * det_normal_y) / denom;
    
    if (t <= 0) return;
    
    // Intersection point
    float intersect_x = src_x + t * ray_x;
    float intersect_y = src_y + t * ray_y;
    float intersect_z = src_z + t * ray_z;
    
    // Convert to detector coordinates
    float det_u_vec_x = -sinf(angle_rad);
    float det_u_vec_y = cosf(angle_rad);
    
    float u = (intersect_x - det_center_x) * det_u_vec_x + 
              (intersect_y - det_center_y) * det_u_vec_y;
    float v = intersect_z - det_center_z;
    
    // Convert to pixel coordinates
    float det_col_f = u / geom.detector_pixel_size + proj_cols*0.5f - 0.5f;
    float det_row_f = v / geom.detector_pixel_size + proj_rows*0.5f - 0.5f;
    
    // Bilinear interpolation
    int col0 = (int)floorf(det_col_f);
    int col1 = col0 + 1;
    int row0 = (int)floorf(det_row_f);
    int row1 = row0 + 1;
    
    if (col0 >= 0 && col1 < proj_cols && row0 >= 0 && row1 < proj_rows) {
        float fx = det_col_f - col0;
        float fy = det_row_f - row0;
        
        float w00 = (1.0f - fx) * (1.0f - fy);
        float w01 = fx * (1.0f - fy);
        float w10 = (1.0f - fx) * fy;
        float w11 = fx * fy;
        
        size_t idx00 = row0 * proj_cols + col0;
        size_t idx01 = row0 * proj_cols + col1;
        size_t idx10 = row1 * proj_cols + col0;
        size_t idx11 = row1 * proj_cols + col1;
        
        float interpolated_value = w00 * projection_diff[idx00] +
                                  w01 * projection_diff[idx01] +
                                  w10 * projection_diff[idx10] +
                                  w11 * projection_diff[idx11];
        
        //float ray_segment_length = t_max - t_min; 
        float contribution = interpolated_value * distance_weight;
        
        size_t vol_idx = (size_t)vox_z * nx * ny + (size_t)vox_y * nx + vox_x;
        atomicAdd(&volume[vol_idx], contribution);
    }
}

bool readProjectionMetadata(const std::string& metadata_file, int& angles, int& rows, int& cols) {
    std::ifstream file(metadata_file);
    if (!file.is_open()) return false;
    
    std::string line;
    if (std::getline(file, line)) {
        std::istringstream iss(line);
        iss >> angles >> rows >> cols;
        return !iss.fail() && iss.eof();  // Check if extraction succeeded and we consumed the entire line
    }
    return false;
}

bool readProjectionData(const std::string& binary_file, ProjectionData& proj_data) {
    if (!readProjectionMetadata("J:/raw/projections_shape.txt", 
                               proj_data.angles, proj_data.rows, proj_data.cols)) {
        return false;
    }
    
    proj_data.total_elements = (size_t)proj_data.angles * proj_data.rows * proj_data.cols;
    proj_data.total_bytes = proj_data.total_elements * sizeof(float);
    
    std::cout << "Reading projection data: " << proj_data.angles << "x" 
              << proj_data.rows << "x" << proj_data.cols << std::endl;
    
    proj_data.h_data = new float[proj_data.total_elements];
    if (!proj_data.h_data) return false;
    
    std::ifstream file(binary_file, std::ios::binary);
    if (!file.is_open()) {
        delete[] proj_data.h_data;
        return false;
    }
    
    file.read(reinterpret_cast<char*>(proj_data.h_data), proj_data.total_bytes);
    bool read_success = file.good();
    file.close();
    
    if (!read_success) {
        delete[] proj_data.h_data;
        return false;
    }
    
    // Allocate and copy to GPU
    if (cudaMalloc(&proj_data.d_data, proj_data.total_bytes) != cudaSuccess ||
        cudaMemcpy(proj_data.d_data, proj_data.h_data, proj_data.total_bytes, 
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        delete[] proj_data.h_data;
        if (proj_data.d_data) cudaFree(proj_data.d_data);
        return false;
    }
    
    return true;
}

bool initializeReconstructionVolume(ReconstructionVolume& vol, int nx, int ny, int nz, float voxel_size) {
    vol.nx = nx; vol.ny = ny; vol.nz = nz;
    vol.voxel_size = voxel_size;
    vol.total_elements = (size_t)nx * ny * nz;
    vol.total_bytes = vol.total_elements * sizeof(float);
    
    vol.h_volume = new float[vol.total_elements];
    if (!vol.h_volume) return false;
    
    for (size_t i = 0; i < vol.total_elements; i++) {
        vol.h_volume[i] = 0.1f; // Small positive initialization
    }

    
    if (cudaMalloc(&vol.d_volume, vol.total_bytes) != cudaSuccess ||
        cudaMemcpy(vol.d_volume, vol.h_volume, vol.total_bytes, 
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        delete[] vol.h_volume;
        if (vol.d_volume) cudaFree(vol.d_volume);
        return false;
    }
    
    std::cout << "Initialized volume: " << nx << "x" << ny << "x" << nz 
              << " (" << vol.total_bytes / (1024*1024) << " MB)" << std::endl;
    return true;
}

bool saveReconstructedVolume(const ReconstructionVolume& vol, const std::string& filename) {
    if (cudaMemcpy(vol.h_volume, vol.d_volume, vol.total_bytes, 
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        return false;
    }
    
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;
    
    file.write(reinterpret_cast<const char*>(vol.h_volume), vol.total_bytes);
    bool write_success = file.good();
    file.close();
    
    if (write_success) {
        // Save metadata
        std::string metadata_file = filename.substr(0, filename.find_last_of('.')) + "_shape.txt";
        std::ofstream meta_file(metadata_file);
        if (meta_file.is_open()) {
            meta_file << vol.nx << " " << vol.ny << " " << vol.nz << std::endl;
            meta_file.close();
        }
        std::cout << "Saved volume to: " << filename << std::endl;
    }
    
    return write_success;
}

void cleanup(ProjectionData& proj_data, ReconstructionVolume& vol) {
    if (proj_data.h_data) delete[] proj_data.h_data;
    if (proj_data.d_data) cudaFree(proj_data.d_data);
    if (vol.h_volume) delete[] vol.h_volume;
    if (vol.d_volume) cudaFree(vol.d_volume);
}

int main() {
    // Geometry setup
    GeometryParams geom = {160.0f, 40.0f, 0.048f};
    
    // Load projection data
    ProjectionData proj_data = {0};
    if (!readProjectionData("J:/raw/projections_float32.bin", proj_data)) {
        std::cerr << "Failed to read projection data" << std::endl;
        return -1;
    }
    
    // Initialize volume
    ReconstructionVolume volume;
    if (!initializeReconstructionVolume(volume, 800, 800, 600, 0.1f)){
        cleanup(proj_data, volume);
        return -1;
    }

    // Allocate temporary arrays
    size_t proj_slice_bytes = proj_data.rows * proj_data.cols * sizeof(float);
    float *d_forward_proj, *d_proj_diff, *d_correction;

    float *d_voxel_weights, *d_ray_lengths;
    if (cudaMalloc(&d_voxel_weights, volume.total_bytes) != cudaSuccess ||
        cudaMalloc(&d_ray_lengths, proj_slice_bytes) != cudaSuccess) {
        std::cerr << "Failed to allocate normalization memory" << std::endl;
        cleanup(proj_data, volume);
        return -1;
    }
    
    // SIRT parameters
    int max_iterations = 150;
    float relaxation_factor = 0.1f; // proj_data.angles Normalized relaxation
    
    if (cudaMalloc(&d_forward_proj, proj_slice_bytes) != cudaSuccess ||
        cudaMalloc(&d_proj_diff, proj_slice_bytes) != cudaSuccess ||
        cudaMalloc(&d_correction, volume.total_bytes) != cudaSuccess) {
        std::cerr << "Failed to allocate temporary memory" << std::endl;
        cleanup(proj_data, volume);
        return -1;
    }
    
    // Kernel launch parameters
    dim3 proj_block(16, 16);
    dim3 proj_grid((proj_data.cols + 15) / 16, (proj_data.rows + 15) / 16);
    
    dim3 vol_block(8, 8, 8);
    dim3 vol_grid((volume.nx + 7) / 8, (volume.ny + 7) / 8, (volume.nz + 7) / 8);
    
    dim3 linear_block(256);
    dim3 proj_linear_grid((proj_data.rows * proj_data.cols + 255) / 256);
    dim3 vol_linear_grid((volume.total_elements + 255) / 256);
    
    // BEFORE the main SIRT loop, compute voxel sensitivity:
    std::cout << "Computing voxel sensitivity maps..." << std::endl;
    cudaMemset(d_voxel_weights, 0, volume.total_bytes);

    for (int angle_idx = 0; angle_idx < proj_data.angles; angle_idx++) {
        float angle_rad = angle_idx * 2.0f * M_PI / proj_data.angles;
    
        computeVoxelSensitivityKernel<<<vol_grid, vol_block>>>(
            d_voxel_weights, geom, volume.nx, volume.ny, volume.nz, volume.voxel_size,
            proj_data.rows, proj_data.cols, angle_rad);
    }
    
    std::cout << "Starting SIRT reconstruction (" << max_iterations << " iterations)" << std::endl;
    
    // SIRT main loop
    for (int iter = 0; iter < max_iterations; iter++) {
        cudaMemset(d_correction, 0, volume.total_bytes);
        
        // Process each projection angle
        for (int angle_idx = 0; angle_idx < proj_data.angles; angle_idx++) {
            float angle_rad = angle_idx * 2.0f * M_PI / proj_data.angles;
            float* measured_proj = proj_data.d_data + angle_idx * proj_data.rows * proj_data.cols;
            
            // Forward project current volume
            forwardProjectKernel<<<proj_grid, proj_block>>>(
                volume.d_volume, d_forward_proj, geom,
                volume.nx, volume.ny, volume.nz, volume.voxel_size,
                proj_data.rows, proj_data.cols, angle_rad);
            
            // Compute difference: measured - forward_projected
            subtractArraysKernel<<<proj_linear_grid, linear_block>>>(
                d_proj_diff, measured_proj, d_forward_proj, proj_data.rows * proj_data.cols);
            
            // Back project the difference
            backProjectKernel<<<vol_grid, vol_block>>>(
                d_correction, d_proj_diff, geom,
                volume.nx, volume.ny, volume.nz, volume.voxel_size,
                proj_data.rows, proj_data.cols, angle_rad);
        }

        // ADD THIS: Normalize correction by voxel weights
        normalizeVoxelCorrectionKernel<<<vol_linear_grid, linear_block>>>(
        d_correction, d_voxel_weights, volume.total_elements);

        // Update volume with accumulated corrections
        updateVolumeKernel<<<vol_linear_grid, linear_block>>>(
            volume.d_volume, d_correction, volume.total_elements, relaxation_factor);
        
        if ((iter + 1) % 5 == 0) {
            std::cout << "Completed iteration " << (iter + 1) << std::endl;
        }
    }
    
    std::cout << "SIRT reconstruction completed" << std::endl;
    
    // Save result
    if (!saveReconstructedVolume(volume, "J:/raw/reconstructed_volume_float32.bin")) {
        std::cerr << "Failed to save volume" << std::endl;
    }
    
    // Cleanup
    cudaFree(d_forward_proj);
    cudaFree(d_proj_diff);
    cudaFree(d_correction);
    cudaFree(d_voxel_weights);
    cudaFree(d_ray_lengths);
    cleanup(proj_data, volume);
    
    return 0;
}