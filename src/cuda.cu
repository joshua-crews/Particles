#include "cuda.cuh"
#include "helper.h"

#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

#include <thrust/device_vector.h>
#include <thrust/scan.h>

// #define VALIDATION 1

///
/// Algorithm storage
///
// Number of particles in d_particles
unsigned int cuda_particles_count;
// Device pointer to a list of particles
Particle* d_particles;
// Device pointer to a histogram of the number of particles contributing to each pixel
unsigned int* d_pixel_contribs;
// Device pointer to an index of unique offsets for each pixels contributing colours
unsigned int* d_pixel_index;
// Device pointer to storage for each pixels contributing colours
unsigned char* d_pixel_contrib_colours;
// Device pointer to storage for each pixels contributing colours' depth
float* d_pixel_contrib_depth;
// The number of contributors d_pixel_contrib_colours and d_pixel_contrib_depth have been allocated for
unsigned int cuda_pixel_contrib_count;
// Host storage of the output image dimensions
int cuda_output_image_width;
int cuda_output_image_height;
// Device storage of the output image dimensions
__constant__ int D_OUTPUT_IMAGE_WIDTH;
__constant__ int D_OUTPUT_IMAGE_HEIGHT;
unsigned char* d_output_image_data;

void cuda_begin(const Particle* init_particles, const unsigned int init_particles_count,
    const unsigned int out_image_width, const unsigned int out_image_height) {
    cuda_particles_count = init_particles_count;
    CUDA_CALL(cudaMalloc(&d_particles, init_particles_count * sizeof(Particle)));
    CUDA_CALL(cudaMemcpy(d_particles, init_particles, init_particles_count * sizeof(Particle), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMalloc(&d_pixel_contribs, out_image_width * out_image_height * sizeof(unsigned int)));
    CUDA_CALL(cudaMalloc(&d_pixel_index, (out_image_width * out_image_height + 1) * sizeof(unsigned int)));
    d_pixel_contrib_colours = 0;
    d_pixel_contrib_depth = 0;
    cuda_pixel_contrib_count = 0;

    cuda_output_image_width = (int)out_image_width;
    cuda_output_image_height = (int)out_image_height;
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_WIDTH, &cuda_output_image_width, sizeof(int)));
    CUDA_CALL(cudaMemcpyToSymbol(D_OUTPUT_IMAGE_HEIGHT, &cuda_output_image_height, sizeof(int)));
    const int CHANNELS = 3;
    CUDA_CALL(cudaMalloc(&d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char)));
}

__global__ void stage1_kernel(const Particle* __restrict__ cuda_particles, const unsigned int particles_count, unsigned int* pixel_contribs, const unsigned int image_width, const unsigned int image_height) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < particles_count) {
        int x_min = (int)roundf(cuda_particles[tid].location[0] - cuda_particles[tid].radius);
        int y_min = (int)roundf(cuda_particles[tid].location[1] - cuda_particles[tid].radius);
        int x_max = (int)roundf(cuda_particles[tid].location[0] + cuda_particles[tid].radius);
        int y_max = (int)roundf(cuda_particles[tid].location[1] + cuda_particles[tid].radius);

        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= image_width ? image_width - 1 : x_max;
        y_max = y_max >= image_height ? image_height - 1 : y_max;

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - cuda_particles[tid].location[0];
                const float y_ab = (float)y + 0.5f - cuda_particles[tid].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= cuda_particles[tid].radius) {
                    const unsigned int pixel_offset = y * image_width + x;
                    atomicAdd(&pixel_contribs[pixel_offset], 1);
                }
            }
        }
    }
}

void cuda_stage1() {
    const unsigned int threads_per_block = 512;
    unsigned int num_blocks = (cuda_particles_count + threads_per_block - 1) / threads_per_block;
    cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));

    stage1_kernel<<<num_blocks, threads_per_block>>>(d_particles, cuda_particles_count, d_pixel_contribs, cuda_output_image_width, cuda_output_image_height);

#ifdef VALIDATION
    Particle* h_particles = new Particle[cuda_particles_count];
    unsigned int* h_pixel_contribs = new unsigned int[cuda_output_image_width * cuda_output_image_height];

    cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    printf("\n"); //This is purely formatting for my own sanity sake
    validate_pixel_contribs(h_particles, cuda_particles_count, h_pixel_contribs, cuda_output_image_width, cuda_output_image_height);

    delete[] h_particles;
    delete[] h_pixel_contribs;
#endif
}

//I am passing width and height here to the kernel because I found I get slightly better performance compared to using the device pointers itself
__global__ void stage2_kernel_part1(const Particle* __restrict__ particles, const unsigned int particles_count, unsigned int* pixel_contribs,
    unsigned int* pixel_index, unsigned char* pixel_contrib_colours, float* pixel_contrib_depth,
    const unsigned int width, const unsigned int height) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < particles_count) {
        int x_min = (int)roundf(particles[idx].location[0] - particles[idx].radius);
        int y_min = (int)roundf(particles[idx].location[1] - particles[idx].radius);
        int x_max = (int)roundf(particles[idx].location[0] + particles[idx].radius);
        int y_max = (int)roundf(particles[idx].location[1] + particles[idx].radius);

        x_min = x_min < 0 ? 0 : x_min;
        y_min = y_min < 0 ? 0 : y_min;
        x_max = x_max >= width ? width - 1 : x_max;
        y_max = y_max >= height ? height - 1 : y_max;

        for (int x = x_min; x <= x_max; ++x) {
            for (int y = y_min; y <= y_max; ++y) {
                const float x_ab = (float)x + 0.5f - particles[idx].location[0];
                const float y_ab = (float)y + 0.5f - particles[idx].location[1];
                const float pixel_distance = sqrtf(x_ab * x_ab + y_ab * y_ab);
                if (pixel_distance <= particles[idx].radius) {
                    const unsigned int pixel_offset = y * width + x;
                    const unsigned int storage_offset = pixel_index[pixel_offset] + atomicAdd(&pixel_contribs[pixel_offset], 1);
                    pixel_contrib_colours[4 * storage_offset] = particles[idx].color[0];
                    pixel_contrib_colours[4 * storage_offset + 1] = particles[idx].color[1];
                    pixel_contrib_colours[4 * storage_offset + 2] = particles[idx].color[2];
                    pixel_contrib_colours[4 * storage_offset + 3] = particles[idx].color[3];
                    pixel_contrib_depth[storage_offset] = particles[idx].location[2];
                }
            }
        }
    }
}

__device__ void swap_pairs(float& key1, float& key2, unsigned char* color1, unsigned char* color2) {
    float temp_key = key1;
    key1 = key2;
    key2 = temp_key;
    unsigned char temp_color[4];
    memcpy(temp_color, color1, 4 * sizeof(unsigned char));
    memcpy(color1, color2, 4 * sizeof(unsigned char));
    memcpy(color2, temp_color, 4 * sizeof(unsigned char));
}

__global__ void stage2_kernel_part2(float* keys, unsigned char* colors, unsigned int* indices, int num_pixels) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= num_pixels) {
        return;
    }
    int start = indices[index];
    int end = indices[index + 1] - 1;
    for (int i = start; i <= end; i++) {
        for (int j = start; j < end; j++) {
            if (keys[j] > keys[j + 1]) {
                swap_pairs(keys[j], keys[j + 1], colors + j * 4, colors + (j + 1) * 4);
            }
        }
    }
}

void cuda_stage2() {
    unsigned int* h_pixel_index = new unsigned int[cuda_output_image_width * cuda_output_image_height + 1];

    // Part 1
    cudaMemset(d_pixel_index, 0, (cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int));
    thrust::device_ptr<unsigned int> dev_pixel_contribs(d_pixel_contribs);
    thrust::device_ptr<unsigned int> dev_pixel_index(d_pixel_index);
    thrust::exclusive_scan(dev_pixel_contribs, dev_pixel_contribs + cuda_output_image_width * cuda_output_image_height + 1, dev_pixel_index);

    cudaMemcpy(h_pixel_index, d_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    const unsigned int TOTAL_CONTRIBS = h_pixel_index[cuda_output_image_width * cuda_output_image_height];
    if (TOTAL_CONTRIBS > cuda_pixel_contrib_count) {
        // (Re)Allocate colour storage
        if (d_pixel_contrib_colours) cudaFree(d_pixel_contrib_colours);
        if (d_pixel_contrib_depth) cudaFree(d_pixel_contrib_depth);
        cudaMalloc(&d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char));
        cudaMalloc(&d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float));
        cuda_pixel_contrib_count = TOTAL_CONTRIBS;
    }
    cudaMemset(d_pixel_contribs, 0, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int));

    // Part 2 and 3
    const unsigned int BLOCK_SIZE = 512;
    stage2_kernel_part1<<<(cuda_particles_count + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE >>>(d_particles, cuda_particles_count, d_pixel_contribs,
        d_pixel_index, d_pixel_contrib_colours, d_pixel_contrib_depth,
        cuda_output_image_width, cuda_output_image_height);

    // Part 4
    stage2_kernel_part2<<<(cuda_output_image_width * cuda_output_image_height + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(d_pixel_contrib_depth, d_pixel_contrib_colours, d_pixel_index, cuda_output_image_width * cuda_output_image_height);

#ifdef VALIDATION
    Particle* h_particles = new Particle[cuda_particles_count];
    unsigned int* h_pixel_contribs = new unsigned int[cuda_output_image_width * cuda_output_image_height];
    unsigned char* h_pixel_contrib_colours = new unsigned char[cuda_output_image_width * cuda_output_image_height * 4];
    float* h_pixel_contrib_depth = new float[cuda_output_image_width * cuda_output_image_height];

    cudaMemcpy(h_particles, d_particles, cuda_particles_count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_contribs, d_pixel_contribs, cuda_output_image_width * cuda_output_image_height * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_index, d_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    if (TOTAL_CONTRIBS >= cuda_pixel_contrib_count) {
        h_pixel_contrib_colours = new unsigned char[TOTAL_CONTRIBS * 4];
        h_pixel_contrib_depth = new float[TOTAL_CONTRIBS];
        cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, TOTAL_CONTRIBS * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pixel_contrib_depth, d_pixel_contrib_depth, TOTAL_CONTRIBS * sizeof(float), cudaMemcpyDeviceToHost);
    } else {
        cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, cuda_output_image_width * cuda_output_image_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_pixel_contrib_depth, d_pixel_contrib_depth, cuda_output_image_width * cuda_output_image_height, cudaMemcpyDeviceToHost);
    }

    validate_pixel_index(h_pixel_contribs, h_pixel_index, cuda_output_image_width, cuda_output_image_height);
    validate_sorted_pairs(h_particles, cuda_particles_count, h_pixel_index, cuda_output_image_width, cuda_output_image_height, h_pixel_contrib_colours, h_pixel_contrib_depth);

    //Cleanup
    delete[] h_particles;
    delete[] h_pixel_contribs;
    delete[] h_pixel_contrib_colours;
    delete[] h_pixel_contrib_depth;
#endif
    delete[] h_pixel_index;
}

__global__ void cuda_stage3(unsigned char* d_output_image_data, const unsigned int* __restrict__ d_pixel_contribs, const unsigned int* __restrict__ d_pixel_index, const unsigned char* __restrict__ d_pixel_contrib_colours) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= D_OUTPUT_IMAGE_WIDTH * D_OUTPUT_IMAGE_HEIGHT) {
        return;
    }

    for (unsigned int j = d_pixel_index[i]; j < d_pixel_index[i + 1]; ++j) {
        const float opacity = (float)d_pixel_contrib_colours[j * 4 + 3] / (float)255;
        d_output_image_data[(i * 3) + 0] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 0] * opacity + (float)d_output_image_data[(i * 3) + 0] * (1 - opacity));
        d_output_image_data[(i * 3) + 1] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 1] * opacity + (float)d_output_image_data[(i * 3) + 1] * (1 - opacity));
        d_output_image_data[(i * 3) + 2] = (unsigned char)((float)d_pixel_contrib_colours[j * 4 + 2] * opacity + (float)d_output_image_data[(i * 3) + 2] * (1 - opacity));
    }
}

void cuda_stage3() {
    cudaMemset(d_output_image_data, 255, cuda_output_image_width * cuda_output_image_height * 3);
    const int block_size = 512;
    int grid_size = (cuda_output_image_width * cuda_output_image_height + block_size - 1) / block_size;

    cuda_stage3<<<grid_size, block_size>>>(d_output_image_data, d_pixel_contribs, d_pixel_index, d_pixel_contrib_colours);
    cudaDeviceSynchronize();

#ifdef VALIDATION
    unsigned int* h_pixel_index = new unsigned int[cuda_output_image_width * cuda_output_image_height + 1];
    unsigned char* h_pixel_contrib_colours = new unsigned char[cuda_output_image_width * cuda_output_image_height * 4];
    unsigned char* h_output_image_data = new unsigned char[cuda_output_image_width * cuda_output_image_height * 3];

    cudaMemcpy(h_pixel_index, d_pixel_index, ((cuda_output_image_width * cuda_output_image_height + 1) * sizeof(unsigned int)), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, cuda_output_image_width * cuda_output_image_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_image_data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if (h_pixel_index[cuda_output_image_width * cuda_output_image_height] >= cuda_pixel_contrib_count) {
        h_pixel_contrib_colours = new unsigned char[(h_pixel_index[cuda_output_image_width * cuda_output_image_height]) * 4];
        cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, (h_pixel_index[cuda_output_image_width * cuda_output_image_height]) * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }
    else {
        cudaMemcpy(h_pixel_contrib_colours, d_pixel_contrib_colours, cuda_output_image_width * cuda_output_image_height * 4 * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }

    CImage cuda_output_image;
    cuda_output_image.width = (int)cuda_output_image_width;
    cuda_output_image.height = (int)cuda_output_image_height;
    cuda_output_image.channels = 3;
    cuda_output_image.data = (unsigned char*)malloc(cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));
    memcpy(cuda_output_image.data, h_output_image_data, cuda_output_image_width * cuda_output_image_height * 3 * sizeof(unsigned char));

    validate_blend(h_pixel_index, h_pixel_contrib_colours, &cuda_output_image);

    delete[] h_pixel_index;
    delete[] h_pixel_contrib_colours;
#endif    
}
void cuda_end(CImage *output_image) {
    const int CHANNELS = 3;
    output_image->width = cuda_output_image_width;
    output_image->height = cuda_output_image_height;
    output_image->channels = CHANNELS;
    CUDA_CALL(cudaMemcpy(output_image->data, d_output_image_data, cuda_output_image_width * cuda_output_image_height * CHANNELS * sizeof(unsigned char), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaFree(d_pixel_contrib_depth));
    CUDA_CALL(cudaFree(d_pixel_contrib_colours));
    CUDA_CALL(cudaFree(d_output_image_data));
    CUDA_CALL(cudaFree(d_pixel_index));
    CUDA_CALL(cudaFree(d_pixel_contribs));
    CUDA_CALL(cudaFree(d_particles));
    d_pixel_contrib_depth = 0;
    d_pixel_contrib_colours = 0;
    d_output_image_data = 0;
    d_pixel_index = 0;
    d_pixel_contribs = 0;
    d_particles = 0;
}
