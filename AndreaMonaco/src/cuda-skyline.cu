/****************************************************************************
 *
 * skyline.c - Serial implementaiton of the skyline operator
 *
 * Copyright (C) 2024 Moreno Marzolla
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * --------------------------------------------------------------------------
 *
 * Questo programma calcola lo skyline di un insieme di punti in D
 * dimensioni letti da standard input. Per una descrizione completa
 * si veda la specifica del progetto sulla piattaforma "Virtuale".
 *
 * Per compilare:
 *
 *      gcc -std=c99 -Wall -Wpedantic -O2 skyline.c -o skyline
 *
 * Per eseguire il programma:
 *
 *      ./skyline < input > output
 *
 ****************************************************************************/

#if _XOPEN_SOURCE < 600
#define _XOPEN_SOURCE 600
#endif

#include <assert.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#include "hpc.h"

typedef struct {
    float *P; /* coordinates P[i][j] of point i               */
    int N;    /* Number of points (rows of matrix P)          */
    int D;    /* Number of dimensions (columns of matrix P)   */
} points_t;

/**
 * Read input from stdin. Input format is:
 *
 * d [other ignored stuff]
 * N
 * p0,0 p0,1 ... p0,d-1
 * p1,0 p1,1 ... p1,d-1
 * ...
 * pn-1,0 pn-1,1 ... pn-1,d-1
 *
 */
void read_input(points_t *points) {
    char buf[1024];
    int N, D;
    float *P;

    if (1 != scanf("%d", &D)) {
        fprintf(stderr, "FATAL: can not read the dimension\n");
        exit(EXIT_FAILURE);
    }
    assert(D >= 2);
    if (NULL == fgets(buf, sizeof(buf), stdin)) {
        fprintf(stderr, "FATAL: can not read the first line\n");
        exit(EXIT_FAILURE);
    }
    if (1 != scanf("%d", &N)) {
        fprintf(stderr, "FATAL: can not read the number of points\n");
        exit(EXIT_FAILURE);
    }
    P = (float *)malloc(D * N * sizeof(*P));
    assert(P);
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < D; k++) {
            if (1 != scanf("%f", &(P[i * D + k]))) {
                fprintf(stderr,
                        "FATAL: failed to get coordinate %d of point %d\n", k,
                        i);
                exit(EXIT_FAILURE);
            }
        }
    }
    points->P = P;
    points->N = N;
    points->D = D;
}

void free_points(points_t *points) {
    free(points->P);
    points->P = NULL;
    points->N = points->D = -1;
}

__global__ void init_s(int *s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s[i] = 1;
    }
}

/* Returns 1 iff |p| dominates |q| */
__device__ int dominates(const float *p, const float *q, int D) {
    /* The following loops could be merged, but the keep them separated
       for the sake of readability */
    for (int k = 0; k < D; k++) {
        if (p[k] < q[k]) {
            return 0;
        }
    }
    for (int k = 0; k < D; k++) {
        if (p[k] > q[k]) {
            return 1;
        }
    }
    return 0;
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` iff point
 * `i` belongs to the skyline.The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
__global__ void skyline_kernel_2(float *P, int *s, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N || j >= N) return;

    if (dominates(&(P[i * D]), &(P[j * D]), D)) {
        atomicExch(&s[j], 0);
    }
}

/**
 * Compute the skyline of `points`. At the end, `s[i] == 1` iff point
 * `i` belongs to the skyline.The caller is responsible for
 * allocating the array `s` of length at least `points->N`.
 */
__global__ void skyline_kernel(float *P, int *s, int N, int D) {
    long long i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N) return;

    for (int j = 0; j < N && s[i]; j++) {
        if (dominates(&(P[i * D]), &(P[j * D]), D)) {
            atomicExch(&s[j], 0);
        }
    }
}

/**
 * Print the coordinates of points belonging to the skyline `s` to
 * standard ouptut. `s[i] == 1` iff point `i` belongs to the skyline.
 * The output format is the same as the input format, so that this
 * program can process its own output.
 */
void print_skyline(const points_t *points, const int *s, int r) {
    const int D = points->D;
    const int N = points->N;
    const float *P = points->P;

    printf("%d\n", D);
    printf("%d\n", r);
    for (int i = 0; i < N; i++) {
        if (s[i]) {
            for (int k = 0; k < D; k++) {
                printf("%f ", P[i * D + k]);
            }
            printf("\n");
        }
    }
}

int main(int argc, char *argv[]) {
    points_t points;

    if (argc != 1) {
        fprintf(stderr, "Usage: %s < input_file > output_file\n", argv[0]);
        return EXIT_FAILURE;
    }

    read_input(&points);  // unchanged

    int *s = (int *)malloc(points.N * sizeof(int));
    assert(s);

    const double tstart = hpc_gettime();

    float *d_P;
    int *d_s;
    cudaSafeCall(cudaMalloc(&d_P, points.N * points.D * sizeof(float)));
    cudaSafeCall(cudaMalloc(&d_s, points.N * sizeof(int)));

    cudaSafeCall(cudaMemcpy(d_P, points.P, points.N * points.D * sizeof(float),
                            cudaMemcpyHostToDevice));  // copy P on cuda

    int threads_per_block = 1024;
    int blocks = (points.N + threads_per_block - 1) / threads_per_block;
    init_s<<<blocks, threads_per_block>>>(d_s, points.N);
    cudaCheckError();

    dim3 blocks2D(32, 32);
    dim3 grid2D((points.N + 31) / 32, (points.N + 31) / 32);

    // skyline_kernel<<<blocks, threads_per_block>>>(d_P, d_s, points.N,
    //                                               points.D);

    skyline_kernel_2<<<grid2D, blocks2D>>>(d_P, d_s, points.N, points.D);
    cudaCheckError();

    cudaSafeCall(
        cudaMemcpy(s, d_s, points.N * sizeof(int), cudaMemcpyDeviceToHost));

    int r = 0;
    for (int i = 0; i < points.N; i++) {
        r += s[i];
    }

    const double elapsed = hpc_gettime() - tstart;

    print_skyline(&points, s, r);

    fprintf(stderr, "\n\t%d points\n", points.N);
    fprintf(stderr, "\t%d dimensions\n", points.D);
    fprintf(stderr, "\t%d points in skyline\n\n", r);
    fprintf(stderr, "Execution time (s) %f\n", elapsed);

    free_points(&points);
    free(s);
    cudaSafeCall(cudaFree(d_P));
    cudaSafeCall(cudaFree(d_s));

    return EXIT_SUCCESS;
}