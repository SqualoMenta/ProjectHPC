Questo archivio contiene il materiale per il progetto del corso di
High Performance Computing, laurea in Ingegneria e Scienze
Informatiche, Universita' di Bologna sede di Cesena, AA 2023/2024.

Il file README (questo file) dovra' includere le istruzioni per la
compilazione e l'esecuzione dei programmi consegnati; per comodita',
nella directory src/ e' presente un Makefile che dovrebbe gia' essere
in grado di compilare le versioni OpenMP, MPI e/o CUDA eventualmente
presenti nella directory stessa. Si puo' modificare il Makefile
fornito, oppure si puo' decidere di non usarlo ed effettuare la
compilazione in modo manuale. In tal caso specificare in questo file i
comandi da usare per la compilazione dei programmi.

```c
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        s[i] = 1;
    }

    for (int i = 0; i < N; i++) {
        if (s[i]) {
            int a = 0;
#pragma omp parallel for reduction(+ : a)
            for (int j = 0; j < N; j++) {
                if (s[j] && dominates(&(P[i * D]), &(P[j * D]), D)) {
                    s[j] = 0;
                    a++;
                }
            }
            r -= a;
        }
    }
    return r;
```
Execution time (s) 27.786265 in parallelo test7

Execution time (s) 112.904749 seriale test7

Execution time (s) 8.567973 parallelo worst

Execution time (s) 41.932570 seriale worst

```c
#pragma omp parallel for
    for (int i = 0; i < N; i++) {
        s[i] = 1;
    }

#pragma omp parallel for collapse(2)
    for (long long i = 0; i < N; i++) {
        for (long long j = 0; j < N; j++) {
            if (s[i]) {
                if (s[j] && dominates(&(P[i * D]), &(P[j * D]), D)) {
#pragma omp atomic write
                    s[j] = 0;
                }
            }
        }
    }

#pragma omp parallel for reduction(+ : r)
    for (int i = 0; i < N; i++) {
        r += s[i];
    }

    return r;
```
Execution time (s) 3.120866 test7

Execution time (s) 0.894728 worst

But wait, does the collapse(2) distribute both i and j across threads? If so, multiple threads might handle different i and j pairs that could access the same j. For example, thread 1 could be at (i1, j1) and thread 2 at (i2, j1), both trying to update s[j1]. The atomic ensures that those writes don't interfere.

But another consideration: the check for s[j] might not be atomic. If two threads read s[j] as 1, then both proceed to set it to 0, but with atomic writes, that's okay. The first write would set it to 0, and the second would do nothing (since it's already 0). But since the check is outside the atomic, there's a race condition in reading s[j].

So, two threads could read s[j] as 1, then both enter the if, and both perform the atomic write. That's redundant but safe. The atomic ensures that the write is done correctly, but the check for s[j] is not atomic. So there's a possibility of a race where a thread reads s[j] as 1, but by the time it tries to write, another thread has already set it to 0. However, in that case, the second thread's write would be unnecessary but harmless. The result would still be correct because once s[j] is 0, further writes to 0 have no effect.

Therefore, the code remains correct even if multiple threads check s[j] and try to write it. The atomic ensures that the actual write is safe. So the collapse(2) can be retained, and the atomic write added, which would maintain performance while ensuring correctness.


Mentre lo strictly impoine alla versione seriale di eseguire controlli analoghi, l'altra versione pur portando allo stesso risultato sfrutta il fatto che eseguendo le operazioni in ordine sparso e possibile trovare prima valori appartenenti alla skyline e ottenere speedup superlineari.

Il codice prende un punto i e impone s = 0 il valore di ogni punto j dominato da i. Nel caso ogni punto sia sulla skyline chiaramete la versione parallela che preserva le stesse operazioni ha on ottima resa perche ogni inner loop di sara sempre lungo N, la versione parallela che parallellizza entrambi i cicli invece puo trovare prima i punti sulla skyline e ottenere speedup superlineari.

```c
int skyline(const points_t *points, int *s) {
    const int D = points->D; /* Number of dimensions (columns of matrix P)   */
    const int N = points->N; /* Number of points (rows of matrix P)          */
    const float *P = points->P; /* coordinates P[i][j] of point i */
    int r = 0;

#pragma omp parallel
    {
#pragma omp for
        for (int i = 0; i < N; i++) {
            s[i] = 1;
        }
#pragma omp barrier

#pragma omp for schedule(dynamic)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N && s[i]; j++) {
                if (s[j] && dominates(&(P[i * D]), &(P[j * D]), D)) {
#pragma omp atomic write
                    s[j] = 0;
                }
            }
        }
#pragma omp barrier

#pragma omp for reduction(+ : r)
        for (int i = 0; i < N; i++) {
            r += s[i];
        }
    }
    return r;
}
```

```c
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

__device__ int dominates(const float *p, const float *q, int D) {
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

__global__ void skyline_kernel(float *P, int *s, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int j = 0; j < N && s[i]; j++) {
        if (dominates(&(P[i * D]), &(P[j * D]), D)) {
            s[j] = 0;
        }
    }
}

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

    const double tstart = hpc_gettime();

    float *d_P;
    int *d_s;
    cudaMalloc(&d_P, points.N * points.D * sizeof(float));
    cudaMalloc(&d_s, points.N * sizeof(int));

    cudaMemcpy(d_P, points.P, points.N * points.D * sizeof(float),
               cudaMemcpyHostToDevice);  // copy P on cuda

    int threads_per_block = 1024;
    int blocks = (points.N + threads_per_block - 1) / threads_per_block;
    init_s<<<blocks, threads_per_block>>>(d_s, points.N);  // clean
    cudaDeviceSynchronize();

    skyline_kernel<<<blocks, threads_per_block>>>(d_P, d_s, points.N,
                                                  points.D);  // not perfect
    cudaDeviceSynchronize();

    int *s = (int *)malloc(points.N * sizeof(int));
    cudaMemcpy(s, d_s, points.N * sizeof(int), cudaMemcpyDeviceToHost);

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
    cudaFree(d_P);
    cudaFree(d_s);

    return EXIT_SUCCESS;
}
``` t test7 4 sec

```c
__global__ void skyline_kernel(float *P, int *s, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    for (int j = 0; j < N && s[i]; j++) {
        if (dominates(&(P[i * D]), &(P[j * D]), D)) {
            atomicExch(&s[j], 0);
        }
    }
}

__global__ void init_s(int *s, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        s[i] = 1;
    }
}
```

```c
__global__ void skyline_kernel(float *P, int *s, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j>=N) return;

    if(s[i] && s[j]){
        if (dominates(&(P[i * D]), &(P[j * D]), D)) {
            atomicExch(&s[j], 0);
        }
    }
}
```

```c
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Max blocchi per griglia (X): %d\n", prop.maxGridSize[0]);
printf("Max blocchi per griglia (Y): %d\n", prop.maxGridSize[1]);
```
Max blocchi per griglia (X): 2147483647
Max blocchi per griglia (Y): 65535

```c
dim3 blocks2D(32, 32);
dim3 grid2D((points.N + 31) / 32, (points.N + 31) / 32);

skyline_kernel_2<<<grid2D, blocks2D>>>(d_P, d_s, points.N, points.D);

__global__ void skyline_kernel_2(float *P, int *s, int N, int D) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= N || j >= N) return;

    if (s[i] && s[j]) {
        if (dominates(&(P[i * D]), &(P[j * D]), D)) {
            atomicExch(&s[j], 0);
        }
    }
}
```
E se N fosse 1024*1024?

il problema deriva dal numero di blocchi: sulle x possiamo mettere piu di 2^20 blocchi quindi N non sara mai abbastanza grande da risultare un problema. Invece sulle y l'inidice massimo che possiamo ottenere e i = 65535 * 32 = 2097120 che e maggiore di 1024*1024. Quindi il problema non sussiste.

Dopo queste ottimizzazioni il test 7 e stato eseguito in 1.6 secondi invece che in 4. 

La riduzione di r è ancora eseguita dalla cpu ma richiede 0.000474 secondi.





hipify-clang --perl //genera il file per eseguire hipfy perl
rimuovere cudaSafeCall e cuda check error, la gestione degli errori di allocazione andrebbe gestita in un altro modo
perl hipify-perl cuda-skyline.cu > cuda-skyline.cu.hip //non eseguire se non si vogliono testare delle modifiche
hipcc cuda-skyline.cu.hip -o hip-cuda-skyline

alternativamente si puo semplicemente usare il make
hipify-clang --perl //genera il file per eseguire hipfy perl
rimuovere eventuali cudaSafeCall non compatibili con hipify
make hip

eventuale chmod +x di hip-cuda-skyline

./hip-cuda-skyline < file.in > file.out

Testing HIP:

./hip-cuda-skyline < ../../../datasets/test2-N100000-D4.in > test.out

	100000 points
	4 dimensions
	10352 points in skyline

Execution time (s) 0.306539

./hip-cuda-skyline < ../../../datasets/test3-N100000-D10.in > test.out

	100020 points
	10 dimensions
	24892 points in skyline

Execution time (s) 0.412410


./hip-cuda-skyline < ../../../datasets/test4-N100000-D8.in > test.out

	100009 points
	8 dimensions
	17458 points in skyline

Execution time (s) 0.368484



./hip-cuda-skyline < ../../../datasets/test5-N100000-D20.in > test.out

	100000 points
	20 dimensions
	96973 points in skyline

Execution time (s) 0.600079


./hip-cuda-skyline < ../../../datasets/test6-N100000-D50.in > test.out

	100100 points
	50 dimensions
	100050 points in skyline

Execution time (s) 0.671831


./hip-cuda-skyline < ../../../datasets/test7-N100000-D200.in > test.out

	100400 points
	200 dimensions
	100200 points in skyline

Execution time (s) 0.651555


./hip-cuda-skyline < ../../../datasets/worst-N100000-D10.in > test.out

	100000 points
	10 dimensions
	100000 points in skyline

Execution time (s) 0.490648


./hip-cuda-skyline < ../../../datasets/test1-N100000-D3.in > test.out

	100000 points
	3 dimensions
	26 points in skyline

Execution time (s) 0.272481

Seriale:

./skyline < ../../../datasets/test1-N100000-D3.in > test.out

	100000 points
	3 dimensions
	26 points in skyline

Execution time (s) 0.018173


./skyline < ../../../datasets/test2-N100000-D4.in > test.out

	100000 points
	4 dimensions
	10352 points in skyline

Execution time (s) 1.956307


./skyline < ../../../datasets/test3-N100000-D10.in > test.out

	100020 points
	10 dimensions
	24892 points in skyline

Execution time (s) 15.071623


./skyline < ../../../datasets/test4-N100000-D8.in > test.out

	100009 points
	8 dimensions
	17458 points in skyline

Execution time (s) 9.410433


./skyline < ../../../datasets/test5-N100000-D20.in > test.out

	100000 points
	20 dimensions
	96973 points in skyline

Execution time (s) 53.443936


./skyline < ../../../datasets/test6-N100000-D50.in > test.out

	100100 points
	50 dimensions
	100050 points in skyline

Execution time (s) 74.180849


./skyline < ../../../datasets/test7-N100000-D200.in > test.out

	100400 points
	200 dimensions
	100200 points in skyline

Execution time (s) 100.987629


./skyline < ../../../datasets/worst-N100000-D10.in > test.out

	100000 points
	10 dimensions
	100000 points in skyline

Execution time (s) 42.652728


Open MP:

./omp-skyline < ../../../datasets/test1-N100000-D3.in > test.out

	100000 points
	3 dimensions
	26 points in skyline

Execution time (s) 0.003035


./omp-skyline < ../../../datasets/test2-N100000-D4.in > test.out

	100000 points
	4 dimensions
	10352 points in skyline

Execution time (s) 0.262431


./omp-skyline < ../../../datasets/test3-N100000-D10.in > test.out

	100020 points
	10 dimensions
	24892 points in skyline

Execution time (s) 1.775200


./omp-skyline < ../../../datasets/test4-N100000-D8.in > test.out

	100009 points
	8 dimensions
	17458 points in skyline

Execution time (s) 1.200517


./omp-skyline < ../../../datasets/test5-N100000-D20.in > test.out

	100000 points
	20 dimensions
	96973 points in skyline

Execution time (s) 6.584494


./omp-skyline < ../../../datasets/test6-N100000-D50.in > test.out

	100100 points
	50 dimensions
	100050 points in skyline

Execution time (s) 9.693624


./omp-skyline < ../../../datasets/test7-N100000-D200.in > test.out

	100400 points
	200 dimensions
	100200 points in skyline

Execution time (s) 13.471212


./omp-skyline < ../../../datasets/worst-N100000-D10.in > test.out

	100000 points
	10 dimensions
	100000 points in skyline

Execution time (s) 5.323223
