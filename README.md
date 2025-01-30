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