# Skyline

this program computes skyline with openMP and CUDA.
To compile the program open a terminal in src folder and type:

```bash
make
```

To run the OMP program type:

```bash
./omp-skyline < your_input_file > your_output_file
```

To run the CUDA program type:

```bash
./cuda-skyline < your_input_file > your_output_file
```

If you don't have the possibility to run the cuda program and you have the complete installation of ROCm, and the hipify-clang tool, you can convert the cuda code to hip code and compile with the following command:

```bash
make hip
```

There will be some warnings after the command hipify because of the name of the functions cudaCheckError and cudaSafeCall, but the code will be converted correctly.

Now you can simply run the hip program with the following command:

```bash
./hip-cuda-skyline < your_input_file > your_output_file
```
