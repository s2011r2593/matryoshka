# Running
## Setup
Enter an interactive job (`srun --nodes 1 --gres gpu:a40:4 --mem 8G --ntasks 4 --pty bash`)

Create a container that has CUDA, NCCL, and Open MPI (e.g. `apptainer build **your_container_name.sif** docker://seenry/piccl:nccl`).

Enter the container with `apptainer shell --nv **your_container_name.sif**` and clone this repo (`git clone git@github.com:s2011r2593/matryoshka.git`) 

`cd matryoshka` and run `make`

## Run
`mpirun -np 4 --oversubscribe ./matryoshka`

Should generate a file called perf-out.txt with results.
