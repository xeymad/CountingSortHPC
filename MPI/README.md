# Assignment: CONTEST-MPI: 3 Students GROUPS

## Dependencies

* CMake 3.9+
* MPICH
* Python3
* Pipenv

## How to run

1. Create a build directory and launch cmake

   ```batch
   mkdir build
   cd build
   cmake -DCMAKE_BUILD_TYPE=Release ..
   ```

2. Generate executables with `make`
3. To generate measures (TAKE A LOT OF TIME! Our measures are already included so you should skip this step) run `make generate_measures`
4. To extract mean times and speedup curves from them run `make extract_measures`

Results can be found in the `measures/measure` directory, divided by problem size and version.
