# FAST-GPU
FAST corner detection algorithm optimised for CUDA
## Usage:
```
.\FAST-GPU.exe -f image.jpg -t 75 -p 9 -m 2 -i -c 5
```
## Possible arguments:
* ```-f``` &nbsp; path to input file
* ```-t``` &nbsp; threshold value 0 - 255
* ```-p``` &nbsp; pi value 9 - 12 consecutive positive pixels
* ```-m``` &nbsp; mode 0 - 3 ... 0: Naive CPU, 1: OpenCV, 2: GPU global memory, 3: GPU shared memory
* ```-i``` &nbsp; detecting image
* ```-v``` &nbsp; detecting video
* ```-c``` &nbsp; size of drawn circles

## Example output:
![Output](output.jpg?raw=true "program Output")

## Includes:
* CUDA
* OpenCV