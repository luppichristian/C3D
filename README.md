# C3D

C3D is a small graphics API for 2D/3D-style rendering backed by NVIDIA CUDA.

It exposes a compact graphics-like interface for:
- stage buffers
- textures
- vertex and index buffers
- command buffers
- render passes
- line, quad, and triangle draws

The repo also includes a Windows demo in `test/main.cpp` that renders into a CUDA texture, copies the result back to the CPU, and presents it with GDI.

## Why I Built This

I built C3D to experiment with a graphics API shape without depending on a full native graphics stack like Direct3D, Vulkan, or OpenGL.

The idea is to keep the public API small and C-friendly, but run the heavy lifting on CUDA.

## Project Layout

- `inc/C3D.h`: public API
- `src/*.cu`: CUDA implementation
- `test/main.cpp`: Windows demo application
- `project.bbs`: file for bbc build system

## Building

In order to build you need to this build system https://github.com/luppichristian/bbs

Then you can run the command `bbs build` from the root directory.

To build C3D with [Tracy](https://github.com/wolfpld/tracy) integrated through `bbs`, use one of the profiling configs:

`bbs build -t c3d_test -c debug-profile`

or

`bbs build -t c3d_test -c release-profile`

The `debug` and `release` configs build the normal targets without Tracy enabled. The `debug-profile` and `release-profile` configs enable `TRACY_ENABLE` for both the CUDA library and the demo while using the same `c3d_lib` / `c3d_test` targets.
