#pragma once

#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

struct C3DTexture
{
  C3DTextureInfo info;
  uint8_t* data;
  size_t size;
};

struct C3DBuffer
{
  C3DBufferInfo info;
  uint8_t* hostData;
  uint8_t* deviceData;
};

struct C3DStageBuffer
{
  uint8_t* data;
  C3DStageBufferInfo info;
  bool mapped;
  C3DMemoryAccess access;
};

struct C3DRecordedDraw
{
  C3DDrawInfo info;
};

struct C3DCommandBuffer
{
  bool inRenderPass;
  bool hasRenderPass;
  C3DRenderPassInfo renderPass;
  C3DRecordedDraw* draws;
  size_t drawCount;
  size_t drawCap;
  uint64_t* depthBuffer;
  size_t depthCap;
  void* linePrimitives;
  size_t linePrimitiveCap;
  void* trianglePrimitives;
  size_t trianglePrimitiveCap;
  void* textureViews;
  size_t textureViewCap;
  uint32_t* tileCountsDevice;
  uint32_t* tileOffsetsDevice;
  uint32_t* tileIndicesDevice;
  uint32_t* tileCountsHost;
  uint32_t* tileOffsetsHost;
  size_t tileCountCap;
  size_t tileOffsetCap;
  size_t tileIndexCap;
  size_t tileCountsHostCap;
  size_t tileOffsetsHostCap;
};

static bool c3dCheckCUDA(cudaError_t error, const char* desc)
{
  if (error == cudaSuccess)
    return true;
  char buffer[C3D_ERROR_DESC_CAP];
  snprintf(buffer, sizeof(buffer), "%s: %s (%d)", desc, cudaGetErrorString(error), (int)error);
  c3dThrowError(C3D_ERROR_CUDA, buffer);
  return false;
}

static bool c3dCheckRange(size_t totalSize, size_t offset, size_t size, const char* desc)
{
  if (offset > totalSize || size > totalSize - offset)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, desc);
    return false;
  }

  return true;
}

static __host__ __device__ size_t c3dGetTextureFormatSize(C3DTextureFormat format)
{
  switch (format)
  {
    case C3D_TEXTURE_FORMAT_RGBA8:
    case C3D_TEXTURE_FORMAT_BGRA8:
      return 4;
    case C3D_TEXTURE_FORMAT_DEPTH64:
      return sizeof(uint64_t);
  }

  return 0;
}

static __host__ __device__ size_t c3dGetIndexStride(C3DIndexSize indexSize)
{
  switch (indexSize)
  {
    case C3D_INDEX_SIZE_8:
      return 1;
    case C3D_INDEX_SIZE_16:
      return 2;
    case C3D_INDEX_SIZE_32:
      return 4;
  }

  return 0;
}

void c3dResetCommandBuffer(C3DCommandBuffer* commandBuffer);
