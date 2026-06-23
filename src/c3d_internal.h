#pragma once

#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

struct C3DTexture
{
  C3DTextureInfo info;
  size_t size;
  struct C3DTextureStorage* storages;
  struct C3DTextureStorage* current;
};

struct C3DBuffer
{
  C3DBufferInfo info;
  struct C3DBufferStorage* storages;
  struct C3DBufferStorage* current;
};

struct C3DStageBuffer
{
  C3DStageBufferInfo info;
  bool mapped;
  C3DMemoryAccess access;
  struct C3DStageBufferStorage* storages;
  struct C3DStageBufferStorage* current;
};

struct C3DTextureStorage
{
  C3DTextureInfo info;
  uint8_t* data;
  size_t size;
  uint64_t lastBoundSubmission;
  C3DTextureStorage* next;
};

struct C3DBufferStorage
{
  size_t size;
  uint8_t* hostData;
  uint8_t* deviceData;
  uint64_t lastBoundSubmission;
  C3DBufferStorage* next;
};

struct C3DStageBufferStorage
{
  size_t size;
  uint8_t* data;
  uint64_t lastBoundSubmission;
  C3DStageBufferStorage* next;
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
bool c3dRefreshSubmissionState(void);
bool c3dRegisterSubmission(uint64_t* serialOut, cudaStream_t stream, void (*cleanup)(void*), void* cleanupContext);
bool c3dWaitForSubmission(uint64_t serial);
bool c3dIsSubmissionPending(uint64_t serial);
bool c3dEnqueueSubmissionWait(uint64_t serial, cudaStream_t stream);
bool c3dEnsureTextureStorageForWrite(C3DTexture* texture, bool cycle, C3DTextureStorage** storageOut);
bool c3dEnsureBufferStorageForWrite(C3DBuffer* buffer, bool cycle, C3DBufferStorage** storageOut);
bool c3dEnsureStageStorageForWrite(C3DStageBuffer* stageBuffer, bool cycle, C3DStageBufferStorage** storageOut);
C3DTextureStorage* c3dGetCurrentTextureStorage(C3DTexture* texture);
C3DBufferStorage* c3dGetCurrentBufferStorage(C3DBuffer* buffer);
C3DStageBufferStorage* c3dGetCurrentStageStorage(C3DStageBuffer* stageBuffer);
void c3dBindTextureStorage(C3DTextureStorage* storage, uint64_t serial);
void c3dBindBufferStorage(C3DBufferStorage* storage, uint64_t serial);
void c3dBindStageStorage(C3DStageBufferStorage* storage, uint64_t serial);
