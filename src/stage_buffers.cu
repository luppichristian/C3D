#include <C3D.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c3d_internal.h"

static bool c3dTryResizeHostStorage(uint8_t** data, size_t oldSize, size_t newSize, const char* desc)
{
  uint8_t* newData = nullptr;
  if (newSize != 0)
  {
    if (!c3dCheckCUDA(cudaHostAlloc((void**)&newData, newSize, cudaHostAllocDefault), desc))
    {
      return false;
    }

    size_t copySize = oldSize < newSize ? oldSize : newSize;
    if (*data && copySize != 0)
    {
      memcpy(newData, *data, copySize);
    }

    if (newSize > copySize)
    {
      memset(newData + copySize, 0, newSize - copySize);
    }
  }

  if (*data && !c3dCheckCUDA(cudaFreeHost(*data), "cudaFreeHost failed while replacing stage buffer storage"))
  {
    if (newData)
    {
      cudaFreeHost(newData);
    }

    return false;
  }

  *data = newData;
  return true;
}

static bool c3dCheckHostRange(size_t totalSize, size_t offset, size_t size, const void* buffer, const char* desc)
{
  if (!buffer && size != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null when size is non-zero");
    return false;
  }

  return c3dCheckRange(totalSize, offset, size, desc);
}

static bool c3dIsValidMemoryAccess(C3DMemoryAccess access)
{
  return access >= C3D_MEMORY_ACCESS_READ && access <= C3D_MEMORY_ACCESS_READ_WRITE;
}

C3D_API C3DStageBuffer* c3dCreateStageBuffer(const C3DStageBufferInfo* info)
{
  if (!info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer info must be non-null");
    return nullptr;
  }

  if (info->initSize > info->size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer initSize must be less than or equal to size");
    return nullptr;
  }

  if (!info->initBuffer && info->initSize != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer initBuffer must be non-null when initSize is non-zero");
    return nullptr;
  }

  C3DStageBuffer* stageBuffer = (C3DStageBuffer*)malloc(sizeof(C3DStageBuffer));
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate stage buffer object");
    return nullptr;
  }

  memset(stageBuffer, 0, sizeof(C3DStageBuffer));
  stageBuffer->info = *info;
  if (!c3dTryResizeHostStorage(&stageBuffer->data, 0, info->size, "failed to allocate stage buffer storage"))
  {
    free(stageBuffer);
    return nullptr;
  }

  if (info->initSize != 0)
  {
    memcpy(stageBuffer->data, info->initBuffer, info->initSize);
  }

  return stageBuffer;
}

C3D_API bool c3dDeleteStageBuffer(C3DStageBuffer* stageBuffer)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  if (stageBuffer->data && !c3dCheckCUDA(cudaFreeHost(stageBuffer->data), "cudaFreeHost failed while deleting stage buffer"))
  {
    free(stageBuffer);
    return false;
  }

  free(stageBuffer);
  return true;
}

C3D_API bool c3dResizeStageBuffer(C3DStageBuffer* stageBuffer, size_t size)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  if (!c3dTryResizeHostStorage(&stageBuffer->data, stageBuffer->info.size, size, "failed to resize stage buffer storage"))
  {
    return false;
  }

  stageBuffer->info.size = size;
  if (size == 0)
  {
    stageBuffer->mapped = false;
  }

  return true;
}

C3D_API bool c3dGetStageBufferInfo(C3DStageBuffer* stageBuffer, C3DStageBufferInfo* info)
{
  if (!stageBuffer || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer and info output must be non-null");
    return false;
  }

  *info = stageBuffer->info;
  return true;
}

C3D_API bool c3dReadStageBuffer(C3DStageBuffer* stageBuffer, size_t offset, size_t size, void* buffer)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  if (!c3dCheckHostRange(stageBuffer->info.size, offset, size, buffer, "stage buffer read range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(buffer, stageBuffer->data + offset, size);
  }

  return true;
}

C3D_API bool c3dWriteStageBuffer(C3DStageBuffer* stageBuffer, size_t offset, size_t size, const void* buffer)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  if (!c3dCheckHostRange(stageBuffer->info.size, offset, size, buffer, "stage buffer write range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(stageBuffer->data + offset, buffer, size);
  }

  return true;
}

C3D_API void* c3dMapStageBuffer(C3DStageBuffer* stageBuffer, C3DMemoryAccess access)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return nullptr;
  }

  if (!c3dIsValidMemoryAccess(access))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer access mode is invalid");
    return nullptr;
  }

  stageBuffer->mapped = true;
  stageBuffer->access = access;
  return stageBuffer->data;
}

C3D_API bool c3dUnmapStageBuffer(C3DStageBuffer* stageBuffer)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  stageBuffer->mapped = false;
  stageBuffer->access = C3D_MEMORY_ACCESS_READ_WRITE;
  return true;
}
