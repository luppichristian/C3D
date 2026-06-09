#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c3d_internal.h"

static bool c3dCheckBufferTransferRange(C3DBuffer* buffer, size_t offset, size_t size, const char* desc)
{
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    return false;
  }

  return c3dCheckRange(buffer->info.size, offset, size, desc);
}

static bool c3dCheckStageTransferRange(C3DStageBuffer* stageBuffer, size_t offset, size_t size, const char* desc)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  return c3dCheckRange(stageBuffer->info.size, offset, size, desc);
}

static bool c3dTryResizeBuffer(C3DBuffer* buffer, size_t size, const char* desc)
{
  size_t oldSize = buffer->info.size;
  uint8_t* oldHostData = buffer->hostData;
  uint8_t* oldDeviceData = buffer->deviceData;

  uint8_t* newHostData = nullptr;
  if (size != 0)
  {
    newHostData = (uint8_t*)malloc(size);
    if (!newHostData)
    {
      c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, desc);
      return false;
    }

    size_t copySize = oldSize < size ? oldSize : size;
    if (oldHostData && copySize != 0)
    {
      memcpy(newHostData, oldHostData, copySize);
    }

    if (size > copySize)
    {
      memset(newHostData + copySize, 0, size - copySize);
    }
  }

  uint8_t* newDeviceData = nullptr;
  if (size != 0 && !c3dCheckCUDA(cudaMalloc((void**)&newDeviceData, size), desc))
  {
    free(newHostData);
    return false;
  }

  if (size != 0 && !c3dCheckCUDA(cudaMemcpy(newDeviceData, newHostData, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while updating buffer storage"))
  {
    free(newHostData);
    if (newDeviceData)
    {
      cudaFree(newDeviceData);
    }

    return false;
  }

  buffer->hostData = newHostData;
  buffer->deviceData = newDeviceData;
  free(oldHostData);
  if (oldDeviceData && !c3dCheckCUDA(cudaFree(oldDeviceData), "cudaFree failed while replacing buffer device storage"))
  {
    return false;
  }

  buffer->info.size = size;
  return true;
}

C3D_API C3DBuffer* c3dCreateBuffer(const C3DBufferInfo* info)
{
  if (!info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer info must be non-null");
    return nullptr;
  }

  C3DBuffer* buffer = (C3DBuffer*)malloc(sizeof(C3DBuffer));
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate buffer object");
    return nullptr;
  }

  memset(buffer, 0, sizeof(C3DBuffer));
  if (!c3dTryResizeBuffer(buffer, info->size, "failed to allocate buffer storage"))
  {
    free(buffer);
    return nullptr;
  }

  return buffer;
}

C3D_API bool c3dDeleteBuffer(C3DBuffer* buffer)
{
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    return false;
  }

  free(buffer->hostData);
  if (buffer->deviceData && !c3dCheckCUDA(cudaFree(buffer->deviceData), "cudaFree failed while deleting buffer"))
  {
    free(buffer);
    return false;
  }

  free(buffer);
  return true;
}

C3D_API bool c3dResizeBuffer(C3DBuffer* buffer, size_t size)
{
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    return false;
  }

  return c3dTryResizeBuffer(buffer, size, "failed to resize buffer storage");
}

C3D_API bool c3dGetBufferInfo(C3DBuffer* buffer, C3DBufferInfo* info)
{
  if (!buffer || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer and info output must be non-null");
    return false;
  }

  *info = buffer->info;
  return true;
}

C3D_API bool c3dReadBuffer(C3DBuffer* buffer, size_t bufferOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset)
{
  if (!c3dCheckBufferTransferRange(buffer, bufferOffset, size, "buffer read range is out of bounds") || !c3dCheckStageTransferRange(stageBuffer, stageOffset, size, "stage buffer write range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(stageBuffer->data + stageOffset, buffer->hostData + bufferOffset, size);
  }

  return true;
}

C3D_API bool c3dWriteBuffer(C3DBuffer* buffer, size_t bufferOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset)
{
  if (!c3dCheckBufferTransferRange(buffer, bufferOffset, size, "buffer write range is out of bounds") || !c3dCheckStageTransferRange(stageBuffer, stageOffset, size, "stage buffer read range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(buffer->hostData + bufferOffset, stageBuffer->data + stageOffset, size);
    if (!c3dCheckCUDA(cudaMemcpy(buffer->deviceData + bufferOffset, stageBuffer->data + stageOffset, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing buffer"))
    {
      return false;
    }
  }

  return true;
}

C3D_API bool c3dBufferCopy(C3DBuffer* destination, size_t destinationOffset, C3DBuffer* source, size_t sourceOffset, size_t size)
{
  if (!c3dCheckBufferTransferRange(destination, destinationOffset, size, "destination buffer copy range is out of bounds") || !c3dCheckBufferTransferRange(source, sourceOffset, size, "source buffer copy range is out of bounds"))
  {
    return false;
  }

  if (size == 0)
  {
    return true;
  }

  if (destination == source && destinationOffset < sourceOffset + size && sourceOffset < destinationOffset + size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "overlapping in-place buffer copies are not supported");
    return false;
  }

  memmove(destination->hostData + destinationOffset, source->hostData + sourceOffset, size);
  return c3dCheckCUDA(cudaMemcpy(destination->deviceData + destinationOffset, source->deviceData + sourceOffset, size, cudaMemcpyDeviceToDevice), "cudaMemcpy failed while copying buffer");
}
