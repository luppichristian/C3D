#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <tracy/TracyC.h>
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

static void c3dDestroyBufferStorageChain(C3DBufferStorage* storage)
{
  while (storage)
  {
    C3DBufferStorage* next = storage->next;
    c3dWaitForSubmission(storage->lastBoundSubmission);
    free(storage->hostData);
    if (storage->deviceData)
    {
      cudaFree(storage->deviceData);
    }

    free(storage);
    storage = next;
  }
}

static bool c3dTryResizeBuffer(C3DBuffer* buffer, size_t size)
{
  buffer->info.size = size;
  c3dDestroyBufferStorageChain(buffer->storages);
  buffer->storages = nullptr;
  buffer->current = nullptr;
  return c3dEnsureBufferStorageForWrite(buffer, true, &buffer->current);
}

C3D_API C3DBuffer* c3dCreateBuffer(const C3DBufferInfo* info)
{
  TracyCZoneN(zone, "c3dCreateBuffer", 1);
  if (!info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer info must be non-null");
    TracyCZoneEnd(zone);
    return nullptr;
  }

  C3DBuffer* buffer = (C3DBuffer*)malloc(sizeof(C3DBuffer));
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate buffer object");
    TracyCZoneEnd(zone);
    return nullptr;
  }

  memset(buffer, 0, sizeof(*buffer));
  if (!c3dTryResizeBuffer(buffer, info->size))
  {
    free(buffer);
    TracyCZoneEnd(zone);
    return nullptr;
  }

  TracyCZoneEnd(zone);
  return buffer;
}

C3D_API bool c3dDeleteBuffer(C3DBuffer* buffer)
{
  TracyCZoneN(zone, "c3dDeleteBuffer", 1);
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  c3dDestroyBufferStorageChain(buffer->storages);
  free(buffer);
  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dResizeBuffer(C3DBuffer* buffer, size_t size)
{
  TracyCZoneN(zone, "c3dResizeBuffer", 1);
  if (!buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  bool result = c3dTryResizeBuffer(buffer, size);
  TracyCZoneEnd(zone);
  return result;
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
  TracyCZoneN(zone, "c3dReadBuffer", 1);
  if (!c3dCheckBufferTransferRange(buffer, bufferOffset, size, "buffer read range is out of bounds") || !c3dCheckStageTransferRange(stageBuffer, stageOffset, size, "stage buffer write range is out of bounds"))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DBufferStorage* bufferStorage = c3dGetCurrentBufferStorage(buffer);
  C3DStageBufferStorage* stageStorage = c3dGetCurrentStageStorage(stageBuffer);
  if (!c3dWaitForSubmission(bufferStorage ? bufferStorage->lastBoundSubmission : 0) || !c3dWaitForSubmission(stageStorage ? stageStorage->lastBoundSubmission : 0))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  if (size != 0)
  {
    memcpy(stageStorage->data + stageOffset, bufferStorage->hostData + bufferOffset, size);
  }

  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dWriteBuffer(C3DBuffer* buffer, size_t bufferOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset, bool cycle)
{
  TracyCZoneN(zone, "c3dWriteBuffer", 1);
  if (!c3dCheckBufferTransferRange(buffer, bufferOffset, size, "buffer write range is out of bounds") || !c3dCheckStageTransferRange(stageBuffer, stageOffset, size, "stage buffer read range is out of bounds"))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DBufferStorage* bufferStorage = nullptr;
  if (!c3dEnsureBufferStorageForWrite(buffer, cycle, &bufferStorage))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DStageBufferStorage* stageStorage = c3dGetCurrentStageStorage(stageBuffer);
  if (!c3dWaitForSubmission(stageStorage ? stageStorage->lastBoundSubmission : 0))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  if (size != 0)
  {
    memcpy(bufferStorage->hostData + bufferOffset, stageStorage->data + stageOffset, size);
    if (!c3dCheckCUDA(cudaMemcpy(bufferStorage->deviceData + bufferOffset, stageStorage->data + stageOffset, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing buffer"))
    {
      TracyCZoneEnd(zone);
      return false;
    }
  }

  TracyCZoneEnd(zone);
  return true;
}

C3D_API bool c3dBufferCopy(C3DBuffer* destination, size_t destinationOffset, C3DBuffer* source, size_t sourceOffset, size_t size, bool cycle)
{
  TracyCZoneN(zone, "c3dBufferCopy", 1);
  if (!c3dCheckBufferTransferRange(destination, destinationOffset, size, "destination buffer copy range is out of bounds") || !c3dCheckBufferTransferRange(source, sourceOffset, size, "source buffer copy range is out of bounds"))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  if (size == 0)
  {
    TracyCZoneEnd(zone);
    return true;
  }

  C3DBufferStorage* destinationStorage = nullptr;
  if (!c3dEnsureBufferStorageForWrite(destination, cycle, &destinationStorage))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DBufferStorage* sourceStorage = c3dGetCurrentBufferStorage(source);
  if (!c3dWaitForSubmission(sourceStorage ? sourceStorage->lastBoundSubmission : 0))
  {
    TracyCZoneEnd(zone);
    return false;
  }

  if (destination == source && destinationStorage == sourceStorage && destinationOffset < sourceOffset + size && sourceOffset < destinationOffset + size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "overlapping in-place buffer copies are not supported");
    TracyCZoneEnd(zone);
    return false;
  }

  memmove(destinationStorage->hostData + destinationOffset, sourceStorage->hostData + sourceOffset, size);
  bool result = c3dCheckCUDA(cudaMemcpy(destinationStorage->deviceData + destinationOffset, sourceStorage->deviceData + sourceOffset, size, cudaMemcpyDeviceToDevice), "cudaMemcpy failed while copying buffer");
  TracyCZoneEnd(zone);
  return result;
}
