#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include <tracy/TracyC.h>
#include "c3d_internal.h"

typedef struct C3DSubmissionRecord
{
  uint64_t serial;
  cudaEvent_t event;
  void (*cleanup)(void*);
  void* cleanupContext;
  C3DSubmissionRecord* next;
} C3DSubmissionRecord;

static std::mutex gSubmissionLock;
static C3DSubmissionRecord* gSubmissionHead = nullptr;
static uint64_t gNextSubmissionSerial = 1;

static bool c3dStorageIsBound(uint64_t serial)
{
  return serial != 0 && c3dIsSubmissionPending(serial);
}

static C3DTextureStorage* c3dAllocateTextureStorage(C3DTextureInfo info, size_t size)
{
  C3DTextureStorage* storage = (C3DTextureStorage*)malloc(sizeof(C3DTextureStorage));
  if (!storage)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture storage node");
    return nullptr;
  }

  memset(storage, 0, sizeof(*storage));
  storage->info = info;
  storage->size = size;
  if (!c3dCheckCUDA(cudaMalloc((void**)&storage->data, size), "cudaMalloc failed while creating texture storage"))
  {
    free(storage);
    return nullptr;
  }

  return storage;
}

static C3DBufferStorage* c3dAllocateBufferStorage(size_t size)
{
  C3DBufferStorage* storage = (C3DBufferStorage*)malloc(sizeof(C3DBufferStorage));
  if (!storage)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate buffer storage node");
    return nullptr;
  }

  memset(storage, 0, sizeof(*storage));
  storage->size = size;
  if (size == 0)
  {
    return storage;
  }

  storage->hostData = (uint8_t*)malloc(size);
  if (!storage->hostData)
  {
    free(storage);
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate buffer host storage");
    return nullptr;
  }

  memset(storage->hostData, 0, size);
  if (!c3dCheckCUDA(cudaMalloc((void**)&storage->deviceData, size), "cudaMalloc failed while creating buffer storage"))
  {
    free(storage->hostData);
    free(storage);
    return nullptr;
  }

  if (!c3dCheckCUDA(cudaMemset(storage->deviceData, 0, size), "cudaMemset failed while initializing buffer storage"))
  {
    cudaFree(storage->deviceData);
    free(storage->hostData);
    free(storage);
    return nullptr;
  }

  return storage;
}

static C3DStageBufferStorage* c3dAllocateStageStorage(size_t size)
{
  C3DStageBufferStorage* storage = (C3DStageBufferStorage*)malloc(sizeof(C3DStageBufferStorage));
  if (!storage)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate stage buffer storage node");
    return nullptr;
  }

  memset(storage, 0, sizeof(*storage));
  storage->size = size;
  if (size != 0 && !c3dCheckCUDA(cudaHostAlloc((void**)&storage->data, size, cudaHostAllocDefault), "failed to allocate stage buffer storage"))
  {
    free(storage);
    return nullptr;
  }

  if (storage->data)
  {
    memset(storage->data, 0, size);
  }

  return storage;
}

bool c3dRefreshSubmissionState(void)
{
  TracyCZoneN(zone, "c3dRefreshSubmissionState", 1);
  C3DSubmissionRecord* completed = nullptr;
  {
    std::lock_guard<std::mutex> lock(gSubmissionLock);
    C3DSubmissionRecord** link = &gSubmissionHead;
    while (*link)
    {
      C3DSubmissionRecord* record = *link;
      cudaError_t status = cudaEventQuery(record->event);
      if (status == cudaErrorNotReady)
      {
        link = &record->next;
        continue;
      }

      if (status != cudaSuccess)
      {
        c3dCheckCUDA(status, "cudaEventQuery failed while retiring submissions");
        TracyCZoneEnd(zone);
        return false;
      }

      *link = record->next;
      record->next = completed;
      completed = record;
    }
  }

  while (completed)
  {
    C3DSubmissionRecord* record = completed;
    completed = record->next;
    if (record->cleanup)
    {
      record->cleanup(record->cleanupContext);
    }

    cudaEventDestroy(record->event);
    free(record);
  }

  TracyCZoneEnd(zone);
  return true;
}

bool c3dRegisterSubmission(uint64_t* serialOut, cudaStream_t stream, void (*cleanup)(void*), void* cleanupContext)
{
  TracyCZoneN(zone, "c3dRegisterSubmission", 1);
  if (!serialOut)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "submission serial output must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dRefreshSubmissionState())
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DSubmissionRecord* record = (C3DSubmissionRecord*)malloc(sizeof(C3DSubmissionRecord));
  if (!record)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate submission record");
    TracyCZoneEnd(zone);
    return false;
  }

  memset(record, 0, sizeof(*record));
  if (!c3dCheckCUDA(cudaEventCreateWithFlags(&record->event, cudaEventDisableTiming), "cudaEventCreateWithFlags failed while creating submission event"))
  {
    free(record);
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dCheckCUDA(cudaEventRecord(record->event, stream), "cudaEventRecord failed while registering submission"))
  {
    cudaEventDestroy(record->event);
    free(record);
    TracyCZoneEnd(zone);
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(gSubmissionLock);
    record->serial = gNextSubmissionSerial++;
    record->cleanup = cleanup;
    record->cleanupContext = cleanupContext;
    record->next = gSubmissionHead;
    gSubmissionHead = record;
    *serialOut = record->serial;
  }

  TracyCZoneEnd(zone);
  return true;
}

bool c3dWaitForSubmission(uint64_t serial)
{
  if (serial == 0)
  {
    return true;
  }

  for (;;)
  {
    if (!c3dRefreshSubmissionState())
    {
      return false;
    }

    cudaEvent_t event = nullptr;
    {
      std::lock_guard<std::mutex> lock(gSubmissionLock);
      for (C3DSubmissionRecord* record = gSubmissionHead; record; record = record->next)
      {
        if (record->serial == serial)
        {
          event = record->event;
          break;
        }
      }
    }

    if (!event)
    {
      return true;
    }

    if (!c3dCheckCUDA(cudaEventSynchronize(event), "cudaEventSynchronize failed while waiting for submission"))
    {
      return false;
    }
  }
}

bool c3dIsSubmissionPending(uint64_t serial)
{
  if (serial == 0)
  {
    return false;
  }

  if (!c3dRefreshSubmissionState())
  {
    return true;
  }

  std::lock_guard<std::mutex> lock(gSubmissionLock);
  for (C3DSubmissionRecord* record = gSubmissionHead; record; record = record->next)
  {
    if (record->serial == serial)
    {
      return true;
    }
  }

  return false;
}

bool c3dEnqueueSubmissionWait(uint64_t serial, cudaStream_t stream)
{
  TracyCZoneN(zone, "c3dEnqueueSubmissionWait", 1);
  if (serial == 0)
  {
    TracyCZoneEnd(zone);
    return true;
  }

  if (!c3dRefreshSubmissionState())
  {
    TracyCZoneEnd(zone);
    return false;
  }

  cudaEvent_t event = nullptr;
  {
    std::lock_guard<std::mutex> lock(gSubmissionLock);
    for (C3DSubmissionRecord* record = gSubmissionHead; record; record = record->next)
    {
      if (record->serial == serial)
      {
        event = record->event;
        break;
      }
    }
  }

  if (!event)
  {
    TracyCZoneEnd(zone);
    return true;
  }

  bool success = c3dCheckCUDA(cudaStreamWaitEvent(stream, event, 0), "cudaStreamWaitEvent failed while chaining submission dependency");
  TracyCZoneEnd(zone);
  return success;
}

bool c3dEnsureTextureStorageForWrite(C3DTexture* texture, bool cycle, C3DTextureStorage** storageOut)
{
  TracyCZoneN(zone, "c3dEnsureTextureStorageForWrite", 1);
  if (!texture || !storageOut)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture and storage output must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dRefreshSubmissionState())
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DTextureStorage* storage = texture->current;
  if (!storage)
  {
    storage = c3dAllocateTextureStorage(texture->info, texture->size);
    if (!storage)
    {
      return false;
    }

    storage->next = texture->storages;
    texture->storages = storage;
    texture->current = storage;
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  if (!cycle || !c3dStorageIsBound(storage->lastBoundSubmission))
  {
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  for (storage = texture->storages; storage; storage = storage->next)
  {
    if (!c3dStorageIsBound(storage->lastBoundSubmission))
    {
      texture->current = storage;
      *storageOut = storage;
      TracyCZoneEnd(zone);
      return true;
    }
  }

  storage = c3dAllocateTextureStorage(texture->info, texture->size);
  if (!storage)
  {
    TracyCZoneEnd(zone);
    return false;
  }

  storage->next = texture->storages;
  texture->storages = storage;
  texture->current = storage;
  *storageOut = storage;
  TracyCZoneEnd(zone);
  return true;
}

bool c3dEnsureBufferStorageForWrite(C3DBuffer* buffer, bool cycle, C3DBufferStorage** storageOut)
{
  TracyCZoneN(zone, "c3dEnsureBufferStorageForWrite", 1);
  if (!buffer || !storageOut)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer and storage output must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dRefreshSubmissionState())
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DBufferStorage* storage = buffer->current;
  if (!storage)
  {
    storage = c3dAllocateBufferStorage(buffer->info.size);
    if (!storage)
    {
      return false;
    }

    storage->next = buffer->storages;
    buffer->storages = storage;
    buffer->current = storage;
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  if (!cycle || !c3dStorageIsBound(storage->lastBoundSubmission))
  {
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  for (storage = buffer->storages; storage; storage = storage->next)
  {
    if (!c3dStorageIsBound(storage->lastBoundSubmission))
    {
      buffer->current = storage;
      *storageOut = storage;
      TracyCZoneEnd(zone);
      return true;
    }
  }

  storage = c3dAllocateBufferStorage(buffer->info.size);
  if (!storage)
  {
    TracyCZoneEnd(zone);
    return false;
  }

  storage->next = buffer->storages;
  buffer->storages = storage;
  buffer->current = storage;
  *storageOut = storage;
  TracyCZoneEnd(zone);
  return true;
}

bool c3dEnsureStageStorageForWrite(C3DStageBuffer* stageBuffer, bool cycle, C3DStageBufferStorage** storageOut)
{
  TracyCZoneN(zone, "c3dEnsureStageStorageForWrite", 1);
  if (!stageBuffer || !storageOut)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer and storage output must be non-null");
    TracyCZoneEnd(zone);
    return false;
  }

  if (!c3dRefreshSubmissionState())
  {
    TracyCZoneEnd(zone);
    return false;
  }

  C3DStageBufferStorage* storage = stageBuffer->current;
  if (!storage)
  {
    storage = c3dAllocateStageStorage(stageBuffer->info.size);
    if (!storage)
    {
      return false;
    }

    storage->next = stageBuffer->storages;
    stageBuffer->storages = storage;
    stageBuffer->current = storage;
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  if (!cycle || !c3dStorageIsBound(storage->lastBoundSubmission))
  {
    *storageOut = storage;
    TracyCZoneEnd(zone);
    return true;
  }

  for (storage = stageBuffer->storages; storage; storage = storage->next)
  {
    if (!c3dStorageIsBound(storage->lastBoundSubmission))
    {
      stageBuffer->current = storage;
      *storageOut = storage;
      TracyCZoneEnd(zone);
      return true;
    }
  }

  storage = c3dAllocateStageStorage(stageBuffer->info.size);
  if (!storage)
  {
    TracyCZoneEnd(zone);
    return false;
  }

  storage->next = stageBuffer->storages;
  stageBuffer->storages = storage;
  stageBuffer->current = storage;
  *storageOut = storage;
  TracyCZoneEnd(zone);
  return true;
}

C3DTextureStorage* c3dGetCurrentTextureStorage(C3DTexture* texture)
{
  return texture ? texture->current : nullptr;
}

C3DBufferStorage* c3dGetCurrentBufferStorage(C3DBuffer* buffer)
{
  return buffer ? buffer->current : nullptr;
}

C3DStageBufferStorage* c3dGetCurrentStageStorage(C3DStageBuffer* stageBuffer)
{
  return stageBuffer ? stageBuffer->current : nullptr;
}

void c3dBindTextureStorage(C3DTextureStorage* storage, uint64_t serial)
{
  if (storage)
  {
    storage->lastBoundSubmission = serial;
  }
}

void c3dBindBufferStorage(C3DBufferStorage* storage, uint64_t serial)
{
  if (storage)
  {
    storage->lastBoundSubmission = serial;
  }
}

void c3dBindStageStorage(C3DStageBufferStorage* storage, uint64_t serial)
{
  if (storage)
  {
    storage->lastBoundSubmission = serial;
  }
}
