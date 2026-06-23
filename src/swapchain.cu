#include <C3D.h>
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <mutex>
#include "c3d_internal.h"

typedef enum
{
  C3D_SWAPCHAIN_SLOT_FREE = 0,
  C3D_SWAPCHAIN_SLOT_ACQUIRED,
  C3D_SWAPCHAIN_SLOT_COPYING,
  C3D_SWAPCHAIN_SLOT_READY,
  C3D_SWAPCHAIN_SLOT_PRESENTING,
} C3DSwapchainSlotState;

typedef struct
{
  C3DTexture* texture;
  C3DStageBuffer* stageBuffer;
  cudaStream_t copyStream;
  cudaEvent_t copyEvent;
  uint64_t sequence;
  C3DSwapchainSlotState state;
} C3DSwapchainSlot;

struct C3DSwapchain
{
  C3DSwapchainInfo info;
  std::mutex lock;
  C3DSwapchainSlot* slots;
  size_t slotCount;
  size_t acquiredIndex;
  uint64_t nextSequence;
};

static bool c3dValidateSwapchainInfo(const C3DSwapchainInfo* info)
{
  if (!info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain info must be non-null");
    return false;
  }

  if (info->width == 0 || info->height == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain width and height must be greater than zero");
    return false;
  }

  if (info->format != C3D_TEXTURE_FORMAT_RGBA8)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain currently only supports C3D_TEXTURE_FORMAT_RGBA8");
    return false;
  }

  if (info->presentMode < C3D_PRESENT_MODE_FIFO || info->presentMode > C3D_PRESENT_MODE_IMMEDIATE)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain present mode is invalid");
    return false;
  }

  if (!info->presenter.present)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain presenter callback must be non-null");
    return false;
  }

  return true;
}

static bool c3dEnsureSwapchainStageBufferSize(C3DStageBuffer** stageBuffer, size_t size)
{
  if (!*stageBuffer)
  {
    C3DStageBufferInfo info = {0};
    info.size = size;
    *stageBuffer = c3dCreateStageBuffer(&info);
    return *stageBuffer != NULL;
  }

  C3DStageBufferInfo info = {0};
  if (!c3dGetStageBufferInfo(*stageBuffer, &info))
  {
    return false;
  }

  return info.size == size || c3dResizeStageBuffer(*stageBuffer, size);
}

static bool c3dSwapchainIsIdle(C3DSwapchain* swapchain)
{
  bool idle = swapchain->acquiredIndex == SIZE_MAX;
  for (size_t i = 0; idle && i < swapchain->slotCount; ++i)
  {
    idle = swapchain->slots[i].state == C3D_SWAPCHAIN_SLOT_FREE;
  }

  return idle;
}

static void c3dWaitForSwapchainIdle(C3DSwapchain* swapchain)
{
  for (;;)
  {
    std::unique_lock<std::mutex> lock(swapchain->lock);
    bool idle = c3dSwapchainIsIdle(swapchain);
    lock.unlock();
    if (idle)
    {
      return;
    }

    std::this_thread::yield();
  }
}

static bool c3dResizeSwapchainSlots(C3DSwapchain* swapchain)
{
  size_t size = swapchain->info.width * swapchain->info.height * 4;
  for (size_t i = 0; i < swapchain->slotCount; ++i)
  {
    C3DSwapchainSlot* slot = &swapchain->slots[i];
    if (slot->texture)
    {
      c3dDeleteTexture(slot->texture);
      slot->texture = NULL;
    }

    C3DTextureInfo textureInfo = {0};
    textureInfo.width = swapchain->info.width;
    textureInfo.height = swapchain->info.height;
    textureInfo.depth = 1;
    textureInfo.format = swapchain->info.format;
    slot->texture = c3dCreateTexture(&textureInfo);
    if (!slot->texture)
    {
      return false;
    }

    if (!c3dEnsureSwapchainStageBufferSize(&slot->stageBuffer, size))
    {
      return false;
    }

    if (!slot->copyStream && !c3dCheckCUDA(cudaStreamCreateWithFlags(&slot->copyStream, cudaStreamNonBlocking), "failed to create swapchain copy stream"))
    {
      return false;
    }

    if (!slot->copyEvent && !c3dCheckCUDA(cudaEventCreateWithFlags(&slot->copyEvent, cudaEventDisableTiming), "failed to create swapchain copy event"))
    {
      return false;
    }

    slot->sequence = 0;
    slot->state = C3D_SWAPCHAIN_SLOT_FREE;
  }

  swapchain->acquiredIndex = SIZE_MAX;
  return true;
}

static bool c3dPresentSlotImmediately(C3DSwapchain* swapchain, C3DSwapchainSlot* slot)
{
  void* buffer = c3dMapStageBuffer(slot->stageBuffer, C3D_MEMORY_ACCESS_READ, false);
  if (!buffer)
  {
    return false;
  }

  C3DPresentFrame frame = {0};
  frame.pixels = buffer;
  frame.width = swapchain->info.width;
  frame.height = swapchain->info.height;
  frame.rowPitch = swapchain->info.width * c3dGetTextureFormatSize(swapchain->info.format);
  frame.format = swapchain->info.format;
  frame.frameId = slot->sequence;
  bool presented = swapchain->info.presenter.present(swapchain->info.presenter.userData, &frame);
  c3dUnmapStageBuffer(slot->stageBuffer);
  return presented;
}

static void c3dUpdateCopyingSlots(C3DSwapchain* swapchain)
{
  for (size_t i = 0; i < swapchain->slotCount; ++i)
  {
    C3DSwapchainSlot* slot = &swapchain->slots[i];
    if (slot->state != C3D_SWAPCHAIN_SLOT_COPYING)
    {
      continue;
    }

    cudaError_t copyStatus = cudaEventQuery(slot->copyEvent);
    if (copyStatus == cudaSuccess)
    {
      slot->state = C3D_SWAPCHAIN_SLOT_READY;
    }
    else if (copyStatus != cudaErrorNotReady)
    {
      slot->state = C3D_SWAPCHAIN_SLOT_FREE;
    }
  }
}

static bool c3dDrainReadyPresentations(C3DSwapchain* swapchain)
{
  std::unique_lock<std::mutex> lock(swapchain->lock);
  c3dUpdateCopyingSlots(swapchain);

  for (;;)
  {
    C3DSwapchainSlot* selected = NULL;
    for (size_t i = 0; i < swapchain->slotCount; ++i)
    {
      C3DSwapchainSlot* slot = &swapchain->slots[i];
      if (slot->state != C3D_SWAPCHAIN_SLOT_READY)
      {
        continue;
      }

      if (!selected)
      {
        selected = slot;
        continue;
      }

      if (swapchain->info.presentMode == C3D_PRESENT_MODE_FIFO)
      {
        if (slot->sequence < selected->sequence)
        {
          selected = slot;
        }
      }
      else if (slot->sequence > selected->sequence)
      {
        selected = slot;
      }
    }

    if (selected && swapchain->info.presentMode != C3D_PRESENT_MODE_FIFO)
    {
      for (size_t i = 0; i < swapchain->slotCount; ++i)
      {
        C3DSwapchainSlot* slot = &swapchain->slots[i];
        if (slot != selected && slot->state == C3D_SWAPCHAIN_SLOT_READY)
        {
          slot->state = C3D_SWAPCHAIN_SLOT_FREE;
        }
      }
    }

    if (!selected)
    {
      return true;
    }

    selected->state = C3D_SWAPCHAIN_SLOT_PRESENTING;
    lock.unlock();

    if (!c3dCheckCUDA(cudaEventSynchronize(selected->copyEvent), "failed while waiting for swapchain copy completion"))
    {
      lock.lock();
      selected->state = C3D_SWAPCHAIN_SLOT_FREE;
      return false;
    }

    bool presented = c3dPresentSlotImmediately(swapchain, selected);
    lock.lock();
    selected->state = C3D_SWAPCHAIN_SLOT_FREE;
    if (!presented)
    {
      return false;
    }

    c3dUpdateCopyingSlots(swapchain);
  }
}

C3D_API C3DSwapchain* c3dCreateSwapchain(const C3DSwapchainInfo* info)
{
  if (!c3dValidateSwapchainInfo(info))
  {
    return nullptr;
  }

  C3DSwapchain* swapchain = (C3DSwapchain*)malloc(sizeof(C3DSwapchain));
  if (!swapchain)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate swapchain object");
    return nullptr;
  }

  memset(swapchain, 0, sizeof(C3DSwapchain));
  swapchain->info = *info;
  swapchain->slotCount = info->imageCount < 2 ? 2 : info->imageCount;
  swapchain->acquiredIndex = SIZE_MAX;
  swapchain->slots = (C3DSwapchainSlot*)calloc(swapchain->slotCount, sizeof(C3DSwapchainSlot));
  if (!swapchain->slots)
  {
    free(swapchain);
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate swapchain slots");
    return nullptr;
  }

  if (!c3dResizeSwapchainSlots(swapchain))
  {
    c3dDeleteSwapchain(swapchain);
    return nullptr;
  }

  return swapchain;
}

C3D_API bool c3dDeleteSwapchain(C3DSwapchain* swapchain)
{
  if (!swapchain)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain must be non-null");
    return false;
  }

  c3dDrainReadyPresentations(swapchain);

  if (swapchain->slots)
  {
    for (size_t i = 0; i < swapchain->slotCount; ++i)
    {
      if (swapchain->slots[i].copyEvent)
      {
        cudaEventDestroy(swapchain->slots[i].copyEvent);
      }

      if (swapchain->slots[i].copyStream)
      {
        cudaStreamDestroy(swapchain->slots[i].copyStream);
      }

      if (swapchain->slots[i].stageBuffer)
      {
        c3dDeleteStageBuffer(swapchain->slots[i].stageBuffer);
      }

      if (swapchain->slots[i].texture)
      {
        c3dDeleteTexture(swapchain->slots[i].texture);
      }
    }

    free(swapchain->slots);
  }

  free(swapchain);
  return true;
}

C3D_API bool c3dResizeSwapchain(C3DSwapchain* swapchain, size_t width, size_t height)
{
  if (!swapchain)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain must be non-null");
    return false;
  }

  if (width == 0 || height == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain width and height must be greater than zero");
    return false;
  }

  c3dWaitForSwapchainIdle(swapchain);
  swapchain->info.width = width;
  swapchain->info.height = height;
  return c3dResizeSwapchainSlots(swapchain);
}

C3D_API bool c3dGetSwapchainInfo(C3DSwapchain* swapchain, C3DSwapchainInfo* info)
{
  if (!swapchain || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain and info output must be non-null");
    return false;
  }

  *info = swapchain->info;
  return true;
}

C3D_API bool c3dAcquireNextTexture(C3DSwapchain* swapchain, C3DTexture** texture)
{
  if (!swapchain || !texture)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain and texture output must be non-null");
    return false;
  }

  if (swapchain->acquiredIndex != SIZE_MAX)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain already has an acquired texture");
    return false;
  }

  if (!c3dDrainReadyPresentations(swapchain))
  {
    return false;
  }

  for (;;)
  {
    size_t selectedIndex = SIZE_MAX;
    std::unique_lock<std::mutex> lock(swapchain->lock);
    c3dUpdateCopyingSlots(swapchain);
    for (size_t i = 0; i < swapchain->slotCount; ++i)
    {
      if (swapchain->slots[i].state == C3D_SWAPCHAIN_SLOT_FREE)
      {
        selectedIndex = i;
        break;
      }
    }

    if (selectedIndex == SIZE_MAX && swapchain->info.presentMode != C3D_PRESENT_MODE_FIFO)
    {
      for (size_t i = 0; i < swapchain->slotCount; ++i)
      {
        if (swapchain->slots[i].state != C3D_SWAPCHAIN_SLOT_READY)
        {
          continue;
        }

        if (selectedIndex == SIZE_MAX || swapchain->slots[i].sequence < swapchain->slots[selectedIndex].sequence)
        {
          selectedIndex = i;
        }
      }

      if (selectedIndex != SIZE_MAX)
      {
        swapchain->slots[selectedIndex].state = C3D_SWAPCHAIN_SLOT_FREE;
      }
    }

    if (selectedIndex != SIZE_MAX)
    {
      swapchain->slots[selectedIndex].state = C3D_SWAPCHAIN_SLOT_ACQUIRED;
      swapchain->acquiredIndex = selectedIndex;
      *texture = swapchain->slots[selectedIndex].texture;
      return true;
    }

    if (swapchain->info.presentMode != C3D_PRESENT_MODE_FIFO)
    {
      return false;
    }

    lock.unlock();
    std::this_thread::yield();
  }
}

C3D_API bool c3dPresentSwapchain(C3DSwapchain* swapchain)
{
  if (!swapchain)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain must be non-null");
    return false;
  }

  if (!c3dDrainReadyPresentations(swapchain))
  {
    return false;
  }

  if (swapchain->acquiredIndex == SIZE_MAX)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "swapchain has no acquired texture to present");
    return false;
  }

  C3DSwapchainSlot* slot = &swapchain->slots[swapchain->acquiredIndex];
  size_t size = swapchain->info.width * swapchain->info.height * 4;
  C3DTextureStorage* textureStorage = c3dGetCurrentTextureStorage(slot->texture);
  C3DStageBufferStorage* stageStorage = c3dGetCurrentStageStorage(slot->stageBuffer);
  if (!textureStorage || !stageStorage)
  {
    std::unique_lock<std::mutex> lock(swapchain->lock);
    slot->state = C3D_SWAPCHAIN_SLOT_FREE;
    swapchain->acquiredIndex = SIZE_MAX;
    return false;
  }

  if (!c3dEnqueueSubmissionWait(textureStorage->lastBoundSubmission, slot->copyStream) || !c3dCheckCUDA(cudaMemcpyAsync(stageStorage->data, textureStorage->data, size, cudaMemcpyDeviceToHost, slot->copyStream), "cudaMemcpyAsync failed while reading texture for presentation") || !c3dCheckCUDA(cudaEventRecord(slot->copyEvent, slot->copyStream), "cudaEventRecord failed while queuing presentation copy"))
  {
    std::unique_lock<std::mutex> lock(swapchain->lock);
    slot->state = C3D_SWAPCHAIN_SLOT_FREE;
    swapchain->acquiredIndex = SIZE_MAX;
    return false;
  }

  if (swapchain->info.presentMode == C3D_PRESENT_MODE_IMMEDIATE)
  {
    {
      std::unique_lock<std::mutex> lock(swapchain->lock);
      slot->sequence = ++swapchain->nextSequence;
      slot->state = C3D_SWAPCHAIN_SLOT_COPYING;
      swapchain->acquiredIndex = SIZE_MAX;
    }
    return true;
  }

  {
    std::unique_lock<std::mutex> lock(swapchain->lock);
    slot->sequence = ++swapchain->nextSequence;
    slot->state = C3D_SWAPCHAIN_SLOT_COPYING;
    swapchain->acquiredIndex = SIZE_MAX;
  }
  return true;
}
