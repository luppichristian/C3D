#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c3d_internal.h"

static bool c3dTryGetTextureSize(const C3DTextureInfo* info, size_t* size)
{
  if (!info || !size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture info and size output must be non-null");
    return false;
  }

  if (info->width == 0 || info->height == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture width and height must be greater than zero");
    return false;
  }

  size_t depth = info->depth == 0 ? 1 : info->depth;
  size_t formatSize = c3dGetTextureFormatSize(info->format);
  if (formatSize == 0)
  {
    c3dThrowError(C3D_ERROR_UNSUPPORTED_FORMAT, "texture format is not supported");
    return false;
  }

  size_t planeSize = info->width * info->height;
  if (planeSize / info->width != info->height)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture dimensions overflow size_t");
    return false;
  }

  size_t volumeSize = planeSize * depth;
  if (volumeSize / planeSize != depth)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture dimensions overflow size_t");
    return false;
  }

  size_t totalSize = volumeSize * formatSize;
  if (totalSize / volumeSize != formatSize)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture size overflows size_t");
    return false;
  }

  *size = totalSize;
  return true;
}

static C3DTextureInfo c3dNormalizeTextureInfo(C3DTextureInfo info)
{
  if (info.depth == 0)
  {
    info.depth = 1;
  }

  return info;
}

static bool c3dCheckTextureRange(C3DTexture* texture, size_t offset, size_t size, const char* desc)
{
  if (!texture)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  return c3dCheckRange(texture->size, offset, size, desc);
}

static bool c3dCheckStageRange(C3DStageBuffer* stageBuffer, size_t offset, size_t size, const char* desc)
{
  if (!stageBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "stage buffer must be non-null");
    return false;
  }

  return c3dCheckRange(stageBuffer->info.size, offset, size, desc);
}

static __global__ void c3dFillTextureKernel(uint32_t* pixels, size_t count, uint32_t value)
{
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index < count)
  {
    pixels[index] = value;
  }
}

static bool c3dTryGetFillTextureArgs(
    C3DTexture* texture,
    size_t offset,
    size_t size,
    void* texel,
    uint32_t* value,
    size_t* texelOffset,
    size_t* texelCount)
{
  if (!texel && size != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texel must be non-null when size is non-zero");
    return false;
  }

  if (!c3dCheckTextureRange(texture, offset, size, "texture fill range is out of bounds"))
  {
    return false;
  }

  if (size == 0)
  {
    *texelCount = 0;
    return true;
  }

  size_t formatSize = c3dGetTextureFormatSize(texture->info.format);
  if (formatSize != sizeof(uint32_t))
  {
    c3dThrowError(C3D_ERROR_UNSUPPORTED_FORMAT, "texture fill only supports 32-bit texel formats");
    return false;
  }

  if ((offset % formatSize) != 0 || (size % formatSize) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture fill offset and size must be texel-aligned");
    return false;
  }

  memcpy(value, texel, sizeof(*value));
  *texelOffset = offset / formatSize;
  *texelCount = size / formatSize;
  return true;
}

static bool c3dWaitForTextureStorage(C3DTextureStorage* storage)
{
  return !storage || c3dWaitForSubmission(storage->lastBoundSubmission);
}

static void c3dDestroyTextureStorageChain(C3DTextureStorage* storage)
{
  while (storage)
  {
    C3DTextureStorage* next = storage->next;
    c3dWaitForTextureStorage(storage);
    if (storage->data)
    {
      cudaFree(storage->data);
    }

    free(storage);
    storage = next;
  }
}

C3D_API C3DTexture* c3dCreateTexture(const C3DTextureInfo* info)
{
  size_t size = 0;
  if (!c3dTryGetTextureSize(info, &size))
  {
    return nullptr;
  }

  C3DTexture* texture = (C3DTexture*)malloc(sizeof(C3DTexture));
  if (!texture)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture object");
    return nullptr;
  }

  memset(texture, 0, sizeof(*texture));
  texture->info = c3dNormalizeTextureInfo(*info);
  texture->size = size;
  if (!c3dEnsureTextureStorageForWrite(texture, true, &texture->current))
  {
    free(texture);
    return nullptr;
  }

  if (!texture->current)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to initialize texture storage");
    free(texture);
    return nullptr;
  }

  texture->storages = texture->current;
  return texture;
}

C3D_API bool c3dDeleteTexture(C3DTexture* texture)
{
  if (!texture)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  c3dDestroyTextureStorageChain(texture->storages);
  free(texture);
  return true;
}

C3D_API bool c3dReadTexture(C3DTexture* texture, size_t textureOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset)
{
  if (!c3dCheckTextureRange(texture, textureOffset, size, "texture read range is out of bounds")
      || !c3dCheckStageRange(stageBuffer, stageOffset, size, "stage buffer write range is out of bounds"))
  {
    return false;
  }

  C3DTextureStorage* textureStorage = c3dGetCurrentTextureStorage(texture);
  C3DStageBufferStorage* stageStorage = c3dGetCurrentStageStorage(stageBuffer);
  if (!c3dWaitForTextureStorage(textureStorage) || !c3dWaitForSubmission(stageStorage ? stageStorage->lastBoundSubmission : 0))
  {
    return false;
  }

  if (size == 0)
  {
    return true;
  }

  return c3dCheckCUDA(cudaMemcpy(stageStorage->data + stageOffset, textureStorage->data + textureOffset, size, cudaMemcpyDeviceToHost), "cudaMemcpy failed while reading texture");
}

C3D_API bool c3dWriteTexture(C3DTexture* texture, size_t textureOffset, size_t size, C3DStageBuffer* stageBuffer, size_t stageOffset, bool cycle)
{
  if (!c3dCheckTextureRange(texture, textureOffset, size, "texture write range is out of bounds")
      || !c3dCheckStageRange(stageBuffer, stageOffset, size, "stage buffer read range is out of bounds"))
  {
    return false;
  }

  C3DTextureStorage* textureStorage = nullptr;
  if (!c3dEnsureTextureStorageForWrite(texture, cycle, &textureStorage))
  {
    return false;
  }

  C3DStageBufferStorage* stageStorage = c3dGetCurrentStageStorage(stageBuffer);
  if (!c3dWaitForSubmission(stageStorage ? stageStorage->lastBoundSubmission : 0))
  {
    return false;
  }

  if (size == 0)
  {
    return true;
  }

  return c3dCheckCUDA(cudaMemcpy(textureStorage->data + textureOffset, stageStorage->data + stageOffset, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing texture");
}

C3D_API bool c3dFillTexture(C3DTexture* texture, size_t offset, size_t size, void* texel, bool cycle)
{
  uint32_t value = 0;
  size_t texelOffset = 0;
  size_t texelCount = 0;
  if (!c3dTryGetFillTextureArgs(texture, offset, size, texel, &value, &texelOffset, &texelCount))
  {
    return false;
  }

  C3DTextureStorage* storage = nullptr;
  if (!c3dEnsureTextureStorageForWrite(texture, cycle, &storage))
  {
    return false;
  }

  if (texelCount == 0)
  {
    return true;
  }

  const int threadsPerBlock = 256;
  const int blockCount = (int)((texelCount + (size_t)threadsPerBlock - 1) / (size_t)threadsPerBlock);
  c3dFillTextureKernel<<<blockCount, threadsPerBlock>>>((uint32_t*)storage->data + texelOffset, texelCount, value);
  return c3dCheckCUDA(cudaPeekAtLastError(), "texture fill kernel launch failed") && c3dCheckCUDA(cudaDeviceSynchronize(), "texture fill kernel execution failed");
}

C3D_API bool c3dClearTexture(C3DTexture* texture, void* texel, bool cycle)
{
  if (!texture)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  return c3dFillTexture(texture, 0, texture->size, texel, cycle);
}

C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info)
{
  if (!texture || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture and info output must be non-null");
    return false;
  }

  *info = texture->info;
  return true;
}

C3D_API bool c3dResizeTexture(C3DTexture* texture, size_t width, size_t height, size_t depth)
{
  if (!texture)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  C3DTextureInfo info = texture->info;
  info.width = width;
  info.height = height;
  info.depth = depth;

  size_t newSize = 0;
  if (!c3dTryGetTextureSize(&info, &newSize))
  {
    return false;
  }

  info = c3dNormalizeTextureInfo(info);
  c3dDestroyTextureStorageChain(texture->storages);
  texture->storages = nullptr;
  texture->current = nullptr;
  texture->info = info;
  texture->size = newSize;

  if (!c3dEnsureTextureStorageForWrite(texture, true, &texture->current))
  {
    return false;
  }

  texture->storages = texture->current;
  return true;
}
