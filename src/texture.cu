#include <C3D.h>

#include "c3d_internal.h"

#include <cuda_runtime.h>

#include <stdio.h>
#include <stdint.h>
#include <string.h>

static thread_local char g_cuda_error_desc[512];

static size_t c3dGetFormatSize(C3DTextureFormat format) {
  switch (format) {
    case C3D_TEXTURE_FORMAT_RGBA8:
    case C3D_TEXTURE_FORMAT_BGRA8:
      return 4;
  }

  return 0;
}

static bool c3dTryGetTextureSize(const C3DTextureInfo* info, size_t* size) {
  if (!info || !size) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture info and size output must be non-null");
    return false;
  }

  if (info->width == 0 || info->height == 0) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture width and height must be greater than zero");
    return false;
  }

  size_t depth = info->depth == 0 ? 1 : info->depth;

  size_t format_size = c3dGetFormatSize(info->format);
  if (format_size == 0) {
    c3dThrowError(C3D_ERROR_UNSUPPORTED_FORMAT, "texture format is not supported");
    return false;
  }

  size_t plane_size = info->width * info->height;
  if (plane_size / info->width != info->height) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture dimensions overflow size_t");
    return false;
  }

  size_t volume_size = plane_size * depth;
  if (volume_size / plane_size != depth) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture dimensions overflow size_t");
    return false;
  }

  size_t total_size = volume_size * format_size;
  if (total_size / volume_size != format_size) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture size overflows size_t");
    return false;
  }

  *size = total_size;
  return true;
}

static C3DTextureInfo c3dNormalizeTextureInfo(C3DTextureInfo info) {
  if (info.depth == 0) {
    info.depth = 1;
  }

  return info;
}

static bool c3dCheckTextureRange(C3DTexture* texture, size_t offset, size_t size, void* buffer, const char* operation) {
  if (!texture) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  if (!buffer && size != 0) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer must be non-null when size is non-zero");
    return false;
  }

  if (offset > texture->size || size > texture->size - offset) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, operation);
    return false;
  }

  return true;
}

static bool c3dCheckCUDA(cudaError_t error, const char* desc) {
  if (error == cudaSuccess) {
    return true;
  }

  snprintf(g_cuda_error_desc, sizeof(g_cuda_error_desc), "%s: %s (%d)", desc, cudaGetErrorString(error), (int)error);
  c3dThrowError(C3D_ERROR_CUDA, g_cuda_error_desc);
  return false;
}

static __global__ void c3dFillTextureKernel(uint32_t* pixels, size_t count, uint32_t value) {
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index < count) {
    pixels[index] = value;
  }
}

static bool c3dTryGetFillTextureArgs(C3DTexture* texture, size_t offset, size_t size, void* texel, uint32_t* value, size_t* texel_offset, size_t* texel_count) {
  if (!c3dCheckTextureRange(texture, offset, size, texel, "texture fill range is out of bounds")) {
    return false;
  }

  if (size == 0) {
    *texel_count = 0;
    return true;
  }

  size_t format_size = c3dGetFormatSize(texture->info.format);
  if (format_size != sizeof(uint32_t)) {
    c3dThrowError(C3D_ERROR_UNSUPPORTED_FORMAT, "texture fill only supports 32-bit texel formats");
    return false;
  }

  if ((offset % format_size) != 0 || (size % format_size) != 0) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture fill offset and size must be texel-aligned");
    return false;
  }

  memcpy(value, texel, sizeof(*value));
  *texel_offset = offset / format_size;
  *texel_count = size / format_size;
  return true;
}

C3D_API C3DTexture* c3dCreateTexture(const C3DTextureInfo* info) {
  size_t size = 0;
  if (!c3dTryGetTextureSize(info, &size)) {
    return nullptr;
  }

  C3DTextureInfo normalized_info = c3dNormalizeTextureInfo(*info);

  C3DTexture* texture = new C3DTexture {};
  if (!texture) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture object");
    return nullptr;
  }

  texture->info = normalized_info;
  texture->size = size;

  if (!c3dCheckCUDA(cudaMalloc(&texture->data, texture->size), "cudaMalloc failed while creating texture")) {
    delete texture;
    return nullptr;
  }

  return texture;
}

C3D_API bool c3dDeleteTexture(C3DTexture* texture) {
  if (!texture) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  bool success = c3dCheckCUDA(cudaFree(texture->data), "cudaFree failed while deleting texture");
  delete texture;
  return success;
}

C3D_API bool c3dReadTexture(C3DTexture* texture, size_t offset, size_t size, void* buffer) {
  if (!c3dCheckTextureRange(texture, offset, size, buffer, "texture read range is out of bounds")) {
    return false;
  }

  if (size == 0) {
    return true;
  }

  const char* source = static_cast<const char*>(texture->data) + offset;
  return c3dCheckCUDA(cudaMemcpy(buffer, source, size, cudaMemcpyDeviceToHost), "cudaMemcpy failed while reading texture");
}

C3D_API bool c3dWriteTexture(C3DTexture* texture, size_t offset, size_t size, void* buffer) {
  if (!c3dCheckTextureRange(texture, offset, size, buffer, "texture write range is out of bounds")) {
    return false;
  }

  if (size == 0) {
    return true;
  }

  char* destination = static_cast<char*>(texture->data) + offset;
  return c3dCheckCUDA(cudaMemcpy(destination, buffer, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing texture");
}

C3D_API bool c3dFillTexture(C3DTexture* texture, size_t offset, size_t size, void* texel) {
  uint32_t value = 0;
  size_t texel_offset = 0;
  size_t texel_count = 0;
  if (!c3dTryGetFillTextureArgs(texture, offset, size, texel, &value, &texel_offset, &texel_count)) {
    return false;
  }

  if (texel_count == 0) {
    return true;
  }

  const int threads_per_block = 256;
  const int block_count = (int)((texel_count + (size_t)threads_per_block - 1) / (size_t)threads_per_block);
  uint32_t* destination = static_cast<uint32_t*>(texture->data) + texel_offset;
  c3dFillTextureKernel<<<block_count, threads_per_block>>>(destination, texel_count, value);
  return c3dCheckCUDA(cudaPeekAtLastError(), "texture fill kernel launch failed")
      && c3dCheckCUDA(cudaDeviceSynchronize(), "texture fill kernel execution failed");
}

C3D_API bool c3dClearTexture(C3DTexture* texture, void* texel) {
  if (!texture) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  return c3dFillTexture(texture, 0, texture->size, texel);
}

C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info) {
  if (!texture || !info) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture and info output must be non-null");
    return false;
  }

  *info = texture->info;
  return true;
}

C3D_API bool c3dResizeTexture(C3DTexture* texture, size_t width, size_t height, size_t depth) {
  if (!texture) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture must be non-null");
    return false;
  }

  C3DTextureInfo info = texture->info;
  info.width = width;
  info.height = height;
  info.depth = depth;

  size_t new_size = 0;
  if (!c3dTryGetTextureSize(&info, &new_size)) {
    return false;
  }

  info = c3dNormalizeTextureInfo(info);

  void* new_data = nullptr;
  if (!c3dCheckCUDA(cudaMalloc(&new_data, new_size), "cudaMalloc failed while resizing texture")) {
    return false;
  }

  size_t copy_size = texture->size < new_size ? texture->size : new_size;
  bool success = true;
  if (copy_size != 0) {
    success = c3dCheckCUDA(cudaMemcpy(new_data, texture->data, copy_size, cudaMemcpyDeviceToDevice), "cudaMemcpy failed while resizing texture");
  }

  if (!success) {
    cudaFree(new_data);
    return false;
  }

  if (!c3dCheckCUDA(cudaFree(texture->data), "cudaFree failed while replacing texture storage during resize")) {
    cudaFree(new_data);
    return false;
  }

  texture->data = new_data;
  texture->size = new_size;
  texture->info = info;
  return true;
}
