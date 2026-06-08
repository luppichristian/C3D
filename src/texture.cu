#include <C3D.h>

#include <cuda_runtime.h>

struct C3DTexture {
  C3DTextureInfo info;
  void* data;
  size_t size;
};

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

  if (info->width == 0 || info->height == 0 || info->depth == 0) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture dimensions must be greater than zero");
    return false;
  }

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

  size_t volume_size = plane_size * info->depth;
  if (volume_size / plane_size != info->depth) {
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

  c3dThrowError(C3D_ERROR_CUDA, desc);
  return false;
}

C3D_API C3DTexture* c3dCreateTexture(const C3DTextureInfo* info) {
  size_t size = 0;
  if (!c3dTryGetTextureSize(info, &size)) {
    return nullptr;
  }

  C3DTexture* texture = new C3DTexture {};
  if (!texture) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture object");
    return nullptr;
  }

  texture->info = *info;
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

C3D_API bool c3dGetTextureInfo(C3DTexture* texture, C3DTextureInfo* info) {
  if (!texture || !info) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture and info output must be non-null");
    return false;
  }

  *info = texture->info;
  return true;
}
