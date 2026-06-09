#include <C3D.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c3d_internal.h"

static size_t c3dGetIndexStride(C3DIndexSize index_size)
{
  switch (index_size)
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

static bool c3dTryMultiplySize(size_t a, size_t b, size_t* result, const char* desc)
{
  if (!result)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "size output must be non-null");
    return false;
  }

  if (a != 0 && b > (SIZE_MAX / a))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, desc);
    return false;
  }

  *result = a * b;
  return true;
}

static bool c3dCheckRange(size_t total_size, size_t offset, size_t size, void* buffer, const char* range_desc)
{
  if (!buffer && size != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "buffer value must be non-null when size is non-zero");
    return false;
  }

  if (offset > total_size || size > total_size - offset)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, range_desc);
    return false;
  }

  return true;
}

static bool c3dTryResizeBufferPair(uint8_t** host_data, uint8_t** device_data, size_t old_size, size_t new_size, const char* desc)
{
  uint8_t* new_host_data = nullptr;
  uint8_t* new_device_data = nullptr;
  if (new_size != 0)
  {
    new_host_data = (uint8_t*)malloc(new_size);
    if (!new_host_data)
    {
      c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, desc);
      return false;
    }

    if (!c3dCheckCUDA(cudaMalloc((void**)&new_device_data, new_size), desc))
    {
      free(new_host_data);
      return false;
    }

    size_t copy_size = old_size < new_size ? old_size : new_size;
    if (*host_data && copy_size != 0)
    {
      memcpy(new_host_data, *host_data, copy_size);
    }

    if (new_size > copy_size)
    {
      memset(new_host_data + copy_size, 0, new_size - copy_size);
    }

    if (!c3dCheckCUDA(cudaMemcpy(new_device_data, new_host_data, new_size, cudaMemcpyHostToDevice), "cudaMemcpy failed while updating buffer device storage"))
    {
      free(new_host_data);
      cudaFree(new_device_data);
      return false;
    }
  }

  free(*host_data);
  if (*device_data && !c3dCheckCUDA(cudaFree(*device_data), "cudaFree failed while replacing buffer device storage"))
  {
    free(new_host_data);
    if (new_device_data)
    {
      cudaFree(new_device_data);
    }

    return false;
  }

  *host_data = new_host_data;
  *device_data = new_device_data;
  return true;
}

static bool c3dTryGetIndexBufferSize(const C3DIndexBufferInfo* info, size_t* size)
{
  if (!info || !size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer info and size output must be non-null");
    return false;
  }

  size_t stride = c3dGetIndexStride(info->indexSize);
  if (stride == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index size is invalid");
    return false;
  }

  return c3dTryMultiplySize(info->indexCap, stride, size, "index buffer size overflows size_t");
}

static bool c3dTryGetVertexBufferSize(const C3DVertexBufferInfo* info, size_t* size)
{
  if (!info || !size)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer info and size output must be non-null");
    return false;
  }

  return c3dTryMultiplySize(info->vertexCap, sizeof(C3DVertex), size, "vertex buffer size overflows size_t");
}

static bool c3dCheckIndexBufferRange(C3DIndexBuffer* index_buffer, size_t offset, size_t size, void* buffer, const char* range_desc)
{
  if (!index_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer must be non-null");
    return false;
  }

  return c3dCheckRange(index_buffer->size, offset, size, buffer, range_desc);
}

static bool c3dCheckVertexBufferRange(C3DVertexBuffer* vertex_buffer, size_t offset, size_t size, void* buffer, const char* range_desc)
{
  if (!vertex_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer must be non-null");
    return false;
  }

  return c3dCheckRange(vertex_buffer->size, offset, size, buffer, range_desc);
}

static bool c3dFillBytes(uint8_t* destination, size_t offset, size_t size, const void* value, size_t stride, const char* alignment_desc)
{
  if (size == 0)
  {
    return true;
  }

  if ((offset % stride) != 0 || (size % stride) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, alignment_desc);
    return false;
  }

  uint8_t* write_ptr = destination + offset;
  size_t count = size / stride;
  for (size_t i = 0; i < count; ++i)
  {
    memcpy(write_ptr + (i * stride), value, stride);
  }

  return true;
}

C3D_API C3DIndexBuffer* c3dCreateIndexBuffer(const C3DIndexBufferInfo* info)
{
  size_t size = 0;
  if (!c3dTryGetIndexBufferSize(info, &size))
  {
    return nullptr;
  }

  C3DIndexBuffer* index_buffer = (C3DIndexBuffer*)malloc(sizeof(C3DIndexBuffer));
  if (!index_buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate index buffer object");
    return nullptr;
  }

  memset(index_buffer, 0, sizeof(C3DIndexBuffer));
  index_buffer->info = *info;
  index_buffer->size = size;
  if (!c3dTryResizeBufferPair(&index_buffer->data, &index_buffer->device_data, 0, size, "failed to allocate index buffer storage"))
  {
    free(index_buffer);
    return nullptr;
  }

  return index_buffer;
}

C3D_API bool c3dDeleteIndexBuffer(C3DIndexBuffer* index_buffer)
{
  if (!index_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer must be non-null");
    return false;
  }

  free(index_buffer->data);
  if (index_buffer->device_data && !c3dCheckCUDA(cudaFree(index_buffer->device_data), "cudaFree failed while deleting index buffer"))
  {
    free(index_buffer);
    return false;
  }

  free(index_buffer);
  return true;
}

C3D_API bool c3dResizeIndexBuffer(C3DIndexBuffer* index_buffer, size_t index_cap)
{
  if (!index_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer must be non-null");
    return false;
  }

  C3DIndexBufferInfo info = index_buffer->info;
  info.indexCap = index_cap;

  size_t new_size = 0;
  if (!c3dTryGetIndexBufferSize(&info, &new_size))
  {
    return false;
  }

  if (!c3dTryResizeBufferPair(&index_buffer->data, &index_buffer->device_data, index_buffer->size, new_size, "failed to resize index buffer storage"))
  {
    return false;
  }

  index_buffer->info = info;
  index_buffer->size = new_size;
  return true;
}

C3D_API bool c3dGetIndexBufferInfo(C3DIndexBuffer* index_buffer, C3DIndexBufferInfo* info)
{
  if (!index_buffer || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer and info output must be non-null");
    return false;
  }

  *info = index_buffer->info;
  return true;
}

C3D_API bool c3dReadIndexBuffer(C3DIndexBuffer* index_buffer, size_t offset, size_t size, void* buffer)
{
  if (!c3dCheckIndexBufferRange(index_buffer, offset, size, buffer, "index buffer read range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(buffer, index_buffer->data + offset, size);
  }

  return true;
}

C3D_API bool c3dWriteIndexBuffer(C3DIndexBuffer* index_buffer, size_t offset, size_t size, void* buffer)
{
  if (!c3dCheckIndexBufferRange(index_buffer, offset, size, buffer, "index buffer write range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(index_buffer->data + offset, buffer, size);
    if (!c3dCheckCUDA(cudaMemcpy(index_buffer->device_data + offset, buffer, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing index buffer"))
    {
      return false;
    }
  }

  return true;
}

C3D_API bool c3dFillIndexBuffer(C3DIndexBuffer* index_buffer, size_t offset, size_t size, void* idx)
{
  if (!c3dCheckIndexBufferRange(index_buffer, offset, size, idx, "index buffer fill range is out of bounds"))
  {
    return false;
  }

  size_t stride = c3dGetIndexStride(index_buffer->info.indexSize);
  if (!c3dFillBytes(index_buffer->data, offset, size, idx, stride, "index buffer fill offset and size must be index-aligned"))
  {
    return false;
  }

  return size == 0 || c3dCheckCUDA(cudaMemcpy(index_buffer->device_data + offset, index_buffer->data + offset, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while filling index buffer");
}

C3D_API bool c3dClearIndexBuffer(C3DIndexBuffer* index_buffer, void* idx)
{
  if (!index_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "index buffer must be non-null");
    return false;
  }

  return c3dFillIndexBuffer(index_buffer, 0, index_buffer->size, idx);
}

C3D_API C3DVertexBuffer* c3dCreateVertexBuffer(const C3DVertexBufferInfo* info)
{
  size_t size = 0;
  if (!c3dTryGetVertexBufferSize(info, &size))
  {
    return nullptr;
  }

  C3DVertexBuffer* vertex_buffer = (C3DVertexBuffer*)malloc(sizeof(C3DVertexBuffer));
  if (!vertex_buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate vertex buffer object");
    return nullptr;
  }

  memset(vertex_buffer, 0, sizeof(C3DVertexBuffer));
  vertex_buffer->info = *info;
  vertex_buffer->size = size;
  if (!c3dTryResizeBufferPair(&vertex_buffer->data, &vertex_buffer->device_data, 0, size, "failed to allocate vertex buffer storage"))
  {
    free(vertex_buffer);
    return nullptr;
  }

  return vertex_buffer;
}

C3D_API bool c3dDeleteVertexBuffer(C3DVertexBuffer* vertex_buffer)
{
  if (!vertex_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer must be non-null");
    return false;
  }

  free(vertex_buffer->data);
  if (vertex_buffer->device_data && !c3dCheckCUDA(cudaFree(vertex_buffer->device_data), "cudaFree failed while deleting vertex buffer"))
  {
    free(vertex_buffer);
    return false;
  }

  free(vertex_buffer);
  return true;
}

C3D_API bool c3dResizeVertexBuffer(C3DVertexBuffer* vertex_buffer, size_t vertex_cap)
{
  if (!vertex_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer must be non-null");
    return false;
  }

  C3DVertexBufferInfo info = vertex_buffer->info;
  info.vertexCap = vertex_cap;

  size_t new_size = 0;
  if (!c3dTryGetVertexBufferSize(&info, &new_size))
  {
    return false;
  }

  if (!c3dTryResizeBufferPair(&vertex_buffer->data, &vertex_buffer->device_data, vertex_buffer->size, new_size, "failed to resize vertex buffer storage"))
  {
    return false;
  }

  vertex_buffer->info = info;
  vertex_buffer->size = new_size;
  return true;
}

C3D_API bool c3dGetVertexBufferInfo(C3DVertexBuffer* vertex_buffer, C3DVertexBufferInfo* info)
{
  if (!vertex_buffer || !info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer and info output must be non-null");
    return false;
  }

  *info = vertex_buffer->info;
  return true;
}

C3D_API bool c3dReadVertexBuffer(C3DVertexBuffer* vertex_buffer, size_t offset, size_t size, void* buffer)
{
  if (!c3dCheckVertexBufferRange(vertex_buffer, offset, size, buffer, "vertex buffer read range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(buffer, vertex_buffer->data + offset, size);
  }

  return true;
}

C3D_API bool c3dWriteVertexBuffer(C3DVertexBuffer* vertex_buffer, size_t offset, size_t size, void* buffer)
{
  if (!c3dCheckVertexBufferRange(vertex_buffer, offset, size, buffer, "vertex buffer write range is out of bounds"))
  {
    return false;
  }

  if (size != 0)
  {
    memcpy(vertex_buffer->data + offset, buffer, size);
    if (!c3dCheckCUDA(cudaMemcpy(vertex_buffer->device_data + offset, buffer, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing vertex buffer"))
    {
      return false;
    }
  }

  return true;
}

C3D_API bool c3dFillVertexBuffer(C3DVertexBuffer* vertex_buffer, size_t offset, size_t size, void* vertex)
{
  if (!c3dCheckVertexBufferRange(vertex_buffer, offset, size, vertex, "vertex buffer fill range is out of bounds"))
  {
    return false;
  }

  if (!c3dFillBytes(vertex_buffer->data, offset, size, vertex, sizeof(C3DVertex), "vertex buffer fill offset and size must be vertex-aligned"))
  {
    return false;
  }

  return size == 0 || c3dCheckCUDA(cudaMemcpy(vertex_buffer->device_data + offset, vertex_buffer->data + offset, size, cudaMemcpyHostToDevice), "cudaMemcpy failed while filling vertex buffer");
}

C3D_API bool c3dClearVertexBuffer(C3DVertexBuffer* vertex_buffer, void* vertex)
{
  if (!vertex_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "vertex buffer must be non-null");
    return false;
  }

  return c3dFillVertexBuffer(vertex_buffer, 0, vertex_buffer->size, vertex);
}
