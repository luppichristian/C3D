#pragma once

#include <C3D.h>
#include <stdint.h>

struct C3DTexture
{
  C3DTextureInfo info;
  void* data;
  size_t size;
};

struct C3DIndexBuffer
{
  C3DIndexBufferInfo info;
  uint8_t* data;
  uint8_t* device_data;
  size_t size;
};

struct C3DVertexBuffer
{
  C3DVertexBufferInfo info;
  uint8_t* data;
  uint8_t* device_data;
  size_t size;
};

struct C3DRecordedDraw
{
  C3DDrawInfo info;
};

struct C3DCommandBuffer
{
  bool in_render_pass;
  bool has_render_pass;
  C3DRenderPassInfo render_pass;
  C3DRecordedDraw* draws;
  size_t draw_count;
  size_t draw_cap;
  uint64_t* depth_buffer;
  size_t depth_cap;
  void* line_primitives;
  size_t line_primitive_cap;
  void* triangle_primitives;
  size_t triangle_primitive_cap;
  void* texture_views;
  size_t texture_view_cap;
  uint32_t* tile_counts_device;
  uint32_t* tile_offsets_device;
  uint32_t* tile_indices_device;
  uint32_t* tile_counts_host;
  uint32_t* tile_offsets_host;
  size_t tile_count_cap;
  size_t tile_offset_cap;
  size_t tile_index_cap;
  size_t tile_counts_host_cap;
  size_t tile_offsets_host_cap;
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
