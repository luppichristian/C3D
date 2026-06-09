#include <C3D.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "c3d_internal.h"

struct C3DTextureView
{
  const uint8_t* pixels;
  size_t width;
  size_t height;
  C3DTextureFormat format;
  C3DSampler sampler;
};

struct C3DPixel
{
  float r;
  float g;
  float b;
  float a;
};

struct C3DVertexSample
{
  float x;
  float y;
  float z;
  float w;
  float r;
  float g;
  float b;
  float a;
  float u;
  float v;
  int texid;
};

struct C3DLinePrimitive
{
  C3DVertexSample a;
  C3DVertexSample b;
  int min_x;
  int min_y;
  int max_x;
  int max_y;
  uint32_t order;
  int valid;
};

struct C3DTrianglePrimitive
{
  C3DVertexSample a;
  C3DVertexSample b;
  C3DVertexSample c;
  float area;
  int min_x;
  int min_y;
  int max_x;
  int max_y;
  uint32_t order;
  int valid;
};

static const int C3D_TILE_SIZE = 16;

static bool c3dTryResizeDeviceBuffer(void** buffer, size_t* cap, size_t required_bytes, const char* desc)
{
  if (*cap >= required_bytes)
  {
    return true;
  }

  if (*buffer && !c3dCheckCUDA(cudaFree(*buffer), "cudaFree failed while replacing command buffer scratch storage"))
  {
    return false;
  }

  *buffer = nullptr;
  *cap = 0;
  if (required_bytes == 0)
  {
    return true;
  }

  if (!c3dCheckCUDA(cudaMalloc(buffer, required_bytes), desc))
  {
    return false;
  }

  *cap = required_bytes;
  return true;
}

static bool c3dTryResizeHostBuffer(uint32_t** buffer, size_t* cap, size_t count, const char* desc)
{
  if (*cap >= count)
  {
    return true;
  }

  uint32_t* new_buffer = (uint32_t*)realloc(*buffer, count * sizeof(uint32_t));
  if (!new_buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, desc);
    return false;
  }

  *buffer = new_buffer;
  *cap = count;
  return true;
}

static __host__ __device__ size_t c3dGetTextureTexelSize(C3DTextureFormat format)
{
  switch (format)
  {
    case C3D_TEXTURE_FORMAT_RGBA8:
    case C3D_TEXTURE_FORMAT_BGRA8:
      return 4;
  }

  return 0;
}

static __host__ __device__ size_t c3dGetIndexStride(C3DIndexSize index_size)
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

static __host__ __device__ float c3dClamp01(float value)
{
  if (value < 0.0f)
  {
    return 0.0f;
  }

  if (value > 1.0f)
  {
    return 1.0f;
  }

  return value;
}

static __host__ __device__ float c3dClampf(float value, float min_value, float max_value)
{
  if (value < min_value)
  {
    return min_value;
  }

  if (value > max_value)
  {
    return max_value;
  }

  return value;
}

static __host__ __device__ uint8_t c3dFloatToByte(float value)
{
  float scaled = c3dClamp01(value) * 255.0f;
  return (uint8_t)(scaled + 0.5f);
}

static __host__ __device__ uint64_t c3dEncodeDepthOrder(float depth, uint32_t order)
{
  uint32_t depth_bits = (uint32_t)(c3dClamp01(depth) * 4294967295.0f + 0.5f);
  return ((uint64_t)depth_bits << 32) | (uint64_t)order;
}

static __host__ __device__ C3DPixel c3dUnpackPixel(const uint8_t* texel, C3DTextureFormat format)
{
  C3DPixel pixel = {0};
  switch (format)
  {
    case C3D_TEXTURE_FORMAT_RGBA8:
      pixel.r = texel[0] / 255.0f;
      pixel.g = texel[1] / 255.0f;
      pixel.b = texel[2] / 255.0f;
      pixel.a = texel[3] / 255.0f;
      break;
    case C3D_TEXTURE_FORMAT_BGRA8:
      pixel.b = texel[0] / 255.0f;
      pixel.g = texel[1] / 255.0f;
      pixel.r = texel[2] / 255.0f;
      pixel.a = texel[3] / 255.0f;
      break;
  }

  return pixel;
}

static __host__ __device__ void c3dPackPixel(uint8_t* texel, C3DTextureFormat format, C3DPixel pixel)
{
  uint8_t r = c3dFloatToByte(pixel.r);
  uint8_t g = c3dFloatToByte(pixel.g);
  uint8_t b = c3dFloatToByte(pixel.b);
  uint8_t a = c3dFloatToByte(pixel.a);

  switch (format)
  {
    case C3D_TEXTURE_FORMAT_RGBA8:
      texel[0] = r;
      texel[1] = g;
      texel[2] = b;
      texel[3] = a;
      break;
    case C3D_TEXTURE_FORMAT_BGRA8:
      texel[0] = b;
      texel[1] = g;
      texel[2] = r;
      texel[3] = a;
      break;
  }
}

static __host__ __device__ C3DPixel c3dMulPixel(C3DPixel a, C3DPixel b)
{
  C3DPixel pixel = {0};
  pixel.r = a.r * b.r;
  pixel.g = a.g * b.g;
  pixel.b = a.b * b.b;
  pixel.a = a.a * b.a;
  return pixel;
}

static __host__ __device__ C3DPixel c3dBlendPixel(C3DPixel source, C3DPixel destination, C3DBlendMode blend_mode)
{
  switch (blend_mode)
  {
    case C3D_BLEND_MODE_NONE:
      return source;
    case C3D_BLEND_MODE_NORMAL:
    {
      C3DPixel pixel = {0};
      float inv_alpha = 1.0f - source.a;
      pixel.r = source.r + (destination.r * inv_alpha);
      pixel.g = source.g + (destination.g * inv_alpha);
      pixel.b = source.b + (destination.b * inv_alpha);
      pixel.a = source.a + (destination.a * inv_alpha);
      return pixel;
    }
    case C3D_BLEND_MODE_ADDITIVE:
    {
      C3DPixel pixel = {0};
      pixel.r = c3dClamp01(destination.r + (source.r * source.a));
      pixel.g = c3dClamp01(destination.g + (source.g * source.a));
      pixel.b = c3dClamp01(destination.b + (source.b * source.a));
      pixel.a = c3dClamp01(destination.a + source.a);
      return pixel;
    }
  }

  return source;
}

static __device__ uint8_t* c3dGetTargetTexel(C3DTextureInfo target_info, uint8_t* pixels, size_t x, size_t y)
{
  size_t texel_size = c3dGetTextureTexelSize(target_info.format);
  return pixels + (((y * target_info.width) + x) * texel_size);
}

static __host__ __device__ float c3dWrapCoord(float value)
{
  float wrapped = value - floorf(value);
  if (wrapped < 0.0f)
  {
    wrapped += 1.0f;
  }

  return wrapped;
}

static __host__ __device__ float c3dClampCoord(float value)
{
  if (value < 0.0f)
  {
    return 0.0f;
  }

  if (value > 1.0f)
  {
    return 1.0f;
  }

  return value;
}

static __host__ __device__ float c3dApplySamplerAddress(float value, C3DSampler sampler)
{
  switch (sampler)
  {
    case C3D_SAMPLER_POINT_WRAP:
    case C3D_SAMPLER_LINEAR_WRAP:
      return c3dWrapCoord(value);
    case C3D_SAMPLER_POINT_CLAMP:
    case C3D_SAMPLER_LINEAR_CLAMP:
      return c3dClampCoord(value);
  }

  return value;
}

static __host__ __device__ size_t c3dRoundToIndex(float value, size_t limit)
{
  if (limit == 0)
  {
    return 0;
  }

  if (value <= 0.0f)
  {
    return 0;
  }

  size_t index = (size_t)(value + 0.5f);
  size_t max_index = limit - 1;
  return index > max_index ? max_index : index;
}

static __device__ C3DPixel c3dSampleTextureNearest(const C3DTextureView* texture, float u, float v)
{
  float sample_u = c3dApplySamplerAddress(u, texture->sampler);
  float sample_v = c3dApplySamplerAddress(v, texture->sampler);
  float x = sample_u * (float)(texture->width - 1);
  float y = sample_v * (float)(texture->height - 1);
  size_t xi = c3dRoundToIndex(x, texture->width);
  size_t yi = c3dRoundToIndex(y, texture->height);
  const uint8_t* texel = texture->pixels + (((yi * texture->width) + xi) * 4);
  return c3dUnpackPixel(texel, texture->format);
}

static __host__ __device__ C3DPixel c3dLerpPixel(C3DPixel a, C3DPixel b, float t)
{
  C3DPixel pixel = {0};
  pixel.r = a.r + ((b.r - a.r) * t);
  pixel.g = a.g + ((b.g - a.g) * t);
  pixel.b = a.b + ((b.b - a.b) * t);
  pixel.a = a.a + ((b.a - a.a) * t);
  return pixel;
}

static __device__ C3DPixel c3dSampleTextureLinear(const C3DTextureView* texture, float u, float v)
{
  float sample_u = c3dApplySamplerAddress(u, texture->sampler);
  float sample_v = c3dApplySamplerAddress(v, texture->sampler);
  float x = sample_u * (float)(texture->width - 1);
  float y = sample_v * (float)(texture->height - 1);

  size_t x0 = c3dRoundToIndex(floorf(x), texture->width);
  size_t y0 = c3dRoundToIndex(floorf(y), texture->height);
  size_t x1 = c3dRoundToIndex(floorf(x) + 1.0f, texture->width);
  size_t y1 = c3dRoundToIndex(floorf(y) + 1.0f, texture->height);

  float tx = x - floorf(x);
  float ty = y - floorf(y);

  const uint8_t* texel00 = texture->pixels + (((y0 * texture->width) + x0) * 4);
  const uint8_t* texel10 = texture->pixels + (((y0 * texture->width) + x1) * 4);
  const uint8_t* texel01 = texture->pixels + (((y1 * texture->width) + x0) * 4);
  const uint8_t* texel11 = texture->pixels + (((y1 * texture->width) + x1) * 4);

  C3DPixel top = c3dLerpPixel(c3dUnpackPixel(texel00, texture->format), c3dUnpackPixel(texel10, texture->format), tx);
  C3DPixel bottom = c3dLerpPixel(c3dUnpackPixel(texel01, texture->format), c3dUnpackPixel(texel11, texture->format), tx);
  return c3dLerpPixel(top, bottom, ty);
}

static __device__ C3DPixel c3dSampleBoundTexture(const C3DTextureView* texture_views, size_t texture_count, int texid, float u, float v)
{
  if (texid < 0 || (size_t)texid >= texture_count)
  {
    C3DPixel white = {1.0f, 1.0f, 1.0f, 1.0f};
    return white;
  }

  const C3DTextureView* texture = &texture_views[texid];
  switch (texture->sampler)
  {
    case C3D_SAMPLER_POINT_CLAMP:
    case C3D_SAMPLER_POINT_WRAP:
      return c3dSampleTextureNearest(texture, u, v);
    case C3D_SAMPLER_LINEAR_CLAMP:
    case C3D_SAMPLER_LINEAR_WRAP:
      return c3dSampleTextureLinear(texture, u, v);
  }

  C3DPixel white = {1.0f, 1.0f, 1.0f, 1.0f};
  return white;
}

static __host__ __device__ size_t c3dReadIndexValueRaw(const uint8_t* index_data, C3DIndexSize index_size, size_t index)
{
  const uint8_t* ptr = index_data + (index * c3dGetIndexStride(index_size));
  switch (index_size)
  {
    case C3D_INDEX_SIZE_8:
      return (size_t)(*(const uint8_t*)ptr);
    case C3D_INDEX_SIZE_16:
      return (size_t)(*(const uint16_t*)ptr);
    case C3D_INDEX_SIZE_32:
      return (size_t)(*(const uint32_t*)ptr);
  }

  return 0;
}

static bool c3dValidateDrawRanges(const C3DDrawInfo* draw_info)
{
  size_t index_stride = c3dGetIndexStride(draw_info->indexBuffer->info.indexSize);
  size_t index_start = draw_info->indexOffset;
  size_t index_bytes = draw_info->count * index_stride;
  if (index_start > draw_info->indexBuffer->size || index_bytes > draw_info->indexBuffer->size - index_start)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index range is out of bounds");
    return false;
  }

  size_t first_index = draw_info->indexOffset / index_stride;
  size_t vertex_limit = draw_info->vertexBuffer->info.vertexCap;
  for (size_t i = 0; i < draw_info->count; ++i)
  {
    size_t index = c3dReadIndexValueRaw(draw_info->indexBuffer->data, draw_info->indexBuffer->info.indexSize, first_index + i);
    size_t vertex_index = draw_info->vertexOffset + draw_info->indexBase + index;
    if (vertex_index >= vertex_limit)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw vertex range is out of bounds");
      return false;
    }
  }

  return true;
}

static __device__ C3DVertexSample c3dLoadVertexRaw(
    const uint8_t* index_data,
    C3DIndexSize index_size,
    size_t first_index,
    size_t draw_index,
    size_t index_base,
    const C3DVertex* vertex_data,
    size_t vertex_offset)
{
  size_t index = c3dReadIndexValueRaw(index_data, index_size, first_index + draw_index);
  const C3DVertex* vertex = vertex_data + vertex_offset + index_base + index;

  C3DVertexSample sample = {0};
  sample.x = vertex->pos[0];
  sample.y = vertex->pos[1];
  sample.z = vertex->pos[2];
  sample.w = vertex->pos[3];
  sample.r = vertex->col[0];
  sample.g = vertex->col[1];
  sample.b = vertex->col[2];
  sample.a = vertex->col[3];
  sample.u = vertex->uv[0];
  sample.v = vertex->uv[1];
  sample.texid = vertex->texid;
  return sample;
}

static __host__ __device__ float c3dSampleDepth(const C3DVertexSample* vertex)
{
  float inv_w = vertex->w != 0.0f ? (1.0f / vertex->w) : 1.0f;
  return c3dClamp01((vertex->z * inv_w * 0.5f) + 0.5f);
}

static __host__ __device__ void c3dNdcToScreen(C3DTextureInfo target_info, C3DVertexSample* vertex)
{
  float inv_w = vertex->w != 0.0f ? (1.0f / vertex->w) : 1.0f;
  float x = vertex->x * inv_w;
  float y = vertex->y * inv_w;
  vertex->x = ((x * 0.5f) + 0.5f) * (float)(target_info.width - 1);
  vertex->y = (1.0f - ((y * 0.5f) + 0.5f)) * (float)(target_info.height - 1);
}

static __device__ void c3dWritePixel(C3DTextureInfo target_info, uint8_t* pixels, size_t x, size_t y, C3DPixel color, C3DBlendMode blend_mode)
{
  uint8_t* texel = c3dGetTargetTexel(target_info, pixels, x, y);
  C3DPixel destination = c3dUnpackPixel(texel, target_info.format);
  C3DPixel blended = c3dBlendPixel(color, destination, blend_mode);
  c3dPackPixel(texel, target_info.format, blended);
}

static __device__ void c3dShadePixel(C3DTextureInfo target_info, uint8_t* pixels, C3DBlendMode blend_mode, const C3DTextureView* texture_views, size_t texture_count, size_t x, size_t y, const C3DVertexSample* sample)
{
  C3DPixel vertex_color = {sample->r, sample->g, sample->b, sample->a};
  C3DPixel texture_color = c3dSampleBoundTexture(texture_views, texture_count, sample->texid, sample->u, sample->v);
  c3dWritePixel(target_info, pixels, x, y, c3dMulPixel(vertex_color, texture_color), blend_mode);
}

static __host__ __device__ C3DVertexSample c3dLerpVertex(const C3DVertexSample* a, const C3DVertexSample* b, float t)
{
  C3DVertexSample sample = {0};
  sample.x = a->x + ((b->x - a->x) * t);
  sample.y = a->y + ((b->y - a->y) * t);
  sample.z = a->z + ((b->z - a->z) * t);
  sample.w = a->w + ((b->w - a->w) * t);
  sample.r = a->r + ((b->r - a->r) * t);
  sample.g = a->g + ((b->g - a->g) * t);
  sample.b = a->b + ((b->b - a->b) * t);
  sample.a = a->a + ((b->a - a->a) * t);
  sample.u = a->u + ((b->u - a->u) * t);
  sample.v = a->v + ((b->v - a->v) * t);
  sample.texid = t < 0.5f ? a->texid : b->texid;
  return sample;
}

static __host__ __device__ float c3dEdge(float ax, float ay, float bx, float by, float px, float py)
{
  return ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
}

static __device__ void c3dGetTriangleDrawIndices(C3DTopology topology, size_t primitive_index, size_t* a, size_t* b, size_t* c)
{
  if (topology == C3D_TOPOLOGY_QUAD)
  {
    size_t quad = primitive_index / 2;
    size_t base = quad * 4;
    if ((primitive_index & 1) == 0)
    {
      *a = base + 0;
      *b = base + 1;
      *c = base + 2;
    }
    else
    {
      *a = base + 0;
      *b = base + 2;
      *c = base + 3;
    }
    return;
  }

  size_t base = primitive_index * 3;
  *a = base + 0;
  *b = base + 1;
  *c = base + 2;
}

static __global__ void c3dClearDepthBufferKernel(uint64_t* depth_buffer, size_t count)
{
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index < count)
  {
    depth_buffer[index] = ~0ull;
  }
}

static __global__ void c3dFinalizeTileOffsetsKernel(uint32_t* tile_offsets, const uint32_t* tile_counts, size_t tile_count)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    if (tile_count == 0)
    {
      tile_offsets[0] = 0;
      return;
    }

    tile_offsets[tile_count] = tile_offsets[tile_count - 1] + tile_counts[tile_count - 1];
  }
}

static __global__ void c3dTileScanStepKernel(const uint32_t* input, uint32_t* output, size_t count, size_t stride)
{
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index >= count)
  {
    return;
  }

  uint32_t value = input[index];
  if (index >= stride)
  {
    value += input[index - stride];
  }

  output[index] = value;
}

static __global__ void c3dTileExclusiveShiftKernel(const uint32_t* inclusive, uint32_t* exclusive, size_t count)
{
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index >= count)
  {
    return;
  }

  exclusive[index] = index == 0 ? 0u : inclusive[index - 1];
}

static __global__ void c3dSetupLinePrimitivesKernel(
    C3DLinePrimitive* primitives,
    C3DTextureInfo target_info,
    const uint8_t* index_data,
    C3DIndexSize index_size,
    size_t first_index,
    size_t index_base,
    const C3DVertex* vertex_data,
    size_t vertex_offset,
    size_t primitive_count,
    uint32_t order_base)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  C3DLinePrimitive primitive = {0};
  primitive.a = c3dLoadVertexRaw(index_data, index_size, first_index, primitive_index * 2 + 0, index_base, vertex_data, vertex_offset);
  primitive.b = c3dLoadVertexRaw(index_data, index_size, first_index, primitive_index * 2 + 1, index_base, vertex_data, vertex_offset);
  c3dNdcToScreen(target_info, &primitive.a);
  c3dNdcToScreen(target_info, &primitive.b);
  primitive.min_x = (int)floorf(fminf(primitive.a.x, primitive.b.x)) - 1;
  primitive.min_y = (int)floorf(fminf(primitive.a.y, primitive.b.y)) - 1;
  primitive.max_x = (int)ceilf(fmaxf(primitive.a.x, primitive.b.x)) + 1;
  primitive.max_y = (int)ceilf(fmaxf(primitive.a.y, primitive.b.y)) + 1;
  if (primitive.min_x < 0) primitive.min_x = 0;
  if (primitive.min_y < 0) primitive.min_y = 0;
  if (primitive.max_x >= (int)target_info.width) primitive.max_x = (int)target_info.width - 1;
  if (primitive.max_y >= (int)target_info.height) primitive.max_y = (int)target_info.height - 1;
  primitive.order = order_base + (uint32_t)primitive_index;
  primitive.valid = 1;
  primitives[primitive_index] = primitive;
}

static __global__ void c3dSetupTrianglePrimitivesKernel(
    C3DTrianglePrimitive* primitives,
    C3DTextureInfo target_info,
    C3DTopology topology,
    const uint8_t* index_data,
    C3DIndexSize index_size,
    size_t first_index,
    size_t index_base,
    const C3DVertex* vertex_data,
    size_t vertex_offset,
    size_t primitive_count,
    uint32_t order_base)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  size_t ai = 0;
  size_t bi = 0;
  size_t ci = 0;
  c3dGetTriangleDrawIndices(topology, primitive_index, &ai, &bi, &ci);

  C3DTrianglePrimitive primitive = {0};
  primitive.a = c3dLoadVertexRaw(index_data, index_size, first_index, ai, index_base, vertex_data, vertex_offset);
  primitive.b = c3dLoadVertexRaw(index_data, index_size, first_index, bi, index_base, vertex_data, vertex_offset);
  primitive.c = c3dLoadVertexRaw(index_data, index_size, first_index, ci, index_base, vertex_data, vertex_offset);
  c3dNdcToScreen(target_info, &primitive.a);
  c3dNdcToScreen(target_info, &primitive.b);
  c3dNdcToScreen(target_info, &primitive.c);
  primitive.area = c3dEdge(primitive.a.x, primitive.a.y, primitive.b.x, primitive.b.y, primitive.c.x, primitive.c.y);
  primitive.valid = primitive.area != 0.0f;
  primitive.min_x = (int)floorf(fminf(primitive.a.x, fminf(primitive.b.x, primitive.c.x)));
  primitive.min_y = (int)floorf(fminf(primitive.a.y, fminf(primitive.b.y, primitive.c.y)));
  primitive.max_x = (int)ceilf(fmaxf(primitive.a.x, fmaxf(primitive.b.x, primitive.c.x)));
  primitive.max_y = (int)ceilf(fmaxf(primitive.a.y, fmaxf(primitive.b.y, primitive.c.y)));
  if (primitive.min_x < 0) primitive.min_x = 0;
  if (primitive.min_y < 0) primitive.min_y = 0;
  if (primitive.max_x >= (int)target_info.width) primitive.max_x = (int)target_info.width - 1;
  if (primitive.max_y >= (int)target_info.height) primitive.max_y = (int)target_info.height - 1;
  primitive.order = order_base + (uint32_t)primitive_index;
  primitives[primitive_index] = primitive;
}

static __global__ void c3dBinLinePrimitivesKernel(
    const C3DLinePrimitive* primitives,
    size_t primitive_count,
    size_t tiles_x,
    uint32_t* tile_counts)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  const C3DLinePrimitive* primitive = &primitives[primitive_index];
  if (!primitive->valid)
  {
    return;
  }

  int min_tile_x = primitive->min_x / C3D_TILE_SIZE;
  int min_tile_y = primitive->min_y / C3D_TILE_SIZE;
  int max_tile_x = primitive->max_x / C3D_TILE_SIZE;
  int max_tile_y = primitive->max_y / C3D_TILE_SIZE;
  for (int tile_y = min_tile_y; tile_y <= max_tile_y; ++tile_y)
  {
    for (int tile_x = min_tile_x; tile_x <= max_tile_x; ++tile_x)
    {
      atomicAdd(&tile_counts[(tile_y * (int)tiles_x) + tile_x], 1u);
    }
  }
}

static __global__ void c3dBinTrianglePrimitivesKernel(
    const C3DTrianglePrimitive* primitives,
    size_t primitive_count,
    size_t tiles_x,
    uint32_t* tile_counts)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  const C3DTrianglePrimitive* primitive = &primitives[primitive_index];
  if (!primitive->valid)
  {
    return;
  }

  int min_tile_x = primitive->min_x / C3D_TILE_SIZE;
  int min_tile_y = primitive->min_y / C3D_TILE_SIZE;
  int max_tile_x = primitive->max_x / C3D_TILE_SIZE;
  int max_tile_y = primitive->max_y / C3D_TILE_SIZE;
  for (int tile_y = min_tile_y; tile_y <= max_tile_y; ++tile_y)
  {
    for (int tile_x = min_tile_x; tile_x <= max_tile_x; ++tile_x)
    {
      atomicAdd(&tile_counts[(tile_y * (int)tiles_x) + tile_x], 1u);
    }
  }
}

static __global__ void c3dScatterLinePrimitivesKernel(
    const C3DLinePrimitive* primitives,
    size_t primitive_count,
    size_t tiles_x,
    uint32_t* tile_offsets,
    uint32_t* tile_indices)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  const C3DLinePrimitive* primitive = &primitives[primitive_index];
  if (!primitive->valid)
  {
    return;
  }

  int min_tile_x = primitive->min_x / C3D_TILE_SIZE;
  int min_tile_y = primitive->min_y / C3D_TILE_SIZE;
  int max_tile_x = primitive->max_x / C3D_TILE_SIZE;
  int max_tile_y = primitive->max_y / C3D_TILE_SIZE;
  for (int tile_y = min_tile_y; tile_y <= max_tile_y; ++tile_y)
  {
    for (int tile_x = min_tile_x; tile_x <= max_tile_x; ++tile_x)
    {
      uint32_t tile_index = (uint32_t)((tile_y * (int)tiles_x) + tile_x);
      uint32_t offset = atomicAdd(&tile_offsets[tile_index], 1u);
      tile_indices[offset] = (uint32_t)primitive_index;
    }
  }
}

static __global__ void c3dScatterTrianglePrimitivesKernel(
    const C3DTrianglePrimitive* primitives,
    size_t primitive_count,
    size_t tiles_x,
    uint32_t* tile_offsets,
    uint32_t* tile_indices)
{
  size_t primitive_index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitive_index >= primitive_count)
  {
    return;
  }

  const C3DTrianglePrimitive* primitive = &primitives[primitive_index];
  if (!primitive->valid)
  {
    return;
  }

  int min_tile_x = primitive->min_x / C3D_TILE_SIZE;
  int min_tile_y = primitive->min_y / C3D_TILE_SIZE;
  int max_tile_x = primitive->max_x / C3D_TILE_SIZE;
  int max_tile_y = primitive->max_y / C3D_TILE_SIZE;
  for (int tile_y = min_tile_y; tile_y <= max_tile_y; ++tile_y)
  {
    for (int tile_x = min_tile_x; tile_x <= max_tile_x; ++tile_x)
    {
      uint32_t tile_index = (uint32_t)((tile_y * (int)tiles_x) + tile_x);
      uint32_t offset = atomicAdd(&tile_offsets[tile_index], 1u);
      tile_indices[offset] = (uint32_t)primitive_index;
    }
  }
}

static __global__ void c3dRasterizeLinesKernel(
    C3DTextureInfo target_info,
    uint8_t* pixels,
    uint64_t* depth_buffer,
    C3DBlendMode blend_mode,
    const C3DTextureView* texture_views,
    size_t texture_count,
    const C3DLinePrimitive* primitives,
    size_t tiles_x,
    const uint32_t* tile_offsets,
    const uint32_t* tile_indices)
{
  size_t tile_x = (size_t)blockIdx.x;
  size_t tile_y = (size_t)blockIdx.y;
  size_t x = (tile_x * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.x;
  size_t y = (tile_y * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.y;
  if (x >= target_info.width || y >= target_info.height)
  {
    return;
  }

  size_t tile_index = (tile_y * tiles_x) + tile_x;
  uint32_t start = tile_offsets[tile_index];
  uint32_t end = tile_offsets[tile_index + 1];

  size_t pixel_index = (y * target_info.width) + x;
  uint64_t best_key = depth_buffer[pixel_index];
  C3DVertexSample best_sample = {0};
  bool has_candidate = false;
  float px = (float)x + 0.5f;
  float py = (float)y + 0.5f;
  const float max_distance_sq = 0.25f;

  for (uint32_t i = start; i < end; ++i)
  {
    const C3DLinePrimitive* primitive = &primitives[tile_indices[i]];
    if (!primitive->valid)
    {
      continue;
    }

    if ((int)x < primitive->min_x || (int)x > primitive->max_x || (int)y < primitive->min_y || (int)y > primitive->max_y)
    {
      continue;
    }

    float dx = primitive->b.x - primitive->a.x;
    float dy = primitive->b.y - primitive->a.y;
    float length_sq = (dx * dx) + (dy * dy);
    float t = 0.0f;
    if (length_sq > 0.0f)
    {
      t = (((px - primitive->a.x) * dx) + ((py - primitive->a.y) * dy)) / length_sq;
      t = c3dClampf(t, 0.0f, 1.0f);
    }

    float sx = primitive->a.x + (dx * t);
    float sy = primitive->a.y + (dy * t);
    float ddx = px - sx;
    float ddy = py - sy;
    float distance_sq = (ddx * ddx) + (ddy * ddy);
    if (distance_sq > max_distance_sq)
    {
      continue;
    }

    C3DVertexSample sample = c3dLerpVertex(&primitive->a, &primitive->b, t);
    uint64_t key = c3dEncodeDepthOrder(c3dSampleDepth(&sample), primitive->order);
    if (key < best_key)
    {
      best_key = key;
      best_sample = sample;
      has_candidate = true;
    }
  }

  if (has_candidate)
  {
    depth_buffer[pixel_index] = best_key;
    c3dShadePixel(target_info, pixels, blend_mode, texture_views, texture_count, x, y, &best_sample);
  }
}

static __global__ void c3dRasterizeTrianglesKernel(
    C3DTextureInfo target_info,
    uint8_t* pixels,
    uint64_t* depth_buffer,
    C3DBlendMode blend_mode,
    const C3DTextureView* texture_views,
    size_t texture_count,
    const C3DTrianglePrimitive* primitives,
    size_t tiles_x,
    const uint32_t* tile_offsets,
    const uint32_t* tile_indices)
{
  size_t tile_x = (size_t)blockIdx.x;
  size_t tile_y = (size_t)blockIdx.y;
  size_t x = (tile_x * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.x;
  size_t y = (tile_y * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.y;
  if (x >= target_info.width || y >= target_info.height)
  {
    return;
  }

  size_t tile_index = (tile_y * tiles_x) + tile_x;
  uint32_t start = tile_offsets[tile_index];
  uint32_t end = tile_offsets[tile_index + 1];

  size_t pixel_index = (y * target_info.width) + x;
  uint64_t best_key = depth_buffer[pixel_index];
  C3DVertexSample best_sample = {0};
  bool has_candidate = false;
  float px = (float)x + 0.5f;
  float py = (float)y + 0.5f;

  for (uint32_t i = start; i < end; ++i)
  {
    const C3DTrianglePrimitive* primitive = &primitives[tile_indices[i]];
    if (!primitive->valid)
    {
      continue;
    }

    if ((int)x < primitive->min_x || (int)x > primitive->max_x || (int)y < primitive->min_y || (int)y > primitive->max_y)
    {
      continue;
    }

    float w0 = c3dEdge(primitive->b.x, primitive->b.y, primitive->c.x, primitive->c.y, px, py);
    float w1 = c3dEdge(primitive->c.x, primitive->c.y, primitive->a.x, primitive->a.y, px, py);
    float w2 = c3dEdge(primitive->a.x, primitive->a.y, primitive->b.x, primitive->b.y, px, py);
    bool inside = primitive->area > 0.0f ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
    if (!inside)
    {
      continue;
    }

    float inv_area = 1.0f / primitive->area;
    float wa = w0 * inv_area;
    float wb = w1 * inv_area;
    float wc = w2 * inv_area;
    float inv_wa = primitive->a.w != 0.0f ? (1.0f / primitive->a.w) : 1.0f;
    float inv_wb = primitive->b.w != 0.0f ? (1.0f / primitive->b.w) : 1.0f;
    float inv_wc = primitive->c.w != 0.0f ? (1.0f / primitive->c.w) : 1.0f;
    float pwa = wa * inv_wa;
    float pwb = wb * inv_wb;
    float pwc = wc * inv_wc;
    float denom = pwa + pwb + pwc;
    if (denom == 0.0f)
    {
      continue;
    }

    float inv_denom = 1.0f / denom;

    C3DVertexSample sample = {0};
    sample.z = ((primitive->a.z * pwa) + (primitive->b.z * pwb) + (primitive->c.z * pwc)) * inv_denom;
    sample.w = 1.0f / denom;
    sample.r = ((primitive->a.r * pwa) + (primitive->b.r * pwb) + (primitive->c.r * pwc)) * inv_denom;
    sample.g = ((primitive->a.g * pwa) + (primitive->b.g * pwb) + (primitive->c.g * pwc)) * inv_denom;
    sample.b = ((primitive->a.b * pwa) + (primitive->b.b * pwb) + (primitive->c.b * pwc)) * inv_denom;
    sample.a = ((primitive->a.a * pwa) + (primitive->b.a * pwb) + (primitive->c.a * pwc)) * inv_denom;
    sample.u = ((primitive->a.u * pwa) + (primitive->b.u * pwb) + (primitive->c.u * pwc)) * inv_denom;
    sample.v = ((primitive->a.v * pwa) + (primitive->b.v * pwb) + (primitive->c.v * pwc)) * inv_denom;
    sample.texid = primitive->a.texid >= 0 ? primitive->a.texid : (primitive->b.texid >= 0 ? primitive->b.texid : primitive->c.texid);

    uint64_t key = c3dEncodeDepthOrder(c3dSampleDepth(&sample), primitive->order);
    if (key < best_key)
    {
      best_key = key;
      best_sample = sample;
      has_candidate = true;
    }
  }

  if (has_candidate)
  {
    depth_buffer[pixel_index] = best_key;
    c3dShadePixel(target_info, pixels, blend_mode, texture_views, texture_count, x, y, &best_sample);
  }
}

static bool c3dBuildTextureViews(C3DCommandBuffer* command_buffer, const C3DRenderPassInfo* render_pass, C3DTextureView** texture_views)
{
  *texture_views = nullptr;
  if (render_pass->textureBindCount == 0)
  {
    return true;
  }

  C3DTextureView* host_texture_views = (C3DTextureView*)malloc(render_pass->textureBindCount * sizeof(C3DTextureView));
  if (!host_texture_views)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate host texture view metadata");
    return false;
  }

  size_t required_bytes = render_pass->textureBindCount * sizeof(C3DTextureView);
  if (!c3dTryResizeDeviceBuffer(&command_buffer->texture_views, &command_buffer->texture_view_cap, required_bytes, "cudaMalloc failed while allocating texture view metadata"))
  {
    free(host_texture_views);
    return false;
  }

  *texture_views = (C3DTextureView*)command_buffer->texture_views;

  for (size_t i = 0; i < render_pass->textureBindCount; ++i)
  {
    const C3DTextureBinding* binding = &render_pass->textureBindings[i];
    host_texture_views[i].pixels = (const uint8_t*)binding->texture->data;
    host_texture_views[i].width = binding->texture->info.width;
    host_texture_views[i].height = binding->texture->info.height;
    host_texture_views[i].format = binding->texture->info.format;
    host_texture_views[i].sampler = binding->sampler;
  }

  bool success = c3dCheckCUDA(
      cudaMemcpy(*texture_views, host_texture_views, render_pass->textureBindCount * sizeof(C3DTextureView), cudaMemcpyHostToDevice),
      "cudaMemcpy failed while uploading texture view metadata");
  free(host_texture_views);
  return success;
}

static size_t c3dGetPrimitiveCount(const C3DDrawInfo* draw_info)
{
  switch (draw_info->topology)
  {
    case C3D_TOPOLOGY_LINE:
      return draw_info->count / 2;
    case C3D_TOPOLOGY_QUAD:
      return draw_info->count / 2;
    case C3D_TOPOLOGY_TRIANGLE:
      return draw_info->count / 3;
  }

  return 0;
}

static bool c3dBuildTileBins(
    C3DCommandBuffer* command_buffer,
    C3DTextureInfo target_info,
    size_t primitive_count,
    bool triangles,
    const void* primitives,
    size_t* tiles_x_out,
    size_t* tiles_y_out)
{
  size_t tiles_x = (target_info.width + (size_t)C3D_TILE_SIZE - 1) / (size_t)C3D_TILE_SIZE;
  size_t tiles_y = (target_info.height + (size_t)C3D_TILE_SIZE - 1) / (size_t)C3D_TILE_SIZE;
  size_t tile_count = tiles_x * tiles_y;
  *tiles_x_out = tiles_x;
  *tiles_y_out = tiles_y;

  if (!c3dTryResizeDeviceBuffer((void**)&command_buffer->tile_counts_device, &command_buffer->tile_count_cap, tile_count * sizeof(uint32_t), "cudaMalloc failed while allocating tile count buffer"))
  {
    return false;
  }

  if (!c3dTryResizeDeviceBuffer((void**)&command_buffer->tile_offsets_device, &command_buffer->tile_offset_cap, (tile_count + 1) * sizeof(uint32_t), "cudaMalloc failed while allocating tile offset buffer"))
  {
    return false;
  }

  if (!c3dTryResizeHostBuffer(&command_buffer->tile_counts_host, &command_buffer->tile_counts_host_cap, 1, "failed to allocate host tile total"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemset(command_buffer->tile_counts_device, 0, tile_count * sizeof(uint32_t)), "cudaMemset failed while clearing tile counts"))
  {
    return false;
  }

  const int bin_threads = 128;
  const int bin_blocks = (int)((primitive_count + (size_t)bin_threads - 1) / (size_t)bin_threads);
  if (triangles)
  {
    c3dBinTrianglePrimitivesKernel<<<bin_blocks, bin_threads>>>((const C3DTrianglePrimitive*)primitives, primitive_count, tiles_x, command_buffer->tile_counts_device);
  }
  else
  {
    c3dBinLinePrimitivesKernel<<<bin_blocks, bin_threads>>>((const C3DLinePrimitive*)primitives, primitive_count, tiles_x, command_buffer->tile_counts_device);
  }

  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile bin count kernel launch failed"))
  {
    return false;
  }

  const int scan_threads = 256;
  const int scan_blocks = (int)((tile_count + (size_t)scan_threads - 1) / (size_t)scan_threads);
  uint32_t* scan_input = command_buffer->tile_counts_device;
  uint32_t* scan_output = command_buffer->tile_offsets_device;
  for (size_t stride = 1; stride < tile_count; stride *= 2)
  {
    c3dTileScanStepKernel<<<scan_blocks, scan_threads>>>(scan_input, scan_output, tile_count, stride);
    if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile scan kernel launch failed"))
    {
      return false;
    }

    uint32_t* temp = scan_input;
    scan_input = scan_output;
    scan_output = temp;
  }

  c3dTileExclusiveShiftKernel<<<scan_blocks, scan_threads>>>(scan_input, command_buffer->tile_offsets_device, tile_count);
  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile exclusive shift kernel launch failed"))
  {
    return false;
  }

  c3dFinalizeTileOffsetsKernel<<<1, 1>>>(command_buffer->tile_offsets_device, command_buffer->tile_counts_device, tile_count);
  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile offset finalize kernel launch failed"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemcpy(command_buffer->tile_counts_host, command_buffer->tile_offsets_device + tile_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy failed while reading tile index count"))
  {
    return false;
  }

  uint32_t total_indices = command_buffer->tile_counts_host[0];

  if (!c3dTryResizeDeviceBuffer((void**)&command_buffer->tile_indices_device, &command_buffer->tile_index_cap, (size_t)total_indices * sizeof(uint32_t), "cudaMalloc failed while allocating tile index buffer"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemcpy(command_buffer->tile_counts_device, command_buffer->tile_offsets_device, tile_count * sizeof(uint32_t), cudaMemcpyDeviceToDevice), "cudaMemcpy failed while preparing tile scatter cursors"))
  {
    return false;
  }

  if (triangles)
  {
    c3dScatterTrianglePrimitivesKernel<<<bin_blocks, bin_threads>>>((const C3DTrianglePrimitive*)primitives, primitive_count, tiles_x, command_buffer->tile_counts_device, command_buffer->tile_indices_device);
  }
  else
  {
    c3dScatterLinePrimitivesKernel<<<bin_blocks, bin_threads>>>((const C3DLinePrimitive*)primitives, primitive_count, tiles_x, command_buffer->tile_counts_device, command_buffer->tile_indices_device);
  }

  return c3dCheckCUDA(cudaPeekAtLastError(), "tile scatter kernel launch failed");
}

static bool c3dExecuteDraw(C3DCommandBuffer* command_buffer, C3DTexture* target, const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, const C3DDrawInfo* draw_info, uint64_t* depth_buffer, uint32_t primitive_order_base, uint32_t* primitive_order_count)
{
  if (!c3dValidateDrawRanges(draw_info))
  {
    return false;
  }

  C3DTextureInfo target_info = target->info;
  uint8_t* pixels = (uint8_t*)target->data;
  const uint8_t* index_data = draw_info->indexBuffer->device_data;
  const C3DVertex* vertex_data = (const C3DVertex*)draw_info->vertexBuffer->device_data;
  size_t first_index = draw_info->indexOffset / c3dGetIndexStride(draw_info->indexBuffer->info.indexSize);
  size_t primitive_count = c3dGetPrimitiveCount(draw_info);
  *primitive_order_count = (uint32_t)primitive_count;

  dim3 raster_threads(C3D_TILE_SIZE, C3D_TILE_SIZE);
  size_t tiles_x = 0;
  size_t tiles_y = 0;

  if (draw_info->topology == C3D_TOPOLOGY_LINE)
  {
    if ((draw_info->count % 2) != 0)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "line draws require an even index count");
      return false;
    }

    if (primitive_count == 0)
    {
      return true;
    }

    if (!c3dTryResizeDeviceBuffer(&command_buffer->line_primitives, &command_buffer->line_primitive_cap, primitive_count * sizeof(C3DLinePrimitive), "cudaMalloc failed while allocating line primitive buffer"))
    {
      return false;
    }
    C3DLinePrimitive* primitives = (C3DLinePrimitive*)command_buffer->line_primitives;

    const int setup_threads = 128;
    const int setup_blocks = (int)((primitive_count + (size_t)setup_threads - 1) / (size_t)setup_threads);
    c3dSetupLinePrimitivesKernel<<<setup_blocks, setup_threads>>>(
        primitives,
        target_info,
        index_data,
        draw_info->indexBuffer->info.indexSize,
        first_index,
        draw_info->indexBase,
        vertex_data,
        draw_info->vertexOffset,
        primitive_count,
        primitive_order_base);
    bool success = c3dCheckCUDA(cudaPeekAtLastError(), "line setup kernel launch failed");
    if (success)
    {
      success = c3dBuildTileBins(command_buffer, target_info, primitive_count, false, primitives, &tiles_x, &tiles_y);
    }
    if (success)
    {
      dim3 raster_blocks((unsigned int)tiles_x, (unsigned int)tiles_y);
      c3dRasterizeLinesKernel<<<raster_blocks, raster_threads>>>(
          target_info,
          pixels,
          depth_buffer,
          render_pass->targetBlend,
          texture_views,
          render_pass->textureBindCount,
          primitives,
          tiles_x,
          command_buffer->tile_offsets_device,
          command_buffer->tile_indices_device);
      success = c3dCheckCUDA(cudaPeekAtLastError(), "line raster kernel launch failed");
    }
    return success;
  }

  if (draw_info->topology == C3D_TOPOLOGY_QUAD && (draw_info->count % 4) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "quad draws require an index count divisible by four");
    return false;
  }

  if (draw_info->topology == C3D_TOPOLOGY_TRIANGLE && (draw_info->count % 3) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "triangle draws require an index count divisible by three");
    return false;
  }

  if (primitive_count == 0)
  {
    return true;
  }

  if (!c3dTryResizeDeviceBuffer(&command_buffer->triangle_primitives, &command_buffer->triangle_primitive_cap, primitive_count * sizeof(C3DTrianglePrimitive), "cudaMalloc failed while allocating triangle primitive buffer"))
  {
    return false;
  }
  C3DTrianglePrimitive* primitives = (C3DTrianglePrimitive*)command_buffer->triangle_primitives;

  const int setup_threads = 128;
  const int setup_blocks = (int)((primitive_count + (size_t)setup_threads - 1) / (size_t)setup_threads);
  c3dSetupTrianglePrimitivesKernel<<<setup_blocks, setup_threads>>>(
      primitives,
      target_info,
      draw_info->topology,
      index_data,
      draw_info->indexBuffer->info.indexSize,
      first_index,
      draw_info->indexBase,
      vertex_data,
      draw_info->vertexOffset,
      primitive_count,
      primitive_order_base);
  bool success = c3dCheckCUDA(cudaPeekAtLastError(), "triangle setup kernel launch failed");
  if (success)
  {
    success = c3dBuildTileBins(command_buffer, target_info, primitive_count, true, primitives, &tiles_x, &tiles_y);
  }
  if (success)
  {
    dim3 raster_blocks((unsigned int)tiles_x, (unsigned int)tiles_y);
    c3dRasterizeTrianglesKernel<<<raster_blocks, raster_threads>>>(
        target_info,
        pixels,
        depth_buffer,
        render_pass->targetBlend,
        texture_views,
        render_pass->textureBindCount,
        primitives,
        tiles_x,
        command_buffer->tile_offsets_device,
        command_buffer->tile_indices_device);
    success = c3dCheckCUDA(cudaPeekAtLastError(), "triangle raster kernel launch failed");
  }
  return success;
}

static bool c3dTryGrowDrawList(C3DCommandBuffer* command_buffer, size_t min_cap)
{
  if (command_buffer->draw_cap >= min_cap)
  {
    return true;
  }

  size_t new_cap = command_buffer->draw_cap == 0 ? 4 : command_buffer->draw_cap * 2;
  while (new_cap < min_cap)
  {
    if (new_cap > (SIZE_MAX / 2))
    {
      new_cap = min_cap;
      break;
    }

    new_cap *= 2;
  }

  if (new_cap > (SIZE_MAX / sizeof(C3DRecordedDraw)))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw list size overflows size_t");
    return false;
  }

  C3DRecordedDraw* draws = (C3DRecordedDraw*)realloc(command_buffer->draws, new_cap * sizeof(C3DRecordedDraw));
  if (!draws)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to grow command buffer draw list");
    return false;
  }

  command_buffer->draws = draws;
  command_buffer->draw_cap = new_cap;
  return true;
}

static void c3dResetRenderPassInfo(C3DRenderPassInfo* render_pass)
{
  free(render_pass->textureBindings);
  memset(render_pass, 0, sizeof(*render_pass));
}

static void c3dResetCommandBuffer(C3DCommandBuffer* command_buffer)
{
  command_buffer->in_render_pass = false;
  command_buffer->has_render_pass = false;
  command_buffer->draw_count = 0;
  c3dResetRenderPassInfo(&command_buffer->render_pass);
}

static void c3dReleaseCommandBufferScratch(C3DCommandBuffer* command_buffer)
{
  cudaFree(command_buffer->depth_buffer);
  cudaFree(command_buffer->line_primitives);
  cudaFree(command_buffer->triangle_primitives);
  cudaFree(command_buffer->texture_views);
  cudaFree(command_buffer->tile_counts_device);
  cudaFree(command_buffer->tile_offsets_device);
  cudaFree(command_buffer->tile_indices_device);
  free(command_buffer->tile_counts_host);
  free(command_buffer->tile_offsets_host);
  command_buffer->depth_buffer = nullptr;
  command_buffer->depth_cap = 0;
  command_buffer->line_primitives = nullptr;
  command_buffer->line_primitive_cap = 0;
  command_buffer->triangle_primitives = nullptr;
  command_buffer->triangle_primitive_cap = 0;
  command_buffer->texture_views = nullptr;
  command_buffer->texture_view_cap = 0;
  command_buffer->tile_counts_device = nullptr;
  command_buffer->tile_offsets_device = nullptr;
  command_buffer->tile_indices_device = nullptr;
  command_buffer->tile_counts_host = nullptr;
  command_buffer->tile_offsets_host = nullptr;
  command_buffer->tile_count_cap = 0;
  command_buffer->tile_offset_cap = 0;
  command_buffer->tile_index_cap = 0;
  command_buffer->tile_counts_host_cap = 0;
  command_buffer->tile_offsets_host_cap = 0;
}

static bool c3dIsValidSampler(C3DSampler sampler)
{
  return sampler >= C3D_SAMPLER_POINT_CLAMP && sampler <= C3D_SAMPLER_LINEAR_WRAP;
}

static bool c3dIsValidTopology(C3DTopology topology)
{
  return topology >= C3D_TOPOLOGY_LINE && topology <= C3D_TOPOLOGY_TRIANGLE;
}

static bool c3dIsValidBlendMode(C3DBlendMode blend_mode)
{
  return blend_mode >= C3D_BLEND_MODE_NONE && blend_mode <= C3D_BLEND_MODE_ADDITIVE;
}

static bool c3dValidateRenderPass(const C3DRenderPassInfo* render_pass)
{
  if (!render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass info must be non-null");
    return false;
  }

  if (!render_pass->target)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass target must be non-null");
    return false;
  }

  if (!c3dIsValidBlendMode(render_pass->targetBlend))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass blend mode is invalid");
    return false;
  }

  if (render_pass->textureBindCount != 0 && !render_pass->textureBindings)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture bindings must be non-null when textureBindCount is non-zero");
    return false;
  }

  for (size_t i = 0; i < render_pass->textureBindCount; ++i)
  {
    const C3DTextureBinding* binding = &render_pass->textureBindings[i];
    if (!binding->texture)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding texture must be non-null");
      return false;
    }

    if (!c3dIsValidSampler(binding->sampler))
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding sampler is invalid");
      return false;
    }
  }

  return true;
}

static bool c3dCopyRenderPassInfo(C3DRenderPassInfo* destination, const C3DRenderPassInfo* source)
{
  *destination = *source;
  destination->textureBindings = nullptr;

  if (source->textureBindCount == 0)
  {
    return true;
  }

  if (source->textureBindCount > (SIZE_MAX / sizeof(C3DTextureBinding)))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding list size overflows size_t");
    return false;
  }

  destination->textureBindings = (C3DTextureBinding*)malloc(source->textureBindCount * sizeof(C3DTextureBinding));
  if (!destination->textureBindings)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture binding list");
    return false;
  }

  memcpy(destination->textureBindings, source->textureBindings, source->textureBindCount * sizeof(C3DTextureBinding));
  return true;
}

static bool c3dValidateDrawInfo(const C3DDrawInfo* draw_info)
{
  if (!draw_info)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw info must be non-null");
    return false;
  }

  if (!c3dIsValidTopology(draw_info->topology))
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw topology is invalid");
    return false;
  }

  if (!draw_info->indexBuffer || !draw_info->vertexBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw buffers must be non-null");
    return false;
  }

  return true;
}

C3D_API C3DCommandBuffer* c3dCreateCommandBuffer(void)
{
  C3DCommandBuffer* command_buffer = (C3DCommandBuffer*)malloc(sizeof(C3DCommandBuffer));
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate command buffer object");
    return nullptr;
  }

  memset(command_buffer, 0, sizeof(C3DCommandBuffer));
  return command_buffer;
}

C3D_API bool c3dDeleteCommandBuffer(C3DCommandBuffer* command_buffer)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  c3dResetRenderPassInfo(&command_buffer->render_pass);
  c3dReleaseCommandBufferScratch(command_buffer);
  free(command_buffer->draws);
  free(command_buffer);
  return true;
}

C3D_API bool c3dSubmitCommandBuffer(C3DCommandBuffer* command_buffer)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (command_buffer->in_render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "cannot submit a command buffer while a render pass is active");
    return false;
  }

  if (!command_buffer->has_render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no recorded render pass to submit");
    return false;
  }

  C3DTexture* target = command_buffer->render_pass.target;
  size_t pixel_count = target->info.width * target->info.height;
  uint64_t* depth_buffer = nullptr;
  C3DTextureView* texture_views = nullptr;
  bool success = c3dTryResizeDeviceBuffer((void**)&command_buffer->depth_buffer, &command_buffer->depth_cap, pixel_count * sizeof(uint64_t), "cudaMalloc failed while allocating depth buffer");
  if (success)
  {
    depth_buffer = command_buffer->depth_buffer;
    success = c3dBuildTextureViews(command_buffer, &command_buffer->render_pass, &texture_views);
  }

  if (success)
  {
    const int clear_threads = 256;
    const int clear_blocks = (int)((pixel_count + (size_t)clear_threads - 1) / (size_t)clear_threads);
    c3dClearDepthBufferKernel<<<clear_blocks, clear_threads>>>(depth_buffer, pixel_count);
    success = c3dCheckCUDA(cudaPeekAtLastError(), "depth clear kernel launch failed");
  }

  uint32_t primitive_order_base = 0;
  if (success)
  {
    for (size_t i = 0; i < command_buffer->draw_count; ++i)
    {
      uint32_t primitive_order_count = 0;
      if (!c3dExecuteDraw(command_buffer, target, &command_buffer->render_pass, texture_views, &command_buffer->draws[i].info, depth_buffer, primitive_order_base, &primitive_order_count))
      {
        success = false;
        break;
      }

      primitive_order_base += primitive_order_count;
    }
  }

  if (success)
  {
    success = c3dCheckCUDA(cudaDeviceSynchronize(), "render kernel execution failed");
  }

  if (!success)
  {
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  return true;
}

C3D_API bool c3dCancelCommandBuffer(C3DCommandBuffer* command_buffer)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  return true;
}

C3D_API bool c3dBeginRenderPass(C3DCommandBuffer* command_buffer, C3DRenderPassInfo* render_pass)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (command_buffer->in_render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer already has an active render pass");
    return false;
  }

  if (!c3dValidateRenderPass(render_pass))
  {
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  if (!c3dCopyRenderPassInfo(&command_buffer->render_pass, render_pass))
  {
    return false;
  }

  command_buffer->in_render_pass = true;
  command_buffer->has_render_pass = true;
  return true;
}

C3D_API bool c3dEndRenderPass(C3DCommandBuffer* command_buffer)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (!command_buffer->in_render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no active render pass");
    return false;
  }

  command_buffer->in_render_pass = false;
  return true;
}

C3D_API bool c3dDraw(C3DCommandBuffer* command_buffer, const C3DDrawInfo* draw_info)
{
  if (!command_buffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (!command_buffer->in_render_pass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw commands require an active render pass");
    return false;
  }

  if (!c3dValidateDrawInfo(draw_info))
  {
    return false;
  }

  if (!c3dTryGrowDrawList(command_buffer, command_buffer->draw_count + 1))
  {
    return false;
  }

  command_buffer->draws[command_buffer->draw_count].info = *draw_info;
  command_buffer->draw_count += 1;
  return true;
}
