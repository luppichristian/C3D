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
  int minX;
  int minY;
  int maxX;
  int maxY;
  uint32_t order;
  int valid;
};

struct C3DTrianglePrimitive
{
  C3DVertexSample a;
  C3DVertexSample b;
  C3DVertexSample c;
  float area;
  int minX;
  int minY;
  int maxX;
  int maxY;
  uint32_t order;
  int valid;
};

static const int C3D_TILE_SIZE = 16;

static bool c3dTryResizeDeviceBuffer(void** buffer, size_t* cap, size_t requiredBytes, const char* desc)
{
  if (*cap >= requiredBytes)
  {
    return true;
  }

  if (*buffer && !c3dCheckCUDA(cudaFree(*buffer), "cudaFree failed while replacing command buffer scratch storage"))
  {
    return false;
  }

  *buffer = nullptr;
  *cap = 0;
  if (requiredBytes == 0)
  {
    return true;
  }

  if (!c3dCheckCUDA(cudaMalloc(buffer, requiredBytes), desc))
  {
    return false;
  }

  *cap = requiredBytes;
  return true;
}

static bool c3dTryResizeHostBuffer(uint32_t** buffer, size_t* cap, size_t count, const char* desc)
{
  if (*cap >= count)
  {
    return true;
  }

  uint32_t* newBuffer = (uint32_t*)realloc(*buffer, count * sizeof(uint32_t));
  if (!newBuffer)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, desc);
    return false;
  }

  *buffer = newBuffer;
  *cap = count;
  return true;
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

static __host__ __device__ float c3dClampf(float value, float minValue, float maxValue)
{
  if (value < minValue)
  {
    return minValue;
  }

  if (value > maxValue)
  {
    return maxValue;
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
  uint32_t depthBits = (uint32_t)(c3dClamp01(depth) * 4294967295.0f + 0.5f);
  return ((uint64_t)depthBits << 32) | (uint64_t)order;
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
    default:
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
    default:
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

static __host__ __device__ C3DPixel c3dBlendPixel(C3DPixel source, C3DPixel destination, C3DBlendMode blendMode)
{
  switch (blendMode)
  {
    case C3D_BLEND_MODE_NONE:
      return source;
    case C3D_BLEND_MODE_NORMAL:
    {
      C3DPixel pixel = {0};
      float invAlpha = 1.0f - source.a;
      pixel.r = source.r + (destination.r * invAlpha);
      pixel.g = source.g + (destination.g * invAlpha);
      pixel.b = source.b + (destination.b * invAlpha);
      pixel.a = source.a + (destination.a * invAlpha);
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

static __device__ uint8_t* c3dGetTargetTexel(C3DTextureInfo targetInfo, uint8_t* pixels, size_t x, size_t y)
{
  size_t texelSize = c3dGetTextureFormatSize(targetInfo.format);
  return pixels + (((y * targetInfo.width) + x) * texelSize);
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
  size_t maxIndex = limit - 1;
  return index > maxIndex ? maxIndex : index;
}

static __device__ C3DPixel c3dSampleTextureNearest(const C3DTextureView* texture, float u, float v)
{
  float sampleU = c3dApplySamplerAddress(u, texture->sampler);
  float sampleV = c3dApplySamplerAddress(v, texture->sampler);
  float x = sampleU * (float)(texture->width - 1);
  float y = sampleV * (float)(texture->height - 1);
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
  float sampleU = c3dApplySamplerAddress(u, texture->sampler);
  float sampleV = c3dApplySamplerAddress(v, texture->sampler);
  float x = sampleU * (float)(texture->width - 1);
  float y = sampleV * (float)(texture->height - 1);

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

static __device__ C3DPixel c3dSampleBoundTexture(const C3DTextureView* textureViews, size_t textureCount, int texid, float u, float v)
{
  if (texid < 0 || (size_t)texid >= textureCount)
  {
    C3DPixel white = {1.0f, 1.0f, 1.0f, 1.0f};
    return white;
  }

  const C3DTextureView* texture = &textureViews[texid];
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

static __host__ __device__ size_t c3dReadIndexValueRaw(const uint8_t* indexData, C3DIndexSize indexSize, size_t index)
{
  const uint8_t* ptr = indexData + (index * c3dGetIndexStride(indexSize));
  switch (indexSize)
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

static bool c3dValidateDrawRanges(const C3DDrawInfo* drawInfo)
{
  size_t indexStride = c3dGetIndexStride(drawInfo->indexSize);
  if (indexStride == 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index size is invalid");
    return false;
  }

  size_t indexStart = drawInfo->indexOffset;
  size_t indexBytes = drawInfo->count * indexStride;
  if (indexStart > drawInfo->indexBuffer->info.size || indexBytes > drawInfo->indexBuffer->info.size - indexStart)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index range is out of bounds");
    return false;
  }

  size_t firstIndex = drawInfo->indexOffset / indexStride;
  size_t vertexLimit = drawInfo->vertexBuffer->info.size / sizeof(C3DVertex);
  for (size_t i = 0; i < drawInfo->count; ++i)
  {
    size_t index = c3dReadIndexValueRaw(drawInfo->indexBuffer->hostData, drawInfo->indexSize, firstIndex + i);
    size_t vertexIndex = drawInfo->vertexOffset + drawInfo->indexBase + index;
    if (vertexIndex >= vertexLimit)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw vertex range is out of bounds");
      return false;
    }
  }

  return true;
}

static __device__ C3DVertexSample c3dLoadVertexRaw(const uint8_t* indexData, C3DIndexSize indexSize, size_t firstIndex, size_t drawIndex, size_t indexBase, const C3DVertex* vertexData, size_t vertexOffset)
{
  size_t index = c3dReadIndexValueRaw(indexData, indexSize, firstIndex + drawIndex);
  const C3DVertex* vertex = vertexData + vertexOffset + indexBase + index;

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
  float invW = vertex->w != 0.0f ? (1.0f / vertex->w) : 1.0f;
  return c3dClamp01((vertex->z * invW * 0.5f) + 0.5f);
}

static __host__ __device__ void c3dNdcToScreen(C3DViewport viewport, C3DVertexSample* vertex)
{
  float invW = vertex->w != 0.0f ? (1.0f / vertex->w) : 1.0f;
  float x = vertex->x * invW;
  float y = vertex->y * invW;
  vertex->x = (float)viewport.x + (((x * 0.5f) + 0.5f) * (float)(viewport.width - 1));
  vertex->y = (float)viewport.y + ((1.0f - ((y * 0.5f) + 0.5f)) * (float)(viewport.height - 1));
}

static __device__ void c3dWritePixel(C3DTextureInfo targetInfo, uint8_t* pixels, size_t x, size_t y, C3DPixel color, C3DBlendMode blendMode)
{
  uint8_t* texel = c3dGetTargetTexel(targetInfo, pixels, x, y);
  C3DPixel destination = c3dUnpackPixel(texel, targetInfo.format);
  C3DPixel blended = c3dBlendPixel(color, destination, blendMode);
  c3dPackPixel(texel, targetInfo.format, blended);
}

static __device__ void c3dShadePixel(C3DTextureInfo targetInfo, uint8_t* pixels, C3DBlendMode blendMode, const C3DTextureView* textureViews, size_t textureCount, size_t x, size_t y, const C3DVertexSample* sample)
{
  C3DPixel vertexColor = {sample->r, sample->g, sample->b, sample->a};
  C3DPixel textureColor = c3dSampleBoundTexture(textureViews, textureCount, sample->texid, sample->u, sample->v);
  c3dWritePixel(targetInfo, pixels, x, y, c3dMulPixel(vertexColor, textureColor), blendMode);
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

static __device__ void c3dGetTriangleDrawIndices(C3DTopology topology, size_t primitiveIndex, size_t* a, size_t* b, size_t* c)
{
  if (topology == C3D_TOPOLOGY_QUAD)
  {
    size_t quad = primitiveIndex / 2;
    size_t base = quad * 4;
    if ((primitiveIndex & 1) == 0)
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

  size_t base = primitiveIndex * 3;
  *a = base + 0;
  *b = base + 1;
  *c = base + 2;
}

static __global__ void c3dClearDepthBufferKernel(uint64_t* depthBuffer, size_t count)
{
  size_t index = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (index < count)
  {
    depthBuffer[index] = ~0ull;
  }
}

static __global__ void c3dFinalizeTileOffsetsKernel(uint32_t* tileOffsets, const uint32_t* tileCounts, size_t tileCount)
{
  if (threadIdx.x == 0 && blockIdx.x == 0)
  {
    if (tileCount == 0)
    {
      tileOffsets[0] = 0;
      return;
    }

    tileOffsets[tileCount] = tileOffsets[tileCount - 1] + tileCounts[tileCount - 1];
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

static __global__ void c3dSetupLinePrimitivesKernel(C3DLinePrimitive* primitives, C3DTextureInfo targetInfo, C3DViewport viewport, const uint8_t* indexData, C3DIndexSize indexSize, size_t firstIndex, size_t indexBase, const C3DVertex* vertexData, size_t vertexOffset, size_t primitiveCount, uint32_t orderBase)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  C3DLinePrimitive primitive = {0};
  primitive.a = c3dLoadVertexRaw(indexData, indexSize, firstIndex, primitiveIndex * 2 + 0, indexBase, vertexData, vertexOffset);
  primitive.b = c3dLoadVertexRaw(indexData, indexSize, firstIndex, primitiveIndex * 2 + 1, indexBase, vertexData, vertexOffset);
  c3dNdcToScreen(viewport, &primitive.a);
  c3dNdcToScreen(viewport, &primitive.b);
  primitive.minX = (int)floorf(fminf(primitive.a.x, primitive.b.x)) - 1;
  primitive.minY = (int)floorf(fminf(primitive.a.y, primitive.b.y)) - 1;
  primitive.maxX = (int)ceilf(fmaxf(primitive.a.x, primitive.b.x)) + 1;
  primitive.maxY = (int)ceilf(fmaxf(primitive.a.y, primitive.b.y)) + 1;
  if (primitive.minX < (int)viewport.x) primitive.minX = (int)viewport.x;
  if (primitive.minY < (int)viewport.y) primitive.minY = (int)viewport.y;
  if (primitive.maxX >= (int)(viewport.x + viewport.width)) primitive.maxX = (int)(viewport.x + viewport.width) - 1;
  if (primitive.maxY >= (int)(viewport.y + viewport.height)) primitive.maxY = (int)(viewport.y + viewport.height) - 1;
  primitive.order = orderBase + (uint32_t)primitiveIndex;
  primitive.valid = 1;
  primitives[primitiveIndex] = primitive;
}

static __global__ void c3dSetupTrianglePrimitivesKernel(C3DTrianglePrimitive* primitives, C3DTextureInfo targetInfo, C3DViewport viewport, C3DTopology topology, const uint8_t* indexData, C3DIndexSize indexSize, size_t firstIndex, size_t indexBase, const C3DVertex* vertexData, size_t vertexOffset, size_t primitiveCount, uint32_t orderBase)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  size_t ai = 0;
  size_t bi = 0;
  size_t ci = 0;
  c3dGetTriangleDrawIndices(topology, primitiveIndex, &ai, &bi, &ci);

  C3DTrianglePrimitive primitive = {0};
  primitive.a = c3dLoadVertexRaw(indexData, indexSize, firstIndex, ai, indexBase, vertexData, vertexOffset);
  primitive.b = c3dLoadVertexRaw(indexData, indexSize, firstIndex, bi, indexBase, vertexData, vertexOffset);
  primitive.c = c3dLoadVertexRaw(indexData, indexSize, firstIndex, ci, indexBase, vertexData, vertexOffset);
  c3dNdcToScreen(viewport, &primitive.a);
  c3dNdcToScreen(viewport, &primitive.b);
  c3dNdcToScreen(viewport, &primitive.c);
  primitive.area = c3dEdge(primitive.a.x, primitive.a.y, primitive.b.x, primitive.b.y, primitive.c.x, primitive.c.y);
  primitive.valid = primitive.area != 0.0f;
  primitive.minX = (int)floorf(fminf(primitive.a.x, fminf(primitive.b.x, primitive.c.x)));
  primitive.minY = (int)floorf(fminf(primitive.a.y, fminf(primitive.b.y, primitive.c.y)));
  primitive.maxX = (int)ceilf(fmaxf(primitive.a.x, fmaxf(primitive.b.x, primitive.c.x)));
  primitive.maxY = (int)ceilf(fmaxf(primitive.a.y, fmaxf(primitive.b.y, primitive.c.y)));
  if (primitive.minX < (int)viewport.x) primitive.minX = (int)viewport.x;
  if (primitive.minY < (int)viewport.y) primitive.minY = (int)viewport.y;
  if (primitive.maxX >= (int)(viewport.x + viewport.width)) primitive.maxX = (int)(viewport.x + viewport.width) - 1;
  if (primitive.maxY >= (int)(viewport.y + viewport.height)) primitive.maxY = (int)(viewport.y + viewport.height) - 1;
  primitive.order = orderBase + (uint32_t)primitiveIndex;
  primitives[primitiveIndex] = primitive;
}

static __global__ void c3dBinLinePrimitivesKernel(const C3DLinePrimitive* primitives, size_t primitiveCount, size_t tilesX, uint32_t* tileCounts)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  const C3DLinePrimitive* primitive = &primitives[primitiveIndex];
  if (!primitive->valid)
  {
    return;
  }

  int minTileX = primitive->minX / C3D_TILE_SIZE;
  int minTileY = primitive->minY / C3D_TILE_SIZE;
  int maxTileX = primitive->maxX / C3D_TILE_SIZE;
  int maxTileY = primitive->maxY / C3D_TILE_SIZE;
  for (int tileY = minTileY; tileY <= maxTileY; ++tileY)
  {
    for (int tileX = minTileX; tileX <= maxTileX; ++tileX)
    {
      atomicAdd(&tileCounts[(tileY * (int)tilesX) + tileX], 1u);
    }
  }
}

static __global__ void c3dBinTrianglePrimitivesKernel(const C3DTrianglePrimitive* primitives, size_t primitiveCount, size_t tilesX, uint32_t* tileCounts)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  const C3DTrianglePrimitive* primitive = &primitives[primitiveIndex];
  if (!primitive->valid)
  {
    return;
  }

  int minTileX = primitive->minX / C3D_TILE_SIZE;
  int minTileY = primitive->minY / C3D_TILE_SIZE;
  int maxTileX = primitive->maxX / C3D_TILE_SIZE;
  int maxTileY = primitive->maxY / C3D_TILE_SIZE;
  for (int tileY = minTileY; tileY <= maxTileY; ++tileY)
  {
    for (int tileX = minTileX; tileX <= maxTileX; ++tileX)
    {
      atomicAdd(&tileCounts[(tileY * (int)tilesX) + tileX], 1u);
    }
  }
}

static __global__ void c3dScatterLinePrimitivesKernel(const C3DLinePrimitive* primitives, size_t primitiveCount, size_t tilesX, uint32_t* tileOffsets, uint32_t* tileIndices)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  const C3DLinePrimitive* primitive = &primitives[primitiveIndex];
  if (!primitive->valid)
  {
    return;
  }

  int minTileX = primitive->minX / C3D_TILE_SIZE;
  int minTileY = primitive->minY / C3D_TILE_SIZE;
  int maxTileX = primitive->maxX / C3D_TILE_SIZE;
  int maxTileY = primitive->maxY / C3D_TILE_SIZE;
  for (int tileY = minTileY; tileY <= maxTileY; ++tileY)
  {
    for (int tileX = minTileX; tileX <= maxTileX; ++tileX)
    {
      uint32_t tileIndex = (uint32_t)((tileY * (int)tilesX) + tileX);
      uint32_t offset = atomicAdd(&tileOffsets[tileIndex], 1u);
      tileIndices[offset] = (uint32_t)primitiveIndex;
    }
  }
}

static __global__ void c3dScatterTrianglePrimitivesKernel(const C3DTrianglePrimitive* primitives, size_t primitiveCount, size_t tilesX, uint32_t* tileOffsets, uint32_t* tileIndices)
{
  size_t primitiveIndex = ((size_t)blockIdx.x * (size_t)blockDim.x) + (size_t)threadIdx.x;
  if (primitiveIndex >= primitiveCount)
  {
    return;
  }

  const C3DTrianglePrimitive* primitive = &primitives[primitiveIndex];
  if (!primitive->valid)
  {
    return;
  }

  int minTileX = primitive->minX / C3D_TILE_SIZE;
  int minTileY = primitive->minY / C3D_TILE_SIZE;
  int maxTileX = primitive->maxX / C3D_TILE_SIZE;
  int maxTileY = primitive->maxY / C3D_TILE_SIZE;
  for (int tileY = minTileY; tileY <= maxTileY; ++tileY)
  {
    for (int tileX = minTileX; tileX <= maxTileX; ++tileX)
    {
      uint32_t tileIndex = (uint32_t)((tileY * (int)tilesX) + tileX);
      uint32_t offset = atomicAdd(&tileOffsets[tileIndex], 1u);
      tileIndices[offset] = (uint32_t)primitiveIndex;
    }
  }
}

static __global__ void c3dRasterizeLinesKernel(C3DTextureInfo targetInfo, uint8_t* pixels, uint64_t* depthBuffer, C3DBlendMode blendMode, const C3DTextureView* textureViews, size_t textureCount, const C3DLinePrimitive* primitives, size_t tilesX, const uint32_t* tileOffsets, const uint32_t* tileIndices)
{
  size_t tileX = (size_t)blockIdx.x;
  size_t tileY = (size_t)blockIdx.y;
  size_t x = (tileX * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.x;
  size_t y = (tileY * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.y;
  if (x >= targetInfo.width || y >= targetInfo.height)
  {
    return;
  }

  size_t tileIndex = (tileY * tilesX) + tileX;
  uint32_t start = tileOffsets[tileIndex];
  uint32_t end = tileOffsets[tileIndex + 1];
  bool depthTestEnabled = depthBuffer != nullptr;

  size_t pixelIndex = (y * targetInfo.width) + x;
  uint64_t bestKey = depthTestEnabled ? depthBuffer[pixelIndex] : 0;
  uint32_t bestOrder = 0;
  C3DVertexSample bestSample = {0};
  bool hasCandidate = false;
  float px = (float)x + 0.5f;
  float py = (float)y + 0.5f;
  const float maxDistanceSq = 0.25f;

  for (uint32_t i = start; i < end; ++i)
  {
    const C3DLinePrimitive* primitive = &primitives[tileIndices[i]];
    if (!primitive->valid)
    {
      continue;
    }

    if ((int)x < primitive->minX || (int)x > primitive->maxX || (int)y < primitive->minY || (int)y > primitive->maxY)
    {
      continue;
    }

    float dx = primitive->b.x - primitive->a.x;
    float dy = primitive->b.y - primitive->a.y;
    float lengthSq = (dx * dx) + (dy * dy);
    float t = 0.0f;
    if (lengthSq > 0.0f)
    {
      t = (((px - primitive->a.x) * dx) + ((py - primitive->a.y) * dy)) / lengthSq;
      t = c3dClampf(t, 0.0f, 1.0f);
    }

    float sx = primitive->a.x + (dx * t);
    float sy = primitive->a.y + (dy * t);
    float ddx = px - sx;
    float ddy = py - sy;
    float distanceSq = (ddx * ddx) + (ddy * ddy);
    if (distanceSq > maxDistanceSq)
    {
      continue;
    }

    C3DVertexSample sample = c3dLerpVertex(&primitive->a, &primitive->b, t);
    if (depthTestEnabled)
    {
      uint64_t key = c3dEncodeDepthOrder(c3dSampleDepth(&sample), primitive->order);
      if (key < bestKey)
      {
        bestKey = key;
        bestSample = sample;
        hasCandidate = true;
      }
    }
    else if (!hasCandidate || primitive->order > bestOrder)
    {
      bestOrder = primitive->order;
      bestSample = sample;
      hasCandidate = true;
    }
  }

  if (hasCandidate)
  {
    if (depthTestEnabled)
    {
      depthBuffer[pixelIndex] = bestKey;
    }

    c3dShadePixel(targetInfo, pixels, blendMode, textureViews, textureCount, x, y, &bestSample);
  }
}

static __global__ void c3dRasterizeTrianglesKernel(C3DTextureInfo targetInfo, uint8_t* pixels, uint64_t* depthBuffer, C3DBlendMode blendMode, const C3DTextureView* textureViews, size_t textureCount, const C3DTrianglePrimitive* primitives, size_t tilesX, const uint32_t* tileOffsets, const uint32_t* tileIndices)
{
  size_t tileX = (size_t)blockIdx.x;
  size_t tileY = (size_t)blockIdx.y;
  size_t x = (tileX * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.x;
  size_t y = (tileY * (size_t)C3D_TILE_SIZE) + (size_t)threadIdx.y;
  if (x >= targetInfo.width || y >= targetInfo.height)
  {
    return;
  }

  size_t tileIndex = (tileY * tilesX) + tileX;
  uint32_t start = tileOffsets[tileIndex];
  uint32_t end = tileOffsets[tileIndex + 1];
  bool depthTestEnabled = depthBuffer != nullptr;

  size_t pixelIndex = (y * targetInfo.width) + x;
  uint64_t bestKey = depthTestEnabled ? depthBuffer[pixelIndex] : 0;
  uint32_t bestOrder = 0;
  C3DVertexSample bestSample = {0};
  bool hasCandidate = false;
  float px = (float)x + 0.5f;
  float py = (float)y + 0.5f;

  for (uint32_t i = start; i < end; ++i)
  {
    const C3DTrianglePrimitive* primitive = &primitives[tileIndices[i]];
    if (!primitive->valid)
    {
      continue;
    }

    if ((int)x < primitive->minX || (int)x > primitive->maxX || (int)y < primitive->minY || (int)y > primitive->maxY)
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

    float invArea = 1.0f / primitive->area;
    float wa = w0 * invArea;
    float wb = w1 * invArea;
    float wc = w2 * invArea;
    float invWa = primitive->a.w != 0.0f ? (1.0f / primitive->a.w) : 1.0f;
    float invWb = primitive->b.w != 0.0f ? (1.0f / primitive->b.w) : 1.0f;
    float invWc = primitive->c.w != 0.0f ? (1.0f / primitive->c.w) : 1.0f;
    float pwa = wa * invWa;
    float pwb = wb * invWb;
    float pwc = wc * invWc;
    float denom = pwa + pwb + pwc;
    if (denom == 0.0f)
    {
      continue;
    }

    float invDenom = 1.0f / denom;

    C3DVertexSample sample = {0};
    sample.z = ((primitive->a.z * pwa) + (primitive->b.z * pwb) + (primitive->c.z * pwc)) * invDenom;
    sample.w = 1.0f / denom;
    sample.r = ((primitive->a.r * pwa) + (primitive->b.r * pwb) + (primitive->c.r * pwc)) * invDenom;
    sample.g = ((primitive->a.g * pwa) + (primitive->b.g * pwb) + (primitive->c.g * pwc)) * invDenom;
    sample.b = ((primitive->a.b * pwa) + (primitive->b.b * pwb) + (primitive->c.b * pwc)) * invDenom;
    sample.a = ((primitive->a.a * pwa) + (primitive->b.a * pwb) + (primitive->c.a * pwc)) * invDenom;
    sample.u = ((primitive->a.u * pwa) + (primitive->b.u * pwb) + (primitive->c.u * pwc)) * invDenom;
    sample.v = ((primitive->a.v * pwa) + (primitive->b.v * pwb) + (primitive->c.v * pwc)) * invDenom;
    sample.texid = primitive->a.texid >= 0 ? primitive->a.texid : (primitive->b.texid >= 0 ? primitive->b.texid : primitive->c.texid);

    if (depthTestEnabled)
    {
      uint64_t key = c3dEncodeDepthOrder(c3dSampleDepth(&sample), primitive->order);
      if (key < bestKey)
      {
        bestKey = key;
        bestSample = sample;
        hasCandidate = true;
      }
    }
    else if (!hasCandidate || primitive->order > bestOrder)
    {
      bestOrder = primitive->order;
      bestSample = sample;
      hasCandidate = true;
    }
  }

  if (hasCandidate)
  {
    if (depthTestEnabled)
    {
      depthBuffer[pixelIndex] = bestKey;
    }

    c3dShadePixel(targetInfo, pixels, blendMode, textureViews, textureCount, x, y, &bestSample);
  }
}

static bool c3dBuildTextureViews(C3DCommandBuffer* commandBuffer, const C3DRenderPassInfo* renderPass, C3DTextureView** textureViews)
{
  *textureViews = nullptr;
  if (renderPass->textureBindCount == 0)
  {
    return true;
  }

  C3DTextureView* hostTextureViews = (C3DTextureView*)malloc(renderPass->textureBindCount * sizeof(C3DTextureView));
  if (!hostTextureViews)
  {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate host texture view metadata");
    return false;
  }

  size_t requiredBytes = renderPass->textureBindCount * sizeof(C3DTextureView);
  if (!c3dTryResizeDeviceBuffer(&commandBuffer->textureViews, &commandBuffer->textureViewCap, requiredBytes, "cudaMalloc failed while allocating texture view metadata"))
  {
    free(hostTextureViews);
    return false;
  }

  *textureViews = (C3DTextureView*)commandBuffer->textureViews;

  for (size_t i = 0; i < renderPass->textureBindCount; ++i)
  {
    const C3DTextureBinding* binding = &renderPass->textureBindings[i];
    hostTextureViews[i].pixels = (const uint8_t*)binding->texture->data;
    hostTextureViews[i].width = binding->texture->info.width;
    hostTextureViews[i].height = binding->texture->info.height;
    hostTextureViews[i].format = binding->texture->info.format;
    hostTextureViews[i].sampler = binding->sampler;
  }

  bool success = c3dCheckCUDA(cudaMemcpy(*textureViews, hostTextureViews, renderPass->textureBindCount * sizeof(C3DTextureView), cudaMemcpyHostToDevice), "cudaMemcpy failed while uploading texture view metadata");
  free(hostTextureViews);
  return success;
}

static size_t c3dGetPrimitiveCount(const C3DDrawInfo* drawInfo)
{
  switch (drawInfo->topology)
  {
    case C3D_TOPOLOGY_LINE:
      return drawInfo->count / 2;
    case C3D_TOPOLOGY_QUAD:
      return drawInfo->count / 2;
    case C3D_TOPOLOGY_TRIANGLE:
      return drawInfo->count / 3;
  }

  return 0;
}

static bool c3dBuildTileBins(C3DCommandBuffer* commandBuffer, C3DTextureInfo targetInfo, size_t primitiveCount, bool triangles, const void* primitives, size_t* tilesXOut, size_t* tilesYOut)
{
  size_t tilesX = (targetInfo.width + (size_t)C3D_TILE_SIZE - 1) / (size_t)C3D_TILE_SIZE;
  size_t tilesY = (targetInfo.height + (size_t)C3D_TILE_SIZE - 1) / (size_t)C3D_TILE_SIZE;
  size_t tileCount = tilesX * tilesY;
  *tilesXOut = tilesX;
  *tilesYOut = tilesY;

  if (!c3dTryResizeDeviceBuffer((void**)&commandBuffer->tileCountsDevice, &commandBuffer->tileCountCap, tileCount * sizeof(uint32_t), "cudaMalloc failed while allocating tile count buffer"))
  {
    return false;
  }

  if (!c3dTryResizeDeviceBuffer((void**)&commandBuffer->tileOffsetsDevice, &commandBuffer->tileOffsetCap, (tileCount + 1) * sizeof(uint32_t), "cudaMalloc failed while allocating tile offset buffer"))
  {
    return false;
  }

  if (!c3dTryResizeHostBuffer(&commandBuffer->tileCountsHost, &commandBuffer->tileCountsHostCap, 1, "failed to allocate host tile total"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemset(commandBuffer->tileCountsDevice, 0, tileCount * sizeof(uint32_t)), "cudaMemset failed while clearing tile counts"))
  {
    return false;
  }

  const int binThreads = 128;
  const int binBlocks = (int)((primitiveCount + (size_t)binThreads - 1) / (size_t)binThreads);
  if (triangles)
  {
    c3dBinTrianglePrimitivesKernel<<<binBlocks, binThreads>>>((const C3DTrianglePrimitive*)primitives, primitiveCount, tilesX, commandBuffer->tileCountsDevice);
  }
  else
  {
    c3dBinLinePrimitivesKernel<<<binBlocks, binThreads>>>((const C3DLinePrimitive*)primitives, primitiveCount, tilesX, commandBuffer->tileCountsDevice);
  }

  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile bin count kernel launch failed"))
  {
    return false;
  }

  const int scanThreads = 256;
  const int scanBlocks = (int)((tileCount + (size_t)scanThreads - 1) / (size_t)scanThreads);
  uint32_t* scanInput = commandBuffer->tileCountsDevice;
  uint32_t* scanOutput = commandBuffer->tileOffsetsDevice;
  for (size_t stride = 1; stride < tileCount; stride *= 2)
  {
    c3dTileScanStepKernel<<<scanBlocks, scanThreads>>>(scanInput, scanOutput, tileCount, stride);
    if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile scan kernel launch failed"))
    {
      return false;
    }

    uint32_t* temp = scanInput;
    scanInput = scanOutput;
    scanOutput = temp;
  }

  c3dTileExclusiveShiftKernel<<<scanBlocks, scanThreads>>>(scanInput, commandBuffer->tileOffsetsDevice, tileCount);
  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile exclusive shift kernel launch failed"))
  {
    return false;
  }

  c3dFinalizeTileOffsetsKernel<<<1, 1>>>(commandBuffer->tileOffsetsDevice, commandBuffer->tileCountsDevice, tileCount);
  if (!c3dCheckCUDA(cudaPeekAtLastError(), "tile offset finalize kernel launch failed"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemcpy(commandBuffer->tileCountsHost, commandBuffer->tileOffsetsDevice + tileCount, sizeof(uint32_t), cudaMemcpyDeviceToHost), "cudaMemcpy failed while reading tile index count"))
  {
    return false;
  }

  uint32_t totalIndices = commandBuffer->tileCountsHost[0];
  if (!c3dTryResizeDeviceBuffer((void**)&commandBuffer->tileIndicesDevice, &commandBuffer->tileIndexCap, (size_t)totalIndices * sizeof(uint32_t), "cudaMalloc failed while allocating tile index buffer"))
  {
    return false;
  }

  if (!c3dCheckCUDA(cudaMemcpy(commandBuffer->tileCountsDevice, commandBuffer->tileOffsetsDevice, tileCount * sizeof(uint32_t), cudaMemcpyDeviceToDevice), "cudaMemcpy failed while preparing tile scatter cursors"))
  {
    return false;
  }

  if (triangles)
  {
    c3dScatterTrianglePrimitivesKernel<<<binBlocks, binThreads>>>((const C3DTrianglePrimitive*)primitives, primitiveCount, tilesX, commandBuffer->tileCountsDevice, commandBuffer->tileIndicesDevice);
  }
  else
  {
    c3dScatterLinePrimitivesKernel<<<binBlocks, binThreads>>>((const C3DLinePrimitive*)primitives, primitiveCount, tilesX, commandBuffer->tileCountsDevice, commandBuffer->tileIndicesDevice);
  }

  return c3dCheckCUDA(cudaPeekAtLastError(), "tile scatter kernel launch failed");
}

static bool c3dExecuteDraw(C3DCommandBuffer* commandBuffer, C3DTexture* target, const C3DRenderPassInfo* renderPass, const C3DTextureView* textureViews, const C3DDrawInfo* drawInfo, uint64_t* depthBuffer, uint32_t primitiveOrderBase, uint32_t* primitiveOrderCount)
{
  if (!c3dValidateDrawRanges(drawInfo))
  {
    return false;
  }

  C3DTextureInfo targetInfo = target->info;
  uint8_t* pixels = (uint8_t*)target->data;
  const uint8_t* indexData = drawInfo->indexBuffer->deviceData;
  const C3DVertex* vertexData = (const C3DVertex*)drawInfo->vertexBuffer->deviceData;
  size_t firstIndex = drawInfo->indexOffset / c3dGetIndexStride(drawInfo->indexSize);
  size_t primitiveCount = c3dGetPrimitiveCount(drawInfo);
  *primitiveOrderCount = (uint32_t)primitiveCount;

  dim3 rasterThreads(C3D_TILE_SIZE, C3D_TILE_SIZE);
  size_t tilesX = 0;
  size_t tilesY = 0;

  if (drawInfo->topology == C3D_TOPOLOGY_LINE)
  {
    if ((drawInfo->count % 2) != 0)
    {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "line draws require an even index count");
      return false;
    }

    if (primitiveCount == 0)
    {
      return true;
    }

    if (!c3dTryResizeDeviceBuffer(&commandBuffer->linePrimitives, &commandBuffer->linePrimitiveCap, primitiveCount * sizeof(C3DLinePrimitive), "cudaMalloc failed while allocating line primitive buffer"))
    {
      return false;
    }

    C3DLinePrimitive* primitives = (C3DLinePrimitive*)commandBuffer->linePrimitives;
    const int setupThreads = 128;
    const int setupBlocks = (int)((primitiveCount + (size_t)setupThreads - 1) / (size_t)setupThreads);
    c3dSetupLinePrimitivesKernel<<<setupBlocks, setupThreads>>>(primitives, targetInfo, renderPass->viewport, indexData, drawInfo->indexSize, firstIndex, drawInfo->indexBase, vertexData, drawInfo->vertexOffset, primitiveCount, primitiveOrderBase);
    bool success = c3dCheckCUDA(cudaPeekAtLastError(), "line setup kernel launch failed");
    if (success)
    {
      success = c3dBuildTileBins(commandBuffer, targetInfo, primitiveCount, false, primitives, &tilesX, &tilesY);
    }
    if (success)
    {
      dim3 rasterBlocks((unsigned int)tilesX, (unsigned int)tilesY);
      c3dRasterizeLinesKernel<<<rasterBlocks, rasterThreads>>>(targetInfo, pixels, depthBuffer, renderPass->targetBlend, textureViews, renderPass->textureBindCount, primitives, tilesX, commandBuffer->tileOffsetsDevice, commandBuffer->tileIndicesDevice);
      success = c3dCheckCUDA(cudaPeekAtLastError(), "line raster kernel launch failed");
    }

    return success;
  }

  if (drawInfo->topology == C3D_TOPOLOGY_QUAD && (drawInfo->count % 4) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "quad draws require an index count divisible by four");
    return false;
  }

  if (drawInfo->topology == C3D_TOPOLOGY_TRIANGLE && (drawInfo->count % 3) != 0)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "triangle draws require an index count divisible by three");
    return false;
  }

  if (primitiveCount == 0)
  {
    return true;
  }

  if (!c3dTryResizeDeviceBuffer(&commandBuffer->trianglePrimitives, &commandBuffer->trianglePrimitiveCap, primitiveCount * sizeof(C3DTrianglePrimitive), "cudaMalloc failed while allocating triangle primitive buffer"))
  {
    return false;
  }

  C3DTrianglePrimitive* primitives = (C3DTrianglePrimitive*)commandBuffer->trianglePrimitives;
  const int setupThreads = 128;
  const int setupBlocks = (int)((primitiveCount + (size_t)setupThreads - 1) / (size_t)setupThreads);
  c3dSetupTrianglePrimitivesKernel<<<setupBlocks, setupThreads>>>(primitives, targetInfo, renderPass->viewport, drawInfo->topology, indexData, drawInfo->indexSize, firstIndex, drawInfo->indexBase, vertexData, drawInfo->vertexOffset, primitiveCount, primitiveOrderBase);
  bool success = c3dCheckCUDA(cudaPeekAtLastError(), "triangle setup kernel launch failed");
  if (success)
  {
    success = c3dBuildTileBins(commandBuffer, targetInfo, primitiveCount, true, primitives, &tilesX, &tilesY);
  }
  if (success)
  {
    dim3 rasterBlocks((unsigned int)tilesX, (unsigned int)tilesY);
    c3dRasterizeTrianglesKernel<<<rasterBlocks, rasterThreads>>>(targetInfo, pixels, depthBuffer, renderPass->targetBlend, textureViews, renderPass->textureBindCount, primitives, tilesX, commandBuffer->tileOffsetsDevice, commandBuffer->tileIndicesDevice);
    success = c3dCheckCUDA(cudaPeekAtLastError(), "triangle raster kernel launch failed");
  }

  return success;
}

C3D_API bool c3dSubmitCommandBuffer(C3DCommandBuffer* commandBuffer)
{
  if (!commandBuffer)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (commandBuffer->inRenderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "cannot submit a command buffer while a render pass is active");
    return false;
  }

  if (!commandBuffer->hasRenderPass)
  {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no recorded render pass to submit");
    return false;
  }

  C3DTexture* target = commandBuffer->renderPass.target;
  size_t pixelCount = target->info.width * target->info.height;
  uint64_t* depthBuffer = nullptr;
  C3DTextureView* textureViews = nullptr;
  bool success = true;
  if (commandBuffer->renderPass.depthTarget)
  {
    depthBuffer = (uint64_t*)commandBuffer->renderPass.depthTarget->data;
  }

  if (success)
  {
    success = c3dBuildTextureViews(commandBuffer, &commandBuffer->renderPass, &textureViews);
  }

  if (success && depthBuffer)
  {
    const int clearThreads = 256;
    const int clearBlocks = (int)((pixelCount + (size_t)clearThreads - 1) / (size_t)clearThreads);
    c3dClearDepthBufferKernel<<<clearBlocks, clearThreads>>>(depthBuffer, pixelCount);
    success = c3dCheckCUDA(cudaPeekAtLastError(), "depth clear kernel launch failed");
  }

  uint32_t primitiveOrderBase = 0;
  if (success)
  {
    for (size_t i = 0; i < commandBuffer->drawCount; ++i)
    {
      uint32_t primitiveOrderCount = 0;
      if (!c3dExecuteDraw(commandBuffer, target, &commandBuffer->renderPass, textureViews, &commandBuffer->draws[i].info, depthBuffer, primitiveOrderBase, &primitiveOrderCount))
      {
        success = false;
        break;
      }

      primitiveOrderBase += primitiveOrderCount;
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

  c3dResetCommandBuffer(commandBuffer);
  return true;
}
