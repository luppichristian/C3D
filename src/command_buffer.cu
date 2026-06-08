#include <C3D.h>

#include "c3d_internal.h"

#include <cuda_runtime.h>

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct C3DTextureView {
  const uint8_t* pixels;
  size_t width;
  size_t height;
  C3DTextureFormat format;
};

struct C3DPixel {
  float r;
  float g;
  float b;
  float a;
};

struct C3DVertexSample {
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

static void c3dFreeTextureViews(size_t count, C3DTextureView* texture_views, uint8_t** texture_buffers);

static thread_local char g_cuda_error_desc[512];

static bool c3dCheckCUDA(cudaError_t error, const char* desc) {
  if (error == cudaSuccess) {
    return true;
  }

  snprintf(g_cuda_error_desc, sizeof(g_cuda_error_desc), "%s: %s (%d)", desc, cudaGetErrorString(error), (int)error);
  c3dThrowError(C3D_ERROR_CUDA, g_cuda_error_desc);
  return false;
}

static size_t c3dGetTextureTexelSize(C3DTextureFormat format) {
  switch (format) {
    case C3D_TEXTURE_FORMAT_RGBA8:
    case C3D_TEXTURE_FORMAT_BGRA8:
      return 4;
  }

  return 0;
}

static size_t c3dGetIndexStride(C3DIndexSize index_size) {
  switch (index_size) {
    case C3D_INDEX_SIZE_8:
      return 1;
    case C3D_INDEX_SIZE_16:
      return 2;
    case C3D_INDEX_SIZE_32:
      return 4;
  }

  return 0;
}

static float c3dClamp01(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }

  if (value > 1.0f) {
    return 1.0f;
  }

  return value;
}

static uint8_t c3dFloatToByte(float value) {
  float scaled = c3dClamp01(value) * 255.0f;
  return (uint8_t)(scaled + 0.5f);
}

static C3DPixel c3dUnpackPixel(const uint8_t* texel, C3DTextureFormat format) {
  C3DPixel pixel = {0};
  switch (format) {
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

static void c3dPackPixel(uint8_t* texel, C3DTextureFormat format, C3DPixel pixel) {
  uint8_t r = c3dFloatToByte(pixel.r);
  uint8_t g = c3dFloatToByte(pixel.g);
  uint8_t b = c3dFloatToByte(pixel.b);
  uint8_t a = c3dFloatToByte(pixel.a);

  switch (format) {
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

static C3DPixel c3dMulPixel(C3DPixel a, C3DPixel b) {
  C3DPixel pixel = {0};
  pixel.r = a.r * b.r;
  pixel.g = a.g * b.g;
  pixel.b = a.b * b.b;
  pixel.a = a.a * b.a;
  return pixel;
}

static C3DPixel c3dBlendPixel(C3DPixel source, C3DPixel destination, C3DBlendMode blend_mode) {
  switch (blend_mode) {
    case C3D_BLEND_MODE_NONE:
      return source;
    case C3D_BLEND_MODE_NORMAL: {
      C3DPixel pixel = {0};
      float inv_alpha = 1.0f - source.a;
      pixel.r = source.r + (destination.r * inv_alpha);
      pixel.g = source.g + (destination.g * inv_alpha);
      pixel.b = source.b + (destination.b * inv_alpha);
      pixel.a = source.a + (destination.a * inv_alpha);
      return pixel;
    }
    case C3D_BLEND_MODE_ADDITIVE: {
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

static uint8_t* c3dGetTargetTexel(C3DTexture* texture, uint8_t* pixels, size_t x, size_t y) {
  size_t texel_size = c3dGetTextureTexelSize(texture->info.format);
  return pixels + (((y * texture->info.width) + x) * texel_size);
}

static float c3dWrapCoord(float value) {
  float wrapped = value - floorf(value);
  if (wrapped < 0.0f) {
    wrapped += 1.0f;
  }

  return wrapped;
}

static float c3dClampCoord(float value) {
  if (value < 0.0f) {
    return 0.0f;
  }

  if (value > 1.0f) {
    return 1.0f;
  }

  return value;
}

static float c3dApplySamplerAddress(float value, C3DSampler sampler) {
  switch (sampler) {
    case C3D_SAMPLER_POINT_WRAP:
    case C3D_SAMPLER_LINEAR_WRAP:
      return c3dWrapCoord(value);
    case C3D_SAMPLER_POINT_CLAMP:
    case C3D_SAMPLER_LINEAR_CLAMP:
      return c3dClampCoord(value);
  }

  return value;
}

static size_t c3dRoundToIndex(float value, size_t limit) {
  if (limit == 0) {
    return 0;
  }

  if (value <= 0.0f) {
    return 0;
  }

  size_t index = (size_t)(value + 0.5f);
  size_t max_index = limit - 1;
  return index > max_index ? max_index : index;
}

static C3DPixel c3dSampleTextureNearest(const C3DTextureView* texture, C3DSampler sampler, float u, float v) {
  float sample_u = c3dApplySamplerAddress(u, sampler);
  float sample_v = c3dApplySamplerAddress(v, sampler);
  float x = sample_u * (float)(texture->width - 1);
  float y = sample_v * (float)(texture->height - 1);
  size_t xi = c3dRoundToIndex(x, texture->width);
  size_t yi = c3dRoundToIndex(y, texture->height);
  const uint8_t* texel = texture->pixels + (((yi * texture->width) + xi) * 4);
  return c3dUnpackPixel(texel, texture->format);
}

static C3DPixel c3dLerpPixel(C3DPixel a, C3DPixel b, float t) {
  C3DPixel pixel = {0};
  pixel.r = a.r + ((b.r - a.r) * t);
  pixel.g = a.g + ((b.g - a.g) * t);
  pixel.b = a.b + ((b.b - a.b) * t);
  pixel.a = a.a + ((b.a - a.a) * t);
  return pixel;
}

static C3DPixel c3dSampleTextureLinear(const C3DTextureView* texture, C3DSampler sampler, float u, float v) {
  float sample_u = c3dApplySamplerAddress(u, sampler);
  float sample_v = c3dApplySamplerAddress(v, sampler);
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

static C3DPixel c3dSampleBoundTexture(const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, int texid, float u, float v) {
  if (texid < 0 || (size_t)texid >= render_pass->textureBindCount) {
    C3DPixel white = {1.0f, 1.0f, 1.0f, 1.0f};
    return white;
  }

  const C3DTextureBinding* binding = &render_pass->textureBindings[texid];
  const C3DTextureView* texture = &texture_views[texid];
  switch (binding->sampler) {
    case C3D_SAMPLER_POINT_CLAMP:
    case C3D_SAMPLER_POINT_WRAP:
      return c3dSampleTextureNearest(texture, binding->sampler, u, v);
    case C3D_SAMPLER_LINEAR_CLAMP:
    case C3D_SAMPLER_LINEAR_WRAP:
      return c3dSampleTextureLinear(texture, binding->sampler, u, v);
  }

  C3DPixel white = {1.0f, 1.0f, 1.0f, 1.0f};
  return white;
}

static bool c3dTryReadTextureToHost(C3DTexture* texture, uint8_t** buffer) {
  *buffer = nullptr;
  if (texture->size == 0) {
    return true;
  }

  *buffer = (uint8_t*)malloc(texture->size);
  if (!*buffer) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate host texture staging buffer");
    return false;
  }

  if (!c3dCheckCUDA(cudaMemcpy(*buffer, texture->data, texture->size, cudaMemcpyDeviceToHost), "cudaMemcpy failed while reading texture for rendering")) {
    free(*buffer);
    *buffer = nullptr;
    return false;
  }

  return true;
}

static bool c3dTryWriteTextureToDevice(C3DTexture* texture, const uint8_t* buffer) {
  if (texture->size == 0) {
    return true;
  }

  return c3dCheckCUDA(cudaMemcpy(texture->data, buffer, texture->size, cudaMemcpyHostToDevice), "cudaMemcpy failed while writing rendered texture");
}

static size_t c3dReadIndexValue(const C3DIndexBuffer* index_buffer, size_t index) {
  const uint8_t* ptr = index_buffer->data + (index * c3dGetIndexStride(index_buffer->info.indexSize));
  switch (index_buffer->info.indexSize) {
    case C3D_INDEX_SIZE_8:
      return (size_t)(*(const uint8_t*)ptr);
    case C3D_INDEX_SIZE_16:
      return (size_t)(*(const uint16_t*)ptr);
    case C3D_INDEX_SIZE_32:
      return (size_t)(*(const uint32_t*)ptr);
  }

  return 0;
}

static bool c3dValidateDrawRanges(const C3DDrawInfo* draw_info) {
  size_t index_stride = c3dGetIndexStride(draw_info->indexBuffer->info.indexSize);
  size_t index_start = draw_info->indexOffset;
  size_t index_bytes = draw_info->count * index_stride;
  if (index_start > draw_info->indexBuffer->size || index_bytes > draw_info->indexBuffer->size - index_start) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw index range is out of bounds");
    return false;
  }

  size_t vertex_limit = draw_info->vertexBuffer->info.vertexCap;
  for (size_t i = 0; i < draw_info->count; ++i) {
    size_t index = c3dReadIndexValue(draw_info->indexBuffer, (draw_info->indexOffset / index_stride) + i);
    size_t vertex_index = draw_info->vertexOffset + draw_info->indexBase + index;
    if (vertex_index >= vertex_limit) {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw vertex range is out of bounds");
      return false;
    }
  }

  return true;
}

static C3DVertexSample c3dLoadVertex(const C3DDrawInfo* draw_info, size_t draw_index) {
  size_t index_stride = c3dGetIndexStride(draw_info->indexBuffer->info.indexSize);
  size_t index = c3dReadIndexValue(draw_info->indexBuffer, (draw_info->indexOffset / index_stride) + draw_index);
  size_t vertex_index = draw_info->vertexOffset + draw_info->indexBase + index;
  const C3DVertex* vertex = ((const C3DVertex*)draw_info->vertexBuffer->data) + vertex_index;

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

static void c3dNdcToScreen(const C3DTexture* target, C3DVertexSample* vertex) {
  float inv_w = vertex->w != 0.0f ? (1.0f / vertex->w) : 1.0f;
  float x = vertex->x * inv_w;
  float y = vertex->y * inv_w;
  vertex->x = ((x * 0.5f) + 0.5f) * (float)(target->info.width - 1);
  vertex->y = ((1.0f - (y * 0.5f + 0.5f))) * (float)(target->info.height - 1);
}

static void c3dWritePixel(C3DTexture* target, uint8_t* pixels, size_t x, size_t y, C3DPixel color, C3DBlendMode blend_mode) {
  uint8_t* texel = c3dGetTargetTexel(target, pixels, x, y);
  C3DPixel destination = c3dUnpackPixel(texel, target->info.format);
  C3DPixel blended = c3dBlendPixel(color, destination, blend_mode);
  c3dPackPixel(texel, target->info.format, blended);
}

static void c3dShadePixel(C3DTexture* target, uint8_t* pixels, const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, size_t x, size_t y, const C3DVertexSample* sample) {
  C3DPixel vertex_color = {sample->r, sample->g, sample->b, sample->a};
  C3DPixel texture_color = c3dSampleBoundTexture(render_pass, texture_views, sample->texid, sample->u, sample->v);
  c3dWritePixel(target, pixels, x, y, c3dMulPixel(vertex_color, texture_color), render_pass->targetBlend);
}

static C3DVertexSample c3dLerpVertex(const C3DVertexSample* a, const C3DVertexSample* b, float t) {
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

static void c3dRasterizeLine(C3DTexture* target, uint8_t* pixels, const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, C3DVertexSample a, C3DVertexSample b) {
  c3dNdcToScreen(target, &a);
  c3dNdcToScreen(target, &b);

  float dx = b.x - a.x;
  float dy = b.y - a.y;
  float length = fabsf(dx) > fabsf(dy) ? fabsf(dx) : fabsf(dy);
  int steps = (int)length;
  if (steps < 1) {
    steps = 1;
  }

  for (int i = 0; i <= steps; ++i) {
    float t = (float)i / (float)steps;
    C3DVertexSample sample = c3dLerpVertex(&a, &b, t);
    long x = lroundf(sample.x);
    long y = lroundf(sample.y);
    if (x >= 0 && y >= 0 && (size_t)x < target->info.width && (size_t)y < target->info.height) {
      c3dShadePixel(target, pixels, render_pass, texture_views, (size_t)x, (size_t)y, &sample);
    }
  }
}

static float c3dEdge(float ax, float ay, float bx, float by, float px, float py) {
  return ((px - ax) * (by - ay)) - ((py - ay) * (bx - ax));
}

static void c3dRasterizeTriangle(C3DTexture* target, uint8_t* pixels, const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, C3DVertexSample a, C3DVertexSample b, C3DVertexSample c) {
  c3dNdcToScreen(target, &a);
  c3dNdcToScreen(target, &b);
  c3dNdcToScreen(target, &c);

  float area = c3dEdge(a.x, a.y, b.x, b.y, c.x, c.y);
  if (area == 0.0f) {
    return;
  }

  float min_xf = floorf(fminf(a.x, fminf(b.x, c.x)));
  float min_yf = floorf(fminf(a.y, fminf(b.y, c.y)));
  float max_xf = ceilf(fmaxf(a.x, fmaxf(b.x, c.x)));
  float max_yf = ceilf(fmaxf(a.y, fmaxf(b.y, c.y)));

  long min_x = (long)min_xf;
  long min_y = (long)min_yf;
  long max_x = (long)max_xf;
  long max_y = (long)max_yf;

  if (min_x < 0) min_x = 0;
  if (min_y < 0) min_y = 0;
  if (max_x >= (long)target->info.width) max_x = (long)target->info.width - 1;
  if (max_y >= (long)target->info.height) max_y = (long)target->info.height - 1;

  for (long y = min_y; y <= max_y; ++y) {
    for (long x = min_x; x <= max_x; ++x) {
      float px = (float)x + 0.5f;
      float py = (float)y + 0.5f;
      float w0 = c3dEdge(b.x, b.y, c.x, c.y, px, py);
      float w1 = c3dEdge(c.x, c.y, a.x, a.y, px, py);
      float w2 = c3dEdge(a.x, a.y, b.x, b.y, px, py);
      bool inside = area > 0.0f ? (w0 >= 0.0f && w1 >= 0.0f && w2 >= 0.0f) : (w0 <= 0.0f && w1 <= 0.0f && w2 <= 0.0f);
      if (!inside) {
        continue;
      }

      float inv_area = 1.0f / area;
      float wa = w0 * inv_area;
      float wb = w1 * inv_area;
      float wc = w2 * inv_area;

      C3DVertexSample sample = {0};
      sample.r = (a.r * wa) + (b.r * wb) + (c.r * wc);
      sample.g = (a.g * wa) + (b.g * wb) + (c.g * wc);
      sample.b = (a.b * wa) + (b.b * wb) + (c.b * wc);
      sample.a = (a.a * wa) + (b.a * wb) + (c.a * wc);
      sample.u = (a.u * wa) + (b.u * wb) + (c.u * wc);
      sample.v = (a.v * wa) + (b.v * wb) + (c.v * wc);
      sample.texid = a.texid >= 0 ? a.texid : (b.texid >= 0 ? b.texid : c.texid);

      c3dShadePixel(target, pixels, render_pass, texture_views, (size_t)x, (size_t)y, &sample);
    }
  }
}

static bool c3dExecuteDraw(C3DTexture* target, uint8_t* pixels, const C3DRenderPassInfo* render_pass, const C3DTextureView* texture_views, const C3DDrawInfo* draw_info) {
  if (!c3dValidateDrawRanges(draw_info)) {
    return false;
  }

  switch (draw_info->topology) {
    case C3D_TOPOLOGY_LINE:
      if ((draw_info->count % 2) != 0) {
        c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "line draws require an even index count");
        return false;
      }

      for (size_t i = 0; i < draw_info->count; i += 2) {
        C3DVertexSample a = c3dLoadVertex(draw_info, i + 0);
        C3DVertexSample b = c3dLoadVertex(draw_info, i + 1);
        c3dRasterizeLine(target, pixels, render_pass, texture_views, a, b);
      }
      return true;
    case C3D_TOPOLOGY_QUAD:
      if ((draw_info->count % 4) != 0) {
        c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "quad draws require an index count divisible by four");
        return false;
      }

      for (size_t i = 0; i < draw_info->count; i += 4) {
        C3DVertexSample a = c3dLoadVertex(draw_info, i + 0);
        C3DVertexSample b = c3dLoadVertex(draw_info, i + 1);
        C3DVertexSample c = c3dLoadVertex(draw_info, i + 2);
        C3DVertexSample d = c3dLoadVertex(draw_info, i + 3);
        c3dRasterizeTriangle(target, pixels, render_pass, texture_views, a, b, c);
        c3dRasterizeTriangle(target, pixels, render_pass, texture_views, a, c, d);
      }
      return true;
    case C3D_TOPOLOGY_TRIANGLE:
      if ((draw_info->count % 3) != 0) {
        c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "triangle draws require an index count divisible by three");
        return false;
      }

      for (size_t i = 0; i < draw_info->count; i += 3) {
        C3DVertexSample a = c3dLoadVertex(draw_info, i + 0);
        C3DVertexSample b = c3dLoadVertex(draw_info, i + 1);
        C3DVertexSample c = c3dLoadVertex(draw_info, i + 2);
        c3dRasterizeTriangle(target, pixels, render_pass, texture_views, a, b, c);
      }
      return true;
  }

  c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw topology is invalid");
  return false;
}

static bool c3dBuildTextureViews(const C3DRenderPassInfo* render_pass, C3DTextureView** texture_views, uint8_t*** texture_buffers) {
  *texture_views = nullptr;
  *texture_buffers = nullptr;

  if (render_pass->textureBindCount == 0) {
    return true;
  }

  *texture_views = (C3DTextureView*)calloc(render_pass->textureBindCount, sizeof(C3DTextureView));
  *texture_buffers = (uint8_t**)calloc(render_pass->textureBindCount, sizeof(uint8_t*));
  if (!*texture_views || !*texture_buffers) {
    free(*texture_views);
    free(*texture_buffers);
    *texture_views = nullptr;
    *texture_buffers = nullptr;
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture binding staging state");
    return false;
  }

  for (size_t i = 0; i < render_pass->textureBindCount; ++i) {
    C3DTexture* texture = render_pass->textureBindings[i].texture;
    if (!c3dTryReadTextureToHost(texture, &(*texture_buffers)[i])) {
      c3dFreeTextureViews(render_pass->textureBindCount, *texture_views, *texture_buffers);
      *texture_views = nullptr;
      *texture_buffers = nullptr;
      return false;
    }

    (*texture_views)[i].pixels = (*texture_buffers)[i];
    (*texture_views)[i].width = texture->info.width;
    (*texture_views)[i].height = texture->info.height;
    (*texture_views)[i].format = texture->info.format;
  }

  return true;
}

static void c3dFreeTextureViews(size_t count, C3DTextureView* texture_views, uint8_t** texture_buffers) {
  if (texture_buffers) {
    for (size_t i = 0; i < count; ++i) {
      free(texture_buffers[i]);
    }
  }

  free(texture_views);
  free(texture_buffers);
}

static bool c3dTryGrowDrawList(C3DCommandBuffer* command_buffer, size_t min_cap) {
  if (command_buffer->draw_cap >= min_cap) {
    return true;
  }

  size_t new_cap = command_buffer->draw_cap == 0 ? 4 : command_buffer->draw_cap * 2;
  while (new_cap < min_cap) {
    if (new_cap > (SIZE_MAX / 2)) {
      new_cap = min_cap;
      break;
    }

    new_cap *= 2;
  }

  if (new_cap > (SIZE_MAX / sizeof(C3DRecordedDraw))) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw list size overflows size_t");
    return false;
  }

  C3DRecordedDraw* draws = (C3DRecordedDraw*)realloc(command_buffer->draws, new_cap * sizeof(C3DRecordedDraw));
  if (!draws) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to grow command buffer draw list");
    return false;
  }

  command_buffer->draws = draws;
  command_buffer->draw_cap = new_cap;
  return true;
}

static void c3dResetRenderPassInfo(C3DRenderPassInfo* render_pass) {
  free(render_pass->textureBindings);
  memset(render_pass, 0, sizeof(*render_pass));
}

static void c3dResetCommandBuffer(C3DCommandBuffer* command_buffer) {
  command_buffer->in_render_pass = false;
  command_buffer->has_render_pass = false;
  command_buffer->draw_count = 0;
  c3dResetRenderPassInfo(&command_buffer->render_pass);
}

static bool c3dIsValidSampler(C3DSampler sampler) {
  return sampler >= C3D_SAMPLER_POINT_CLAMP && sampler <= C3D_SAMPLER_LINEAR_WRAP;
}

static bool c3dIsValidTopology(C3DTopology topology) {
  return topology >= C3D_TOPOLOGY_LINE && topology <= C3D_TOPOLOGY_TRIANGLE;
}

static bool c3dIsValidBlendMode(C3DBlendMode blend_mode) {
  return blend_mode >= C3D_BLEND_MODE_NONE && blend_mode <= C3D_BLEND_MODE_ADDITIVE;
}

static bool c3dValidateRenderPass(const C3DRenderPassInfo* render_pass) {
  if (!render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass info must be non-null");
    return false;
  }

  if (!render_pass->target) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass target must be non-null");
    return false;
  }

  if (!c3dIsValidBlendMode(render_pass->targetBlend)) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "render pass blend mode is invalid");
    return false;
  }

  if (render_pass->textureBindCount != 0 && !render_pass->textureBindings) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture bindings must be non-null when textureBindCount is non-zero");
    return false;
  }

  for (size_t i = 0; i < render_pass->textureBindCount; ++i) {
    const C3DTextureBinding* binding = &render_pass->textureBindings[i];
    if (!binding->texture) {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding texture must be non-null");
      return false;
    }

    if (!c3dIsValidSampler(binding->sampler)) {
      c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding sampler is invalid");
      return false;
    }
  }

  return true;
}

static bool c3dCopyRenderPassInfo(C3DRenderPassInfo* destination, const C3DRenderPassInfo* source) {
  *destination = *source;
  destination->textureBindings = nullptr;

  if (source->textureBindCount == 0) {
    return true;
  }

  if (source->textureBindCount > (SIZE_MAX / sizeof(C3DTextureBinding))) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "texture binding list size overflows size_t");
    return false;
  }

  destination->textureBindings = (C3DTextureBinding*)malloc(source->textureBindCount * sizeof(C3DTextureBinding));
  if (!destination->textureBindings) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate texture binding list");
    return false;
  }

  memcpy(destination->textureBindings, source->textureBindings, source->textureBindCount * sizeof(C3DTextureBinding));
  return true;
}

static bool c3dValidateDrawInfo(const C3DDrawInfo* draw_info) {
  if (!draw_info) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw info must be non-null");
    return false;
  }

  if (!c3dIsValidTopology(draw_info->topology)) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw topology is invalid");
    return false;
  }

  if (!draw_info->indexBuffer || !draw_info->vertexBuffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw buffers must be non-null");
    return false;
  }

  return true;
}

C3D_API C3DCommandBuffer* c3dCreateCommandBuffer(void) {
  C3DCommandBuffer* command_buffer = new C3DCommandBuffer {};
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_OUT_OF_MEMORY, "failed to allocate command buffer object");
    return nullptr;
  }

  return command_buffer;
}

C3D_API bool c3dDeleteCommandBuffer(C3DCommandBuffer* command_buffer) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  c3dResetRenderPassInfo(&command_buffer->render_pass);
  free(command_buffer->draws);
  delete command_buffer;
  return true;
}

C3D_API bool c3dSubmitCommandBuffer(C3DCommandBuffer* command_buffer) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (command_buffer->in_render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "cannot submit a command buffer while a render pass is active");
    return false;
  }

  if (!command_buffer->has_render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no recorded render pass to submit");
    return false;
  }

  C3DTexture* target = command_buffer->render_pass.target;
  uint8_t* target_pixels = nullptr;
  C3DTextureView* texture_views = nullptr;
  uint8_t** texture_buffers = nullptr;
  bool success = c3dTryReadTextureToHost(target, &target_pixels)
      && c3dBuildTextureViews(&command_buffer->render_pass, &texture_views, &texture_buffers);

  if (success) {
    for (size_t i = 0; i < command_buffer->draw_count; ++i) {
      if (!c3dExecuteDraw(target, target_pixels, &command_buffer->render_pass, texture_views, &command_buffer->draws[i].info)) {
        success = false;
        break;
      }
    }
  }

  if (success) {
    success = c3dTryWriteTextureToDevice(target, target_pixels);
  }

  free(target_pixels);
  c3dFreeTextureViews(command_buffer->render_pass.textureBindCount, texture_views, texture_buffers);

  if (!success) {
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  return true;
}

C3D_API bool c3dCancelCommandBuffer(C3DCommandBuffer* command_buffer) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  return true;
}

C3D_API bool c3dBeginRenderPass(C3DCommandBuffer* command_buffer, C3DRenderPassInfo* render_pass) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (command_buffer->in_render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer already has an active render pass");
    return false;
  }

  if (!c3dValidateRenderPass(render_pass)) {
    return false;
  }

  c3dResetCommandBuffer(command_buffer);
  if (!c3dCopyRenderPassInfo(&command_buffer->render_pass, render_pass)) {
    return false;
  }

  command_buffer->in_render_pass = true;
  command_buffer->has_render_pass = true;
  return true;
}

C3D_API bool c3dEndRenderPass(C3DCommandBuffer* command_buffer) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (!command_buffer->in_render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer has no active render pass");
    return false;
  }

  command_buffer->in_render_pass = false;
  return true;
}

C3D_API bool c3dDraw(C3DCommandBuffer* command_buffer, const C3DDrawInfo* draw_info) {
  if (!command_buffer) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "command buffer must be non-null");
    return false;
  }

  if (!command_buffer->in_render_pass) {
    c3dThrowError(C3D_ERROR_INVALID_ARGUMENT, "draw commands require an active render pass");
    return false;
  }

  if (!c3dValidateDrawInfo(draw_info)) {
    return false;
  }

  if (!c3dTryGrowDrawList(command_buffer, command_buffer->draw_count + 1)) {
    return false;
  }

  command_buffer->draws[command_buffer->draw_count].info = *draw_info;
  command_buffer->draw_count += 1;
  return true;
}
