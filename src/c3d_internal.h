#pragma once

#include <C3D.h>

#include <stdint.h>

struct C3DTexture {
  C3DTextureInfo info;
  void* data;
  size_t size;
};

struct C3DIndexBuffer {
  C3DIndexBufferInfo info;
  uint8_t* data;
  size_t size;
};

struct C3DVertexBuffer {
  C3DVertexBufferInfo info;
  uint8_t* data;
  size_t size;
};

struct C3DRecordedDraw {
  C3DDrawInfo info;
};

struct C3DCommandBuffer {
  bool in_render_pass;
  bool has_render_pass;
  C3DRenderPassInfo render_pass;
  C3DRecordedDraw* draws;
  size_t draw_count;
  size_t draw_cap;
};
