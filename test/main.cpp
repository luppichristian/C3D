#define WIN32_LEAN_AND_MEAN
#include <C3D.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <tracy/TracyC.h>
#include <windows.h>

typedef struct
{
  bool initialized;
  C3DTexture* texture;
  C3DTexture* depth_texture;
  C3DBuffer* quad_indices;
  C3DBuffer* quad_vertices;
  C3DBuffer* line_indices;
  C3DBuffer* line_vertices;
  C3DStageBuffer* quad_upload_stage;
  C3DStageBuffer* line_upload_stage;
  C3DStageBuffer* readback_stage;
  C3DCommandBuffer* command_buffer;
} RenderState;

static RenderState g_render_state = {0};
static bool g_error_dialog_shown = false;

static void c3dErrorCallback(C3DErrorID id, const char* desc, C3DErrorLoc loc)
{
  char buffer[1024];
  wsprintfA(
      buffer,
      "C3D error %d\n%s\n%s:%lu\n%s\n",
      (int)id,
      desc ? desc : "",
      loc.filename ? loc.filename : "<unknown>",
      (unsigned long)loc.line,
      loc.function ? loc.function : "<unknown>");
  OutputDebugStringA(buffer);

  if (!g_error_dialog_shown)
  {
    g_error_dialog_shown = true;
    MessageBoxA(NULL, buffer, "C3D Error", MB_OK | MB_ICONERROR);
  }
}

static void clearFallback(C3DTexture* texture, uint8_t r, uint8_t g, uint8_t b)
{
  uint8_t clear_color[4] = {r, g, b, 255};
  c3dClearTexture(texture, clear_color);
}

static bool ensureStageBufferSize(C3DStageBuffer** stage_buffer, size_t size)
{
  if (!*stage_buffer)
  {
    C3DStageBufferInfo info = {0};
    info.size = size;
    *stage_buffer = c3dCreateStageBuffer(&info);
    return *stage_buffer != NULL;
  }

  C3DStageBufferInfo info = {0};
  if (!c3dGetStageBufferInfo(*stage_buffer, &info))
  {
    return false;
  }

  return info.size == size || c3dResizeStageBuffer(*stage_buffer, size);
}

static bool ensureDepthTexture(size_t width, size_t height)
{
  C3DTextureInfo info = {0};
  if (g_render_state.depth_texture && c3dGetTextureInfo(g_render_state.depth_texture, &info) && info.width == width && info.height == height && info.format == C3D_TEXTURE_FORMAT_DEPTH64)
  {
    return true;
  }

  if (g_render_state.depth_texture)
  {
    c3dDeleteTexture(g_render_state.depth_texture);
    g_render_state.depth_texture = NULL;
  }

  info.width = width;
  info.height = height;
  info.depth = 1;
  info.format = C3D_TEXTURE_FORMAT_DEPTH64;
  g_render_state.depth_texture = c3dCreateTexture(&info);
  return g_render_state.depth_texture != NULL;
}

static void destroyRenderState(void)
{
  if (g_render_state.command_buffer)
  {
    c3dDeleteCommandBuffer(g_render_state.command_buffer);
    g_render_state.command_buffer = NULL;
  }

  if (g_render_state.readback_stage)
  {
    c3dDeleteStageBuffer(g_render_state.readback_stage);
    g_render_state.readback_stage = NULL;
  }

  if (g_render_state.line_upload_stage)
  {
    c3dDeleteStageBuffer(g_render_state.line_upload_stage);
    g_render_state.line_upload_stage = NULL;
  }

  if (g_render_state.quad_upload_stage)
  {
    c3dDeleteStageBuffer(g_render_state.quad_upload_stage);
    g_render_state.quad_upload_stage = NULL;
  }

  if (g_render_state.line_vertices)
  {
    c3dDeleteBuffer(g_render_state.line_vertices);
    g_render_state.line_vertices = NULL;
  }

  if (g_render_state.line_indices)
  {
    c3dDeleteBuffer(g_render_state.line_indices);
    g_render_state.line_indices = NULL;
  }

  if (g_render_state.quad_vertices)
  {
    c3dDeleteBuffer(g_render_state.quad_vertices);
    g_render_state.quad_vertices = NULL;
  }

  if (g_render_state.quad_indices)
  {
    c3dDeleteBuffer(g_render_state.quad_indices);
    g_render_state.quad_indices = NULL;
  }

  if (g_render_state.depth_texture)
  {
    c3dDeleteTexture(g_render_state.depth_texture);
    g_render_state.depth_texture = NULL;
  }

  if (g_render_state.texture)
  {
    c3dDeleteTexture(g_render_state.texture);
    g_render_state.texture = NULL;
  }

  g_render_state.initialized = false;
}

static bool createCheckerTexture(void)
{
  C3DTextureInfo texture_info = {0};
  texture_info.width = 64;
  texture_info.height = 64;
  texture_info.depth = 1;
  texture_info.format = C3D_TEXTURE_FORMAT_RGBA8;
  g_render_state.texture = c3dCreateTexture(&texture_info);
  if (!g_render_state.texture)
  {
    return false;
  }

  uint8_t pixels[64 * 64 * 4];
  for (size_t y = 0; y < 64; ++y)
  {
    for (size_t x = 0; x < 64; ++x)
    {
      size_t offset = ((y * 64) + x) * 4;
      bool dark = (((x / 8) + (y / 8)) & 1) != 0;
      pixels[offset + 0] = dark ? 50 : 240;
      pixels[offset + 1] = dark ? 90 : 170;
      pixels[offset + 2] = dark ? 220 : 255;
      pixels[offset + 3] = 255;
    }
  }

  C3DStageBufferInfo stage_info = {0};
  stage_info.size = sizeof(pixels);
  C3DStageBuffer* stage_buffer = c3dCreateStageBuffer(&stage_info);
  if (!stage_buffer)
  {
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(pixels), pixels) && c3dWriteTexture(g_render_state.texture, 0, sizeof(pixels), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  return success;
}

static bool createQuadBuffers(void)
{
  static const uint16_t quad_indices[] = {0, 1, 2, 3};
  C3DBufferInfo index_info = {0};
  index_info.size = sizeof(quad_indices);
  g_render_state.quad_indices = c3dCreateBuffer(&index_info);
  if (!g_render_state.quad_indices)
  {
    return false;
  }

  C3DStageBufferInfo stage_info = {0};
  stage_info.size = sizeof(quad_indices);
  C3DStageBuffer* stage_buffer = c3dCreateStageBuffer(&stage_info);
  if (!stage_buffer)
  {
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(quad_indices), quad_indices) && c3dWriteBuffer(g_render_state.quad_indices, 0, sizeof(quad_indices), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  if (!success)
  {
    return false;
  }

  C3DBufferInfo vertex_info = {0};
  vertex_info.size = sizeof(C3DVertex) * 4;
  g_render_state.quad_vertices = c3dCreateBuffer(&vertex_info);
  return g_render_state.quad_vertices != NULL;
}

static bool createLineBuffers(void)
{
  static const uint16_t line_indices[] = {0, 1, 2, 3};
  C3DBufferInfo index_info = {0};
  index_info.size = sizeof(line_indices);
  g_render_state.line_indices = c3dCreateBuffer(&index_info);
  if (!g_render_state.line_indices)
  {
    return false;
  }

  C3DStageBufferInfo stage_info = {0};
  stage_info.size = sizeof(line_indices);
  C3DStageBuffer* stage_buffer = c3dCreateStageBuffer(&stage_info);
  if (!stage_buffer)
  {
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(line_indices), line_indices) && c3dWriteBuffer(g_render_state.line_indices, 0, sizeof(line_indices), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  if (!success)
  {
    return false;
  }

  C3DBufferInfo vertex_info = {0};
  vertex_info.size = sizeof(C3DVertex) * 4;
  g_render_state.line_vertices = c3dCreateBuffer(&vertex_info);
  return g_render_state.line_vertices != NULL;
}

static bool initializeRenderState(void)
{
  TracyCZoneN(tracy_zone, "initializeRenderState", 1);
  if (g_render_state.initialized)
  {
    TracyCZoneEnd(tracy_zone);
    return true;
  }

  if (!createCheckerTexture() || !createQuadBuffers() || !createLineBuffers())
  {
    destroyRenderState();
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  g_render_state.command_buffer = c3dCreateCommandBuffer();
  if (!g_render_state.command_buffer)
  {
    destroyRenderState();
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  if (!ensureStageBufferSize(&g_render_state.quad_upload_stage, sizeof(C3DVertex) * 4) || !ensureStageBufferSize(&g_render_state.line_upload_stage, sizeof(C3DVertex) * 4))
  {
    destroyRenderState();
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  g_render_state.initialized = true;
  TracyCZoneEnd(tracy_zone);
  return true;
}

static float getSeconds(void)
{
  static LARGE_INTEGER frequency = {0};
  LARGE_INTEGER counter;
  if (frequency.QuadPart == 0)
  {
    QueryPerformanceFrequency(&frequency);
  }

  QueryPerformanceCounter(&counter);
  return (float)((double)counter.QuadPart / (double)frequency.QuadPart);
}

static void render(C3DTexture* texture)
{
  TracyCZoneN(tracy_zone, "render", 1);
  if (!initializeRenderState())
  {
    clearFallback(texture, 255, 0, 255);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  C3DTextureInfo target_info = {0};
  if (!c3dGetTextureInfo(texture, &target_info))
  {
    clearFallback(texture, 255, 0, 255);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  if (!ensureDepthTexture(target_info.width, target_info.height))
  {
    clearFallback(texture, 255, 0, 255);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  uint8_t clear_color[4] = {18, 22, 30, 255};
  if (!c3dClearTexture(texture, clear_color))
  {
    clearFallback(texture, 255, 0, 255);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  float t = getSeconds();
  float scale_x = 0.45f + (0.08f * sinf(t * 1.2f));
  float scale_y = 0.45f + (0.10f * cosf(t * 0.9f));
  float angle = t * 0.85f;
  float s = sinf(angle);
  float c = cosf(angle);

  C3DVertex quad_vertices[4] = {0};
  float corners[4][2] = {
      {-scale_x, -scale_y},
      { scale_x, -scale_y},
      { scale_x,  scale_y},
      {-scale_x,  scale_y},
  };
  float uvs[4][2] = {
      {0.0f, 1.0f},
      {1.0f, 1.0f},
      {1.0f, 0.0f},
      {0.0f, 0.0f},
  };
  float colors[4][4] = {
      { 1.0f, 0.45f, 0.35f, 0.90f},
      {0.25f, 0.95f, 0.55f, 0.90f},
      {0.30f, 0.65f, 1.00f, 0.90f},
      {1.00f, 0.90f, 0.30f, 0.90f},
  };

  for (size_t i = 0; i < 4; ++i)
  {
    float x = corners[i][0];
    float y = corners[i][1];
    quad_vertices[i].pos[0] = (x * c) - (y * s);
    quad_vertices[i].pos[1] = (x * s) + (y * c);
    quad_vertices[i].pos[2] = 0.0f;
    quad_vertices[i].pos[3] = 1.0f;
    quad_vertices[i].col[0] = colors[i][0];
    quad_vertices[i].col[1] = colors[i][1];
    quad_vertices[i].col[2] = colors[i][2];
    quad_vertices[i].col[3] = colors[i][3];
    quad_vertices[i].uv[0] = uvs[i][0];
    quad_vertices[i].uv[1] = uvs[i][1];
    quad_vertices[i].texid = 0;
  }

  void* quad_stage_data = NULL;
  quad_stage_data = c3dMapStageBuffer(g_render_state.quad_upload_stage, C3D_MEMORY_ACCESS_WRITE);
  if (!quad_stage_data)
  {
    clearFallback(texture, 255, 32, 32);
    TracyCZoneEnd(tracy_zone);
    return;
  }
  memcpy(quad_stage_data, quad_vertices, sizeof(quad_vertices));
  c3dUnmapStageBuffer(g_render_state.quad_upload_stage);

  if (!c3dWriteBuffer(g_render_state.quad_vertices, 0, sizeof(quad_vertices), g_render_state.quad_upload_stage, 0))
  {
    clearFallback(texture, 255, 32, 32);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  float pulse = 0.65f + (0.35f * sinf(t * 2.5f));
  C3DVertex line_vertices[4] = {0};
  line_vertices[0].pos[0] = -0.95f;
  line_vertices[0].pos[1] = 0.0f;
  line_vertices[1].pos[0] = 0.95f;
  line_vertices[1].pos[1] = 0.0f;
  line_vertices[2].pos[0] = 0.0f;
  line_vertices[2].pos[1] = -0.95f;
  line_vertices[3].pos[0] = 0.0f;
  line_vertices[3].pos[1] = 0.95f;
  for (size_t i = 0; i < 4; ++i)
  {
    line_vertices[i].pos[2] = 0.0f;
    line_vertices[i].pos[3] = 1.0f;
    line_vertices[i].col[0] = 0.15f + (0.85f * pulse);
    line_vertices[i].col[1] = 0.25f + (0.20f * pulse);
    line_vertices[i].col[2] = 1.00f;
    line_vertices[i].col[3] = 0.55f;
    line_vertices[i].uv[0] = 0.0f;
    line_vertices[i].uv[1] = 0.0f;
    line_vertices[i].texid = -1;
  }

  void* line_stage_data = NULL;
  line_stage_data = c3dMapStageBuffer(g_render_state.line_upload_stage, C3D_MEMORY_ACCESS_WRITE);
  if (!line_stage_data)
  {
    clearFallback(texture, 255, 32, 32);
    TracyCZoneEnd(tracy_zone);
    return;
  }
  memcpy(line_stage_data, line_vertices, sizeof(line_vertices));
  c3dUnmapStageBuffer(g_render_state.line_upload_stage);

  if (!c3dWriteBuffer(g_render_state.line_vertices, 0, sizeof(line_vertices), g_render_state.line_upload_stage, 0))
  {
    clearFallback(texture, 255, 32, 32);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  C3DTextureBinding binding = {0};
  binding.sampler = C3D_SAMPLER_LINEAR_WRAP;
  binding.texture = g_render_state.texture;

  C3DRenderPassInfo render_pass = {0};
  render_pass.target = texture;
  render_pass.depthTarget = g_render_state.depth_texture;
  render_pass.viewport.x = 0;
  render_pass.viewport.y = 0;
  render_pass.viewport.width = target_info.width;
  render_pass.viewport.height = target_info.height;
  render_pass.targetBlend = C3D_BLEND_MODE_NORMAL;
  render_pass.textureBindings = &binding;
  render_pass.textureBindCount = 1;

  if (!c3dBeginRenderPass(g_render_state.command_buffer, &render_pass))
  {
    clearFallback(texture, 255, 128, 0);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  C3DDrawInfo quad_draw = {0};
  quad_draw.topology = C3D_TOPOLOGY_QUAD;
  quad_draw.indexSize = C3D_INDEX_SIZE_16;
  quad_draw.indexBuffer = g_render_state.quad_indices;
  quad_draw.indexOffset = 0;
  quad_draw.indexBase = 0;
  quad_draw.vertexBuffer = g_render_state.quad_vertices;
  quad_draw.vertexOffset = 0;
  quad_draw.count = 4;

  if (!c3dDraw(g_render_state.command_buffer, &quad_draw))
  {
    c3dCancelCommandBuffer(g_render_state.command_buffer);
    clearFallback(texture, 255, 128, 0);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  C3DDrawInfo line_draw = {0};
  line_draw.topology = C3D_TOPOLOGY_LINE;
  line_draw.indexSize = C3D_INDEX_SIZE_16;
  line_draw.indexBuffer = g_render_state.line_indices;
  line_draw.indexOffset = 0;
  line_draw.indexBase = 0;
  line_draw.vertexBuffer = g_render_state.line_vertices;
  line_draw.vertexOffset = 0;
  line_draw.count = 4;

  if (!c3dDraw(g_render_state.command_buffer, &line_draw) || !c3dEndRenderPass(g_render_state.command_buffer))
  {
    c3dCancelCommandBuffer(g_render_state.command_buffer);
    clearFallback(texture, 255, 128, 0);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  if (!c3dSubmitCommandBuffer(g_render_state.command_buffer))
  {
    clearFallback(texture, 255, 255, 0);
  }

  TracyCZoneEnd(tracy_zone);
}

static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam)
{
  switch (message)
  {
    case WM_DESTROY:
    case WM_CLOSE:
    {
      PostQuitMessage(0);
      return 0;
    }
  }

  return DefWindowProcA(window, message, wparam, lparam);
}

static HWND openWindow(HINSTANCE instance, int show_command)
{
  WNDCLASSA window_class = {0};
  window_class.lpfnWndProc = windowProc;
  window_class.hInstance = instance;
  window_class.lpszClassName = "C3DWindowClass";
  window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  if (!RegisterClassA(&window_class))
  {
    return NULL;
  }

  HWND window = CreateWindowExA(
      0,
      window_class.lpszClassName,
      "C3D Test",
      WS_OVERLAPPEDWINDOW | WS_VISIBLE,
      CW_USEDEFAULT,
      CW_USEDEFAULT,
      1280,
      720,
      0,
      0,
      instance,
      0);

  if (!window)
    return NULL;

  ShowWindow(window, show_command);
  UpdateWindow(window);
  return window;
}

static bool pollMessages(void)
{
  MSG message;
  while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE))
  {
    if (message.message == WM_QUIT)
    {
      return false;
    }

    TranslateMessage(&message);
    DispatchMessageA(&message);
  }

  return true;
}

static void presentToWindow(HWND window, C3DTexture* texture, const C3DTextureInfo* info)
{
  TracyCZoneN(tracy_zone, "presentToWindow", 1);
  struct
  {
    BITMAPINFOHEADER header;
    DWORD masks[4];
  } bitmap_info = {0};
  bitmap_info.header.biSize = sizeof(bitmap_info.header);
  bitmap_info.header.biWidth = info->width;
  bitmap_info.header.biHeight = -info->height;
  bitmap_info.header.biPlanes = 1;
  bitmap_info.header.biBitCount = 32;
  bitmap_info.header.biCompression = BI_BITFIELDS;
  bitmap_info.masks[0] = 0x000000FFu;
  bitmap_info.masks[1] = 0x0000FF00u;
  bitmap_info.masks[2] = 0x00FF0000u;
  bitmap_info.masks[3] = 0xFF000000u;

  size_t size = info->width * info->height * 4;
  if (!ensureStageBufferSize(&g_render_state.readback_stage, size) || !c3dReadTexture(texture, 0, size, g_render_state.readback_stage, 0))
  {
    TracyCZoneEnd(tracy_zone);
    return;
  }

  void* buffer = NULL;
  buffer = c3dMapStageBuffer(g_render_state.readback_stage, C3D_MEMORY_ACCESS_READ);
  if (!buffer)
  {
    TracyCZoneEnd(tracy_zone);
    return;
  }

  HDC device_context = GetDC(window);
  StretchDIBits(
      device_context,
      0,
      0,
      info->width,
      info->height,
      0,
      0,
      info->width,
      info->height,
      buffer,
      (BITMAPINFO*)&bitmap_info,
      DIB_RGB_COLORS,
      SRCCOPY);
  ReleaseDC(window, device_context);
  c3dUnmapStageBuffer(g_render_state.readback_stage);
  TracyCZoneEnd(tracy_zone);
}

static void updateWindowTitle(HWND window, LARGE_INTEGER now, LARGE_INTEGER frequency)
{
  static LARGE_INTEGER last_title_update = {0};
  static uint32_t frame_count = 0;

  ++frame_count;
  if (!last_title_update.QuadPart)
  {
    last_title_update = now;
    return;
  }

  double elapsed_seconds = (double)(now.QuadPart - last_title_update.QuadPart) / (double)frequency.QuadPart;
  if (elapsed_seconds < 1.0)
  {
    return;
  }

  char title[64];
  double fps = (double)frame_count / elapsed_seconds;
  snprintf(title, sizeof(title), "C3D Test - %.1f FPS", fps);
  SetWindowTextA(window, title);

  frame_count = 0;
  last_title_update = now;
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous_instance, LPSTR command_line, int show_command)
{
  (void)previous_instance;
  (void)command_line;

  c3dSetErrorCallback(c3dErrorCallback);

  HWND window = openWindow(instance, show_command);
  if (!window)
  {
    return 1;
  }

  RECT client_rect;
  GetClientRect(window, &client_rect);
  int window_width = client_rect.right - client_rect.left;
  int window_height = client_rect.bottom - client_rect.top;

  C3DTextureInfo backbufferInfo = {0};
  backbufferInfo.width = window_width;
  backbufferInfo.height = window_height;
  backbufferInfo.format = C3D_TEXTURE_FORMAT_RGBA8;
  C3DTexture* backbuffer = c3dCreateTexture(&backbufferInfo);
  LARGE_INTEGER performance_frequency;
  QueryPerformanceFrequency(&performance_frequency);

  bool running = true;
  while (running)
  {
    TracyCFrameMark;
    if (!pollMessages())
    {
      break;
    }

    RECT client_rect;
    GetClientRect(window, &client_rect);
    int window_width = client_rect.right - client_rect.left;
    int window_height = client_rect.bottom - client_rect.top;
    if ((window_width != backbufferInfo.width) || (window_height != backbufferInfo.height))
    {
      backbufferInfo.width = window_width;
      backbufferInfo.height = window_height;
      c3dDeleteTexture(backbuffer);
      backbuffer = c3dCreateTexture(&backbufferInfo);
    }

    render(backbuffer);
    presentToWindow(window, backbuffer, &backbufferInfo);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    updateWindowTitle(window, now, performance_frequency);
  }

  destroyRenderState();
  c3dDeleteTexture(backbuffer);
  DestroyWindow(window);
  return 0;
}
