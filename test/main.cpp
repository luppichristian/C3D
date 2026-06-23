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

typedef struct
{
  HDC window_dc;
  HDC bitmap_dc;
  HBITMAP dib_bitmap;
  HGDIOBJ old_bitmap;
  void* dib_pixels;
  size_t width;
  size_t height;
  struct
  {
    BITMAPINFOHEADER header;
    DWORD masks[4];
  } bitmap_info;
} PresentState;

typedef enum
{
  PRESENT_SLOT_FREE = 0,
  PRESENT_SLOT_READY,
  PRESENT_SLOT_PRESENTING,
} PresentSlotState;

typedef struct
{
  C3DStageBuffer* stage_buffer;
  C3DTextureInfo info;
  uint64_t sequence;
  PresentSlotState state;
} PresentSlot;

typedef struct
{
  HWND window;
  HANDLE thread;
  HANDLE wake_event;
  HANDLE shutdown_event;
  CRITICAL_SECTION lock;
  PresentSlot slots[3];
  uint64_t next_sequence;
} PresenterState;

static RenderState g_render_state = {0};
static PresentState g_present_state = {0};
static PresenterState g_presenter_state = {0};
static bool g_error_dialog_shown = false;

static void c3dErrorCallback(C3DErrorID id, const char* desc, C3DErrorLoc loc)
{
  TracyCZoneN(tracy_zone, "c3dErrorCallback", 1);
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
  TracyCZoneEnd(tracy_zone);
}

static void clearFallback(C3DTexture* texture, uint8_t r, uint8_t g, uint8_t b)
{
  TracyCZoneN(tracy_zone, "clearFallback", 1);
  uint8_t clear_color[4] = {r, g, b, 255};
  c3dClearTexture(texture, clear_color);
  TracyCZoneEnd(tracy_zone);
}

static bool ensureStageBufferSize(C3DStageBuffer** stage_buffer, size_t size)
{
  TracyCZoneN(tracy_zone, "ensureStageBufferSize", 1);
  if (!*stage_buffer)
  {
    C3DStageBufferInfo info = {0};
    info.size = size;
    *stage_buffer = c3dCreateStageBuffer(&info);
    bool created = *stage_buffer != NULL;
    TracyCZoneEnd(tracy_zone);
    return created;
  }

  C3DStageBufferInfo info = {0};
  if (!c3dGetStageBufferInfo(*stage_buffer, &info))
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  bool resized = info.size == size || c3dResizeStageBuffer(*stage_buffer, size);
  TracyCZoneEnd(tracy_zone);
  return resized;
}

static bool ensureDepthTexture(size_t width, size_t height)
{
  TracyCZoneN(tracy_zone, "ensureDepthTexture", 1);
  C3DTextureInfo info = {0};
  if (g_render_state.depth_texture && c3dGetTextureInfo(g_render_state.depth_texture, &info) && info.width == width && info.height == height && info.format == C3D_TEXTURE_FORMAT_DEPTH64)
  {
    TracyCZoneEnd(tracy_zone);
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
  bool created = g_render_state.depth_texture != NULL;
  TracyCZoneEnd(tracy_zone);
  return created;
}

static void destroyRenderState(void)
{
  TracyCZoneN(tracy_zone, "destroyRenderState", 1);
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
  TracyCZoneEnd(tracy_zone);
}

static void destroyPresentState(HWND window)
{
  TracyCZoneN(tracy_zone, "destroyPresentState", 1);
  if (g_present_state.bitmap_dc && g_present_state.old_bitmap)
  {
    SelectObject(g_present_state.bitmap_dc, g_present_state.old_bitmap);
    g_present_state.old_bitmap = NULL;
  }

  if (g_present_state.dib_bitmap)
  {
    DeleteObject(g_present_state.dib_bitmap);
    g_present_state.dib_bitmap = NULL;
  }

  if (g_present_state.bitmap_dc)
  {
    DeleteDC(g_present_state.bitmap_dc);
    g_present_state.bitmap_dc = NULL;
  }

  if (g_present_state.window_dc)
  {
    ReleaseDC(window, g_present_state.window_dc);
    g_present_state.window_dc = NULL;
  }

  g_present_state.dib_pixels = NULL;
  g_present_state.width = 0;
  g_present_state.height = 0;
  ZeroMemory(&g_present_state.bitmap_info, sizeof(g_present_state.bitmap_info));
  TracyCZoneEnd(tracy_zone);
}

static bool ensurePresentState(HWND window, const C3DTextureInfo* info)
{
  TracyCZoneN(tracy_zone, "ensurePresentState", 1);
  if (!g_present_state.window_dc)
  {
    g_present_state.window_dc = GetDC(window);
    if (!g_present_state.window_dc)
    {
      TracyCZoneEnd(tracy_zone);
      return false;
    }
  }

  if (!g_present_state.bitmap_dc)
  {
    g_present_state.bitmap_dc = CreateCompatibleDC(g_present_state.window_dc);
    if (!g_present_state.bitmap_dc)
    {
      TracyCZoneEnd(tracy_zone);
      return false;
    }
  }

  if (g_present_state.width != info->width || g_present_state.height != info->height)
  {
    g_present_state.width = info->width;
    g_present_state.height = info->height;
    ZeroMemory(&g_present_state.bitmap_info, sizeof(g_present_state.bitmap_info));
    g_present_state.bitmap_info.header.biSize = sizeof(g_present_state.bitmap_info.header);
    g_present_state.bitmap_info.header.biWidth = (LONG)info->width;
    g_present_state.bitmap_info.header.biHeight = -(LONG)info->height;
    g_present_state.bitmap_info.header.biPlanes = 1;
    g_present_state.bitmap_info.header.biBitCount = 32;
    g_present_state.bitmap_info.header.biCompression = BI_BITFIELDS;
    g_present_state.bitmap_info.masks[0] = 0x000000FFu;
    g_present_state.bitmap_info.masks[1] = 0x0000FF00u;
    g_present_state.bitmap_info.masks[2] = 0x00FF0000u;
    g_present_state.bitmap_info.masks[3] = 0xFF000000u;

    if (g_present_state.bitmap_dc && g_present_state.old_bitmap)
    {
      SelectObject(g_present_state.bitmap_dc, g_present_state.old_bitmap);
      g_present_state.old_bitmap = NULL;
    }

    if (g_present_state.dib_bitmap)
    {
      DeleteObject(g_present_state.dib_bitmap);
      g_present_state.dib_bitmap = NULL;
      g_present_state.dib_pixels = NULL;
    }

    g_present_state.dib_bitmap = CreateDIBSection(
        g_present_state.bitmap_dc,
        (BITMAPINFO*)&g_present_state.bitmap_info,
        DIB_RGB_COLORS,
        &g_present_state.dib_pixels,
        NULL,
        0);
    if (!g_present_state.dib_bitmap || !g_present_state.dib_pixels)
    {
      if (g_present_state.dib_bitmap)
      {
        DeleteObject(g_present_state.dib_bitmap);
        g_present_state.dib_bitmap = NULL;
      }
      g_present_state.dib_pixels = NULL;
      TracyCZoneEnd(tracy_zone);
      return false;
    }

    g_present_state.old_bitmap = SelectObject(g_present_state.bitmap_dc, g_present_state.dib_bitmap);
    if (!g_present_state.old_bitmap)
    {
      DeleteObject(g_present_state.dib_bitmap);
      g_present_state.dib_bitmap = NULL;
      g_present_state.dib_pixels = NULL;
      TracyCZoneEnd(tracy_zone);
      return false;
    }
  }

  TracyCZoneEnd(tracy_zone);
  return true;
}

static bool presenterHasActiveSlot(void)
{
  bool active = false;
  EnterCriticalSection(&g_presenter_state.lock);
  for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
  {
    PresentSlotState state = g_presenter_state.slots[i].state;
    if (state == PRESENT_SLOT_READY || state == PRESENT_SLOT_PRESENTING)
    {
      active = true;
      break;
    }
  }
  LeaveCriticalSection(&g_presenter_state.lock);
  return active;
}

static bool ensurePresenterSlots(size_t width, size_t height)
{
  TracyCZoneN(tracy_zone, "ensurePresenterSlots", 1);
  while (presenterHasActiveSlot())
  {
    Sleep(0);
  }

  EnterCriticalSection(&g_presenter_state.lock);
  for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
  {
    g_presenter_state.slots[i].state = PRESENT_SLOT_FREE;
    g_presenter_state.slots[i].sequence = 0;
    g_presenter_state.slots[i].info.width = width;
    g_presenter_state.slots[i].info.height = height;
    g_presenter_state.slots[i].info.depth = 1;
    g_presenter_state.slots[i].info.format = C3D_TEXTURE_FORMAT_RGBA8;
  }
  LeaveCriticalSection(&g_presenter_state.lock);

  for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
  {
    if (!ensureStageBufferSize(&g_presenter_state.slots[i].stage_buffer, width * height * 4))
    {
      TracyCZoneEnd(tracy_zone);
      return false;
    }
  }

  TracyCZoneEnd(tracy_zone);
  return true;
}

static DWORD WINAPI presenterThreadProc(LPVOID parameter)
{
  TracyCZoneN(tracy_zone, "presenterThreadProc", 1);
  HWND window = (HWND)parameter;
  HANDLE wait_handles[2] = {g_presenter_state.shutdown_event, g_presenter_state.wake_event};

  for (;;)
  {
    DWORD wait_result = WaitForMultipleObjects(2, wait_handles, FALSE, INFINITE);
    if (wait_result == WAIT_OBJECT_0)
    {
      break;
    }

    for (;;)
    {
      PresentSlot* latest_slot = NULL;
      EnterCriticalSection(&g_presenter_state.lock);
      for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
      {
        PresentSlot* slot = &g_presenter_state.slots[i];
        if (slot->state != PRESENT_SLOT_READY)
        {
          continue;
        }

        if (!latest_slot || slot->sequence > latest_slot->sequence)
        {
          latest_slot = slot;
        }
      }

      if (latest_slot)
      {
        for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
        {
          PresentSlot* slot = &g_presenter_state.slots[i];
          if (slot != latest_slot && slot->state == PRESENT_SLOT_READY)
          {
            slot->state = PRESENT_SLOT_FREE;
          }
        }

        latest_slot->state = PRESENT_SLOT_PRESENTING;
      }
      LeaveCriticalSection(&g_presenter_state.lock);

      if (!latest_slot)
      {
        break;
      }

      if (ensurePresentState(window, &latest_slot->info))
      {
        size_t size = latest_slot->info.width * latest_slot->info.height * 4;
        void* buffer = c3dMapStageBuffer(latest_slot->stage_buffer, C3D_MEMORY_ACCESS_READ);
        if (buffer)
        {
          TracyCZoneN(copy_zone, "presenterThread.copyToDibSection", 1);
          memcpy(g_present_state.dib_pixels, buffer, size);
          TracyCZoneEnd(copy_zone);

          TracyCZoneN(blit_zone, "presenterThread.BitBlt", 1);
          BitBlt(
              g_present_state.window_dc,
              0,
              0,
              (int)latest_slot->info.width,
              (int)latest_slot->info.height,
              g_present_state.bitmap_dc,
              0,
              0,
              SRCCOPY);
          TracyCZoneEnd(blit_zone);

          c3dUnmapStageBuffer(latest_slot->stage_buffer);
        }
      }

      EnterCriticalSection(&g_presenter_state.lock);
      latest_slot->state = PRESENT_SLOT_FREE;
      LeaveCriticalSection(&g_presenter_state.lock);
    }
  }

  TracyCZoneEnd(tracy_zone);
  return 0;
}

static bool initializePresenter(HWND window, size_t width, size_t height)
{
  TracyCZoneN(tracy_zone, "initializePresenter", 1);
  ZeroMemory(&g_presenter_state, sizeof(g_presenter_state));
  g_presenter_state.window = window;
  InitializeCriticalSection(&g_presenter_state.lock);
  g_presenter_state.wake_event = CreateEventA(NULL, FALSE, FALSE, NULL);
  g_presenter_state.shutdown_event = CreateEventA(NULL, TRUE, FALSE, NULL);
  if (!g_presenter_state.wake_event || !g_presenter_state.shutdown_event)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  if (!ensurePresenterSlots(width, height))
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  g_presenter_state.thread = CreateThread(NULL, 0, presenterThreadProc, window, 0, NULL);
  bool created = g_presenter_state.thread != NULL;
  TracyCZoneEnd(tracy_zone);
  return created;
}

static void shutdownPresenter(HWND window)
{
  TracyCZoneN(tracy_zone, "shutdownPresenter", 1);
  if (g_presenter_state.shutdown_event)
  {
    SetEvent(g_presenter_state.shutdown_event);
  }

  if (g_presenter_state.thread)
  {
    WaitForSingleObject(g_presenter_state.thread, INFINITE);
    CloseHandle(g_presenter_state.thread);
    g_presenter_state.thread = NULL;
  }

  for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
  {
    if (g_presenter_state.slots[i].stage_buffer)
    {
      c3dDeleteStageBuffer(g_presenter_state.slots[i].stage_buffer);
      g_presenter_state.slots[i].stage_buffer = NULL;
    }
  }

  if (g_presenter_state.wake_event)
  {
    CloseHandle(g_presenter_state.wake_event);
    g_presenter_state.wake_event = NULL;
  }

  if (g_presenter_state.shutdown_event)
  {
    CloseHandle(g_presenter_state.shutdown_event);
    g_presenter_state.shutdown_event = NULL;
  }

  DeleteCriticalSection(&g_presenter_state.lock);
  destroyPresentState(window);
  ZeroMemory(&g_presenter_state, sizeof(g_presenter_state));
  TracyCZoneEnd(tracy_zone);
}

static bool createCheckerTexture(void)
{
  TracyCZoneN(tracy_zone, "createCheckerTexture", 1);
  C3DTextureInfo texture_info = {0};
  texture_info.width = 64;
  texture_info.height = 64;
  texture_info.depth = 1;
  texture_info.format = C3D_TEXTURE_FORMAT_RGBA8;
  g_render_state.texture = c3dCreateTexture(&texture_info);
  if (!g_render_state.texture)
  {
    TracyCZoneEnd(tracy_zone);
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
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(pixels), pixels) && c3dWriteTexture(g_render_state.texture, 0, sizeof(pixels), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  TracyCZoneEnd(tracy_zone);
  return success;
}

static bool createQuadBuffers(void)
{
  TracyCZoneN(tracy_zone, "createQuadBuffers", 1);
  static const uint16_t quad_indices[] = {0, 1, 2, 3};
  C3DBufferInfo index_info = {0};
  index_info.size = sizeof(quad_indices);
  g_render_state.quad_indices = c3dCreateBuffer(&index_info);
  if (!g_render_state.quad_indices)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  C3DStageBufferInfo stage_info = {0};
  stage_info.size = sizeof(quad_indices);
  C3DStageBuffer* stage_buffer = c3dCreateStageBuffer(&stage_info);
  if (!stage_buffer)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(quad_indices), quad_indices) && c3dWriteBuffer(g_render_state.quad_indices, 0, sizeof(quad_indices), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  if (!success)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  C3DBufferInfo vertex_info = {0};
  vertex_info.size = sizeof(C3DVertex) * 4;
  g_render_state.quad_vertices = c3dCreateBuffer(&vertex_info);
  bool created = g_render_state.quad_vertices != NULL;
  TracyCZoneEnd(tracy_zone);
  return created;
}

static bool createLineBuffers(void)
{
  TracyCZoneN(tracy_zone, "createLineBuffers", 1);
  static const uint16_t line_indices[] = {0, 1, 2, 3};
  C3DBufferInfo index_info = {0};
  index_info.size = sizeof(line_indices);
  g_render_state.line_indices = c3dCreateBuffer(&index_info);
  if (!g_render_state.line_indices)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  C3DStageBufferInfo stage_info = {0};
  stage_info.size = sizeof(line_indices);
  C3DStageBuffer* stage_buffer = c3dCreateStageBuffer(&stage_info);
  if (!stage_buffer)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  bool success = c3dWriteStageBuffer(stage_buffer, 0, sizeof(line_indices), line_indices) && c3dWriteBuffer(g_render_state.line_indices, 0, sizeof(line_indices), stage_buffer, 0);
  c3dDeleteStageBuffer(stage_buffer);
  if (!success)
  {
    TracyCZoneEnd(tracy_zone);
    return false;
  }

  C3DBufferInfo vertex_info = {0};
  vertex_info.size = sizeof(C3DVertex) * 4;
  g_render_state.line_vertices = c3dCreateBuffer(&vertex_info);
  bool created = g_render_state.line_vertices != NULL;
  TracyCZoneEnd(tracy_zone);
  return created;
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
  {
    TracyCZoneN(init_zone, "render.initializeRenderState", 1);
    if (!initializeRenderState())
    {
      TracyCZoneN(fallback_zone, "render.fallback.init", 1);
      clearFallback(texture, 255, 0, 255);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(init_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(init_zone);
  }

  C3DTextureInfo target_info = {0};
  {
    TracyCZoneN(info_zone, "render.getTargetInfo", 1);
    if (!c3dGetTextureInfo(texture, &target_info))
    {
      TracyCZoneN(fallback_zone, "render.fallback.targetInfo", 1);
      clearFallback(texture, 255, 0, 255);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(info_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(info_zone);
  }
  TracyCPlotI("C3D Render Target Width", (int64_t)target_info.width);
  TracyCPlotI("C3D Render Target Height", (int64_t)target_info.height);

  {
    TracyCZoneN(depth_zone, "render.ensureDepthTexture", 1);
    if (!ensureDepthTexture(target_info.width, target_info.height))
    {
      TracyCZoneN(fallback_zone, "render.fallback.depth", 1);
      clearFallback(texture, 255, 0, 255);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(depth_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(depth_zone);
  }

  uint8_t clear_color[4] = {18, 22, 30, 255};
  {
    TracyCZoneN(clear_zone, "render.clearTarget", 1);
    if (!c3dClearTexture(texture, clear_color))
    {
      TracyCZoneN(fallback_zone, "render.fallback.clear", 1);
      clearFallback(texture, 255, 0, 255);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(clear_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(clear_zone);
  }

  float t = getSeconds();
  float scale_x = 0.0f;
  float scale_y = 0.0f;
  float angle = 0.0f;
  float s = 0.0f;
  float c = 0.0f;
  {
    TracyCZoneN(anim_zone, "render.animate", 1);
    scale_x = 0.45f + (0.08f * sinf(t * 1.2f));
    scale_y = 0.45f + (0.10f * cosf(t * 0.9f));
    angle = t * 0.85f;
    s = sinf(angle);
    c = cosf(angle);
    TracyCZoneEnd(anim_zone);
  }

  C3DVertex quad_vertices[4] = {0};
  {
    TracyCZoneN(quad_cpu_zone, "render.buildQuadVertices", 1);
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
    TracyCZoneEnd(quad_cpu_zone);
  }

  void* quad_stage_data = NULL;
  {
    TracyCZoneN(quad_map_zone, "render.mapQuadStage", 1);
    quad_stage_data = c3dMapStageBuffer(g_render_state.quad_upload_stage, C3D_MEMORY_ACCESS_WRITE);
    if (!quad_stage_data)
    {
      TracyCZoneN(fallback_zone, "render.fallback.mapQuadStage", 1);
      clearFallback(texture, 255, 32, 32);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(quad_map_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(quad_map_zone);
  }
  {
    TracyCZoneN(quad_copy_zone, "render.copyQuadStage", 1);
    memcpy(quad_stage_data, quad_vertices, sizeof(quad_vertices));
    TracyCZoneEnd(quad_copy_zone);
  }
  {
    TracyCZoneN(quad_unmap_zone, "render.unmapQuadStage", 1);
    c3dUnmapStageBuffer(g_render_state.quad_upload_stage);
    TracyCZoneEnd(quad_unmap_zone);
  }

  {
    TracyCZoneN(quad_upload_zone, "render.uploadQuadVertices", 1);
    if (!c3dWriteBuffer(g_render_state.quad_vertices, 0, sizeof(quad_vertices), g_render_state.quad_upload_stage, 0))
    {
      TracyCZoneN(fallback_zone, "render.fallback.uploadQuad", 1);
      clearFallback(texture, 255, 32, 32);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(quad_upload_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(quad_upload_zone);
  }

  float pulse = 0.65f + (0.35f * sinf(t * 2.5f));
  C3DVertex line_vertices[4] = {0};
  {
    TracyCZoneN(line_cpu_zone, "render.buildLineVertices", 1);
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
    TracyCZoneEnd(line_cpu_zone);
  }

  void* line_stage_data = NULL;
  {
    TracyCZoneN(line_map_zone, "render.mapLineStage", 1);
    line_stage_data = c3dMapStageBuffer(g_render_state.line_upload_stage, C3D_MEMORY_ACCESS_WRITE);
    if (!line_stage_data)
    {
      TracyCZoneN(fallback_zone, "render.fallback.mapLineStage", 1);
      clearFallback(texture, 255, 32, 32);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(line_map_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(line_map_zone);
  }
  {
    TracyCZoneN(line_copy_zone, "render.copyLineStage", 1);
    memcpy(line_stage_data, line_vertices, sizeof(line_vertices));
    TracyCZoneEnd(line_copy_zone);
  }
  {
    TracyCZoneN(line_unmap_zone, "render.unmapLineStage", 1);
    c3dUnmapStageBuffer(g_render_state.line_upload_stage);
    TracyCZoneEnd(line_unmap_zone);
  }

  {
    TracyCZoneN(line_upload_zone, "render.uploadLineVertices", 1);
    if (!c3dWriteBuffer(g_render_state.line_vertices, 0, sizeof(line_vertices), g_render_state.line_upload_stage, 0))
    {
      TracyCZoneN(fallback_zone, "render.fallback.uploadLine", 1);
      clearFallback(texture, 255, 32, 32);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(line_upload_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(line_upload_zone);
  }

  C3DTextureBinding binding = {};
  {
    TracyCZoneN(binding_zone, "render.buildTextureBinding", 1);
    binding.sampler = C3D_SAMPLER_LINEAR_WRAP;
    binding.texture = g_render_state.texture;
    TracyCZoneEnd(binding_zone);
  }

  C3DRenderPassInfo render_pass = {0};
  {
    TracyCZoneN(pass_info_zone, "render.buildRenderPassInfo", 1);
    render_pass.target = texture;
    render_pass.depthTarget = g_render_state.depth_texture;
    render_pass.viewport.x = 0;
    render_pass.viewport.y = 0;
    render_pass.viewport.width = target_info.width;
    render_pass.viewport.height = target_info.height;
    render_pass.targetBlend = C3D_BLEND_MODE_NORMAL;
    render_pass.textureBindings = &binding;
    render_pass.textureBindCount = 1;
    TracyCZoneEnd(pass_info_zone);
  }

  {
    TracyCZoneN(begin_pass_zone, "render.beginRenderPass", 1);
    if (!c3dBeginRenderPass(g_render_state.command_buffer, &render_pass))
    {
      TracyCZoneN(fallback_zone, "render.fallback.beginRenderPass", 1);
      clearFallback(texture, 255, 128, 0);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(begin_pass_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(begin_pass_zone);
  }

  C3DDrawInfo quad_draw = {};
  {
    TracyCZoneN(quad_draw_info_zone, "render.buildQuadDrawInfo", 1);
    quad_draw.topology = C3D_TOPOLOGY_QUAD;
    quad_draw.indexSize = C3D_INDEX_SIZE_16;
    quad_draw.indexBuffer = g_render_state.quad_indices;
    quad_draw.indexOffset = 0;
    quad_draw.indexBase = 0;
    quad_draw.vertexBuffer = g_render_state.quad_vertices;
    quad_draw.vertexOffset = 0;
    quad_draw.count = 4;
    TracyCZoneEnd(quad_draw_info_zone);
  }

  {
    TracyCZoneN(quad_draw_zone, "render.recordQuadDraw", 1);
    if (!c3dDraw(g_render_state.command_buffer, &quad_draw))
    {
      TracyCZoneN(cancel_zone, "render.cancelCommandBuffer.quad", 1);
      c3dCancelCommandBuffer(g_render_state.command_buffer);
      TracyCZoneEnd(cancel_zone);
      TracyCZoneN(fallback_zone, "render.fallback.quadDraw", 1);
      clearFallback(texture, 255, 128, 0);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(quad_draw_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(quad_draw_zone);
  }

  C3DDrawInfo line_draw = {};
  {
    TracyCZoneN(line_draw_info_zone, "render.buildLineDrawInfo", 1);
    line_draw.topology = C3D_TOPOLOGY_LINE;
    line_draw.indexSize = C3D_INDEX_SIZE_16;
    line_draw.indexBuffer = g_render_state.line_indices;
    line_draw.indexOffset = 0;
    line_draw.indexBase = 0;
    line_draw.vertexBuffer = g_render_state.line_vertices;
    line_draw.vertexOffset = 0;
    line_draw.count = 4;
    TracyCZoneEnd(line_draw_info_zone);
  }

  {
    TracyCZoneN(line_draw_zone, "render.recordLineDraw", 1);
    bool line_draw_ok = c3dDraw(g_render_state.command_buffer, &line_draw);
    TracyCZoneEnd(line_draw_zone);
    TracyCZoneN(end_pass_zone, "render.endRenderPass", 1);
    bool end_pass_ok = line_draw_ok && c3dEndRenderPass(g_render_state.command_buffer);
    TracyCZoneEnd(end_pass_zone);
    if (!end_pass_ok)
    {
      TracyCZoneN(cancel_zone, "render.cancelCommandBuffer.lineOrEndPass", 1);
      c3dCancelCommandBuffer(g_render_state.command_buffer);
      TracyCZoneEnd(cancel_zone);
      TracyCZoneN(fallback_zone, "render.fallback.lineOrEndPass", 1);
      clearFallback(texture, 255, 128, 0);
      TracyCZoneEnd(fallback_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
  }

  {
    TracyCZoneN(submit_zone, "render.submitCommandBuffer", 1);
    if (!c3dSubmitCommandBuffer(g_render_state.command_buffer))
    {
      TracyCZoneN(fallback_zone, "render.fallback.submit", 1);
      clearFallback(texture, 255, 255, 0);
      TracyCZoneEnd(fallback_zone);
    }
    TracyCZoneEnd(submit_zone);
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
  TracyCZoneN(tracy_zone, "openWindow", 1);
  WNDCLASSA window_class = {0};
  window_class.lpfnWndProc = windowProc;
  window_class.hInstance = instance;
  window_class.lpszClassName = "C3DWindowClass";
  window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  if (!RegisterClassA(&window_class))
  {
    TracyCZoneEnd(tracy_zone);
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
  {
    TracyCZoneEnd(tracy_zone);
    return NULL;
  }

  ShowWindow(window, show_command);
  UpdateWindow(window);
  TracyCZoneEnd(tracy_zone);
  return window;
}

static bool pollMessages(void)
{
  TracyCZoneN(tracy_zone, "pollMessages", 1);
  MSG message;
  int32_t message_count = 0;
  while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE))
  {
    ++message_count;
    if (message.message == WM_QUIT)
    {
      TracyCPlotI("C3D Message Count", message_count);
      TracyCZoneEnd(tracy_zone);
      return false;
    }

    TranslateMessage(&message);
    DispatchMessageA(&message);
  }

  TracyCPlotI("C3D Message Count", message_count);
  TracyCZoneEnd(tracy_zone);
  return true;
}

static void presentToWindow(C3DTexture* texture, const C3DTextureInfo* info)
{
  TracyCZoneN(tracy_zone, "presentToWindow", 1);
  size_t size = info->width * info->height * 4;
  TracyCPlotI("C3D Present Width", (int64_t)info->width);
  TracyCPlotI("C3D Present Height", (int64_t)info->height);
  TracyCPlotI("C3D Present Readback Bytes", (int64_t)size);

  PresentSlot* slot = NULL;
  {
    TracyCZoneN(acquire_zone, "presentToWindow.acquireSlot", 1);
    EnterCriticalSection(&g_presenter_state.lock);
    for (size_t i = 0; i < _countof(g_presenter_state.slots); ++i)
    {
      if (g_presenter_state.slots[i].state == PRESENT_SLOT_FREE)
      {
        slot = &g_presenter_state.slots[i];
        slot->state = PRESENT_SLOT_PRESENTING;
        slot->info = *info;
        slot->sequence = ++g_presenter_state.next_sequence;
        break;
      }
    }
    LeaveCriticalSection(&g_presenter_state.lock);
    TracyCZoneEnd(acquire_zone);
  }

  if (!slot)
  {
    TracyCZoneN(drop_zone, "presentToWindow.dropFrame", 1);
    TracyCZoneEnd(drop_zone);
    TracyCZoneEnd(tracy_zone);
    return;
  }

  {
    TracyCZoneN(readback_zone, "presentToWindow.readbackTexture", 1);
    bool readback_ok = c3dReadTexture(texture, 0, size, slot->stage_buffer, 0);
    if (!readback_ok)
    {
      EnterCriticalSection(&g_presenter_state.lock);
      slot->state = PRESENT_SLOT_FREE;
      LeaveCriticalSection(&g_presenter_state.lock);
      TracyCZoneEnd(readback_zone);
      TracyCZoneEnd(tracy_zone);
      return;
    }
    TracyCZoneEnd(readback_zone);
  }
  EnterCriticalSection(&g_presenter_state.lock);
  slot->state = PRESENT_SLOT_READY;
  LeaveCriticalSection(&g_presenter_state.lock);
  SetEvent(g_presenter_state.wake_event);
  TracyCZoneEnd(tracy_zone);
}

static void updateWindowTitle(HWND window, LARGE_INTEGER now, LARGE_INTEGER frequency)
{
  TracyCZoneN(tracy_zone, "updateWindowTitle", 1);
  static LARGE_INTEGER last_title_update = {0};
  static uint32_t frame_count = 0;

  ++frame_count;
  if (!last_title_update.QuadPart)
  {
    last_title_update = now;
    TracyCZoneEnd(tracy_zone);
    return;
  }

  double elapsed_seconds = (double)(now.QuadPart - last_title_update.QuadPart) / (double)frequency.QuadPart;
  if (elapsed_seconds < 1.0)
  {
    TracyCZoneEnd(tracy_zone);
    return;
  }

  char title[64];
  double fps = (double)frame_count / elapsed_seconds;
  TracyCPlotI("C3D FPS", (int64_t)(fps * 100.0));
  snprintf(title, sizeof(title), "C3D Test - %.1f FPS", fps);
  SetWindowTextA(window, title);

  frame_count = 0;
  last_title_update = now;
  TracyCZoneEnd(tracy_zone);
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous_instance, LPSTR command_line, int show_command)
{
  TracyCZoneN(tracy_zone, "WinMain", 1);
  (void)previous_instance;
  (void)command_line;

  c3dSetErrorCallback(c3dErrorCallback);

  HWND window = openWindow(instance, show_command);
  if (!window)
  {
    TracyCZoneEnd(tracy_zone);
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
  if (!backbuffer)
  {
    DestroyWindow(window);
    TracyCZoneEnd(tracy_zone);
    return 1;
  }

  if (!initializePresenter(window, backbufferInfo.width, backbufferInfo.height))
  {
    c3dDeleteTexture(backbuffer);
    shutdownPresenter(window);
    DestroyWindow(window);
    TracyCZoneEnd(tracy_zone);
    return 1;
  }
  LARGE_INTEGER performance_frequency;
  QueryPerformanceFrequency(&performance_frequency);
  LARGE_INTEGER last_frame_counter;
  QueryPerformanceCounter(&last_frame_counter);

  bool running = true;
  while (running)
  {
    TracyCZoneN(frame_zone, "frame", 1);
    TracyCFrameMark;
    if (!pollMessages())
    {
      TracyCZoneEnd(frame_zone);
      break;
    }

    RECT client_rect;
    GetClientRect(window, &client_rect);
    int window_width = client_rect.right - client_rect.left;
    int window_height = client_rect.bottom - client_rect.top;
    if ((window_width != backbufferInfo.width) || (window_height != backbufferInfo.height))
    {
      TracyCZoneN(resize_zone, "resizeBackbuffer", 1);
      backbufferInfo.width = window_width;
      backbufferInfo.height = window_height;
      TracyCPlotI("C3D Backbuffer Width", backbufferInfo.width);
      TracyCPlotI("C3D Backbuffer Height", backbufferInfo.height);
      c3dDeleteTexture(backbuffer);
      backbuffer = c3dCreateTexture(&backbufferInfo);
      bool presenter_ok = backbuffer && ensurePresenterSlots(backbufferInfo.width, backbufferInfo.height);
      TracyCZoneEnd(resize_zone);
      if (!presenter_ok)
      {
        TracyCZoneEnd(frame_zone);
        break;
      }
    }

    render(backbuffer);
    presentToWindow(backbuffer, &backbufferInfo);

    LARGE_INTEGER now;
    QueryPerformanceCounter(&now);
    double frame_seconds = (double)(now.QuadPart - last_frame_counter.QuadPart) / (double)performance_frequency.QuadPart;
    TracyCPlotI("C3D Frame Time (us)", (int64_t)(frame_seconds * 1000000.0));
    last_frame_counter = now;
    updateWindowTitle(window, now, performance_frequency);
    TracyCZoneEnd(frame_zone);
  }

  destroyRenderState();
  shutdownPresenter(window);
  c3dDeleteTexture(backbuffer);
  DestroyWindow(window);
  TracyCZoneEnd(tracy_zone);
  return 0;
}
