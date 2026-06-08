#define WIN32_LEAN_AND_MEAN
#include <C3D.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <windows.h>

static void render(C3DTexture* texture) {
  // TODO:
}

static LRESULT CALLBACK windowProc(HWND window, UINT message, WPARAM wparam, LPARAM lparam) {
  switch (message) {
    case WM_DESTROY:
    case WM_CLOSE:   {
      PostQuitMessage(0);
      return 0;
    }
  }

  return DefWindowProcA(window, message, wparam, lparam);
}

static HWND openWindow(HINSTANCE instance, int show_command) {
  WNDCLASSA window_class = {0};
  window_class.lpfnWndProc = windowProc;
  window_class.hInstance = instance;
  window_class.lpszClassName = "C3DWindowClass";
  window_class.style = CS_HREDRAW | CS_VREDRAW | CS_OWNDC;
  if (!RegisterClassA(&window_class)) {
    return NULL;
  }

  HWND window = CreateWindowExA(
      0,
      window_class.lpszClassName,
      "C3D",
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

static bool pollMessages(void) {
  MSG message;
  while (PeekMessageA(&message, 0, 0, 0, PM_REMOVE)) {
    if (message.message == WM_QUIT) {
      return false;
    }

    TranslateMessage(&message);
    DispatchMessageA(&message);
  }

  return true;
}

static void presentToWindow(HWND window, C3DTexture* texture, const C3DTextureInfo* info) {
  struct {
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
  void* buffer = malloc(size);
  c3dReadTexture(texture, 0, size, buffer);

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
  free(buffer);
}

int WINAPI WinMain(HINSTANCE instance, HINSTANCE previous_instance, LPSTR command_line, int show_command) {
  (void)previous_instance;
  (void)command_line;

  HWND window = openWindow(instance, show_command);
  if (!window) {
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

  bool running = true;
  while (running) {
    if (!pollMessages()) {
      break;
    }

    RECT client_rect;
    GetClientRect(window, &client_rect);
    int window_width = client_rect.right - client_rect.left;
    int window_height = client_rect.bottom - client_rect.top;
    if ((window_width != backbufferInfo.width) || (window_height != backbufferInfo.height)) {
      backbufferInfo.width = window_width;
      backbufferInfo.height = window_height;
      c3dDeleteTexture(backbuffer);
      backbuffer = c3dCreateTexture(&backbufferInfo);
    }

    render(backbuffer);
    presentToWindow(window, backbuffer, &backbufferInfo);
  }

  c3dDeleteTexture(backbuffer);
  DestroyWindow(window);
  return 0;
}
