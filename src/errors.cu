#include <C3D.h>

static thread_local C3DErrorID g_error_id = C3D_ERROR_NONE;
static thread_local const char* g_error_desc = "";
static thread_local C3DErrorLoc g_error_loc = {0};
static C3DErrorCallback* g_error_callback = nullptr;

C3D_API void _c3dThrowError(C3DErrorID id, const char* desc, C3DErrorLoc loc) {
  g_error_id = id;
  g_error_desc = desc ? desc : "";
  g_error_loc = loc;
  if (g_error_callback) {
    g_error_callback(g_error_id, g_error_desc, g_error_loc);
  }
}

C3D_API C3DErrorID c3dGetErrorID(void) {
  return g_error_id;
}

C3D_API const char* c3dGetErrorDesc(void) {
  return g_error_desc;
}

C3D_API void c3dSetErrorCallback(C3DErrorCallback* callback) {
  g_error_callback = callback;
}