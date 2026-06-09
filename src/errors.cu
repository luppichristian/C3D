#include <C3D.h>
#include <string.h>

typedef struct
{
  C3DErrorID id;
  char desc[C3D_ERROR_DESC_CAP];
  C3DErrorLoc loc;
  C3DErrorCallback* callback;
} C3DThreadContext;

static thread_local C3DThreadContext threadCtx = {};

C3D_API void _c3dThrowError(C3DErrorID id, const char* desc, C3DErrorLoc loc)
{
  threadCtx.id = id;
  threadCtx.loc = loc;
  strncpy(threadCtx.desc, desc, C3D_ERROR_DESC_CAP);
  if (threadCtx.callback)
    threadCtx.callback(threadCtx.id, threadCtx.desc, threadCtx.loc);
}

C3D_API C3DErrorID c3dGetErrorID(void)
{
  return threadCtx.id;
}

C3D_API const char* c3dGetErrorDesc(void)
{
  return threadCtx.desc;
}

C3D_API void c3dSetErrorCallback(C3DErrorCallback* callback)
{
  threadCtx.callback = callback;
}

C3D_API C3DErrorCallback* c3dGetErrorCallback(void)
{
  return threadCtx.callback;
}