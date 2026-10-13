/*
 * libQnnHtp.so shim for running Genie (QAIRT 2.46) on the SA8650P under QNX 8.0.
 *
 * Genie always passes the HTP context option IO_MEM_ESTIMATION when it creates a context from a binary. The option
 * is meant for targets with several DSP protection domains; the QNX HTP backend of this board rejects it
 * ("HTP Config option: 6 not supported", err 1000) and every Genie model fails to load, although the same context
 * binaries run with qnn-net-run. This library is loaded under the name libQnnHtp.so (put its directory first in
 * LD_LIBRARY_PATH), forwards the real backend (QNN_SHIM_REAL, default /data/adas_vla_genie/shim/libQnnHtpReal.so,
 * a copy of the QAIRT libQnnHtp.so) and drops only that option from the context configs. Build: genie_htp_shim.sh.
 */
#include <dlfcn.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "HTP/QnnHtpContext.h"
#include "QnnInterface.h"

#define MAX_PROVIDERS 8
#define MAX_CONFIGS 64

static QnnInterface_t shim_iface[MAX_PROVIDERS];
static const QnnInterface_t* shim_list[MAX_PROVIDERS];
static QNN_INTERFACE_VER_TYPE real;  // function table of the real backend (provider 0)

static const QnnContext_Config_t** drop_io_mem_estimation(const QnnContext_Config_t** in,
                                                          const QnnContext_Config_t** out) {
  if (in == NULL) return in;
  size_t n = 0;
  for (size_t i = 0; in[i] != NULL && n + 1 < MAX_CONFIGS; ++i) {
    const QnnContext_Config_t* c = in[i];
    if (c->option == QNN_CONTEXT_CONFIG_OPTION_CUSTOM && c->customConfig != NULL &&
        ((const QnnHtpContext_CustomConfig_t*)c->customConfig)->option ==
            QNN_HTP_CONTEXT_CONFIG_OPTION_IO_MEM_ESTIMATION) {
      if (getenv("QNN_SHIM_VERBOSE")) fprintf(stderr, "[htp-shim] dropped IO_MEM_ESTIMATION\n");
      continue;
    }
    out[n++] = c;
  }
  out[n] = NULL;
  return out;
}

static Qnn_ErrorHandle_t shim_create(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                                     const QnnContext_Config_t** config, Qnn_ContextHandle_t* context) {
  const QnnContext_Config_t* buf[MAX_CONFIGS];
  return real.contextCreate(backend, device, drop_io_mem_estimation(config, buf), context);
}

static Qnn_ErrorHandle_t shim_set_config(Qnn_ContextHandle_t context, const QnnContext_Config_t** config) {
  const QnnContext_Config_t* buf[MAX_CONFIGS];
  return real.contextSetConfig(context, drop_io_mem_estimation(config, buf));
}

static Qnn_ErrorHandle_t shim_create_from_binary(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                                                 const QnnContext_Config_t** config, const void* binary,
                                                 Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                                 Qnn_ProfileHandle_t profile) {
  const QnnContext_Config_t* buf[MAX_CONFIGS];
  return real.contextCreateFromBinary(backend, device, drop_io_mem_estimation(config, buf), binary, size, context,
                                      profile);
}

static Qnn_ErrorHandle_t shim_create_from_binary_signal(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                                                        const QnnContext_Config_t** config, const void* binary,
                                                        Qnn_ContextBinarySize_t size, Qnn_ContextHandle_t* context,
                                                        Qnn_ProfileHandle_t profile, Qnn_SignalHandle_t signal) {
  const QnnContext_Config_t* buf[MAX_CONFIGS];
  return real.contextCreateFromBinaryWithSignal(backend, device, drop_io_mem_estimation(config, buf), binary, size,
                                                context, profile, signal);
}

static Qnn_ErrorHandle_t shim_create_from_binary_callback(Qnn_BackendHandle_t backend, Qnn_DeviceHandle_t device,
                                                          const QnnContext_Config_t** config,
                                                          const Qnn_ContextBinaryCallback_t* callback,
                                                          const void* binary, Qnn_ContextBinarySize_t size,
                                                          Qnn_ContextHandle_t* context, Qnn_ProfileHandle_t profile,
                                                          Qnn_SignalHandle_t signal) {
  const QnnContext_Config_t* buf[MAX_CONFIGS];
  return real.contextCreateFromBinaryWithCallback(backend, device, drop_io_mem_estimation(config, buf), callback,
                                                  binary, size, context, profile, signal);
}

Qnn_ErrorHandle_t QnnInterface_getProviders(const QnnInterface_t*** providerList, uint32_t* numProviders) {
  const char* path = getenv("QNN_SHIM_REAL");
  if (path == NULL) path = "/data/adas_vla_genie/shim/libQnnHtpReal.so";
  void* lib = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (lib == NULL) {
    fprintf(stderr, "[htp-shim] cannot load %s: %s\n", path, dlerror());
    return QNN_COMMON_ERROR_GENERAL;
  }
  Qnn_ErrorHandle_t (*get)(const QnnInterface_t***, uint32_t*) =
      (Qnn_ErrorHandle_t (*)(const QnnInterface_t***, uint32_t*))dlsym(lib, "QnnInterface_getProviders");
  const QnnInterface_t** list = NULL;
  uint32_t n = 0;
  if (get == NULL || get(&list, &n) != QNN_SUCCESS || n == 0) return QNN_COMMON_ERROR_GENERAL;
  if (n > MAX_PROVIDERS) n = MAX_PROVIDERS;
  real = list[0]->QNN_INTERFACE_VER_NAME;
  for (uint32_t i = 0; i < n; ++i) {
    shim_iface[i] = *list[i];
    QNN_INTERFACE_VER_TYPE* f = &shim_iface[i].QNN_INTERFACE_VER_NAME;
    if (real.contextCreate) f->contextCreate = shim_create;
    if (real.contextSetConfig) f->contextSetConfig = shim_set_config;
    if (real.contextCreateFromBinary) f->contextCreateFromBinary = shim_create_from_binary;
    if (real.contextCreateFromBinaryWithSignal) f->contextCreateFromBinaryWithSignal = shim_create_from_binary_signal;
    if (real.contextCreateFromBinaryWithCallback)
      f->contextCreateFromBinaryWithCallback = shim_create_from_binary_callback;
    shim_list[i] = &shim_iface[i];
  }
  *providerList = shim_list;
  *numProviders = n;
  return QNN_SUCCESS;
}
