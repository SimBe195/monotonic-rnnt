#include "status.h"

const char *rnntGetStatusString(RNNTStatus status) {
    switch (status) {
        case RNNT_STATUS_SUCCESS:
            return "no error";
        case RNNT_STATUS_MEMOPS_FAILED:
            return "cuda memcpy or memset failed";
        case RNNT_STATUS_INVALID_VALUE:
            return "invalid value";
        case RNNT_STATUS_EXECUTION_FAILED:
            return "execution failed";
        case RNNT_STATUS_UNKNOWN_ERROR:
        default:
            return "unknown error";
    }
}
