#ifndef MONOTONIC_RNNT_STATUS_H
#define MONOTONIC_RNNT_STATUS_H

typedef enum {
    RNNT_STATUS_SUCCESS = 0,
    RNNT_STATUS_MEMOPS_FAILED = 1,
    RNNT_STATUS_INVALID_VALUE = 2,
    RNNT_STATUS_EXECUTION_FAILED = 3,
    RNNT_STATUS_UNKNOWN_ERROR = 4
} RNNTStatus;

/**
 * Returns a string containing a description of status that was passed in
 *  \param[in] status identifies which string should be returned
 *  \return C style string containing the text description
 **/
const char *rnntGetStatusString(RNNTStatus status);


#endif //MONOTONIC_RNNT_STATUS_H
