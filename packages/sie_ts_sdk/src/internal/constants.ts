/**
 * Internal constants for the SIE TypeScript SDK
 */

export const MSGPACK_CONTENT_TYPE = "application/msgpack";
export const JSON_CONTENT_TYPE = "application/json";

export const HTTP_ACCEPTED = 202;
export const HTTP_CLIENT_ERROR_MIN = 400;
export const HTTP_CLIENT_ERROR_MAX = 499;
export const HTTP_SERVER_ERROR_MIN = 500;
export const HTTP_SERVER_ERROR_MAX = 599;

// Default timeouts and delays
export const DEFAULT_TIMEOUT = 30_000; // 30 seconds
export const DEFAULT_PROVISION_TIMEOUT = 300_000; // 5 minutes (300s matches Python SDK)
export const DEFAULT_RETRY_DELAY = 5_000; // 5 seconds (matches Python SDK)
export const DEFAULT_MAX_RETRY_DELAY = 30_000; // 30 seconds
export const DEFAULT_LEASE_RENEWAL_INTERVAL = 60_000; // 1 minute

// LoRA loading retry settings
export const LORA_LOADING_MAX_RETRIES = 10; // Max retries for LoRA loading
export const LORA_LOADING_DEFAULT_DELAY = 1_000; // 1 second default retry delay
export const LORA_LOADING_ERROR_CODE = "LORA_LOADING"; // Error code from server

// Model loading retry settings
export const MODEL_LOADING_MAX_RETRIES = 60; // Max retries (60 * 5s = 5 min)
export const MODEL_LOADING_DEFAULT_DELAY = 5_000; // 5 seconds default retry delay
export const MODEL_LOADING_ERROR_CODE = "MODEL_LOADING"; // Error code from server

// Version negotiation headers
export const SDK_VERSION_HEADER = "X-SIE-SDK-Version";
export const SERVER_VERSION_HEADER = "X-SIE-Server-Version";
