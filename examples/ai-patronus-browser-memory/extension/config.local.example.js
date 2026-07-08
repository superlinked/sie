// Optional local defaults. Copy this to config.local.js, which is gitignored.
// The popup settings override any value here. Never commit real keys.
self.__PATRONUS_DEFAULTS = {
  GEMINI_API_KEY: "",          // https://aistudio.google.com/api-keys
  GEMINI_MODEL: "gemini-2.5-flash",
  SLNG_API_KEY: "",            // https://app.slng.ai
  TAVILY_API_KEY: "",          // https://tavily.com
  MUBIT_API_KEY: "",           // https://console.mubit.ai
  N8N_WEBHOOK_URL: "",         // Optional n8n webhook URL
  SUPERLINKED_URL: "",         // Example: http://localhost:8800
  SUPERLINKED_TOKEN: ""        // Optional bearer token for an auth-enabled endpoint
};
