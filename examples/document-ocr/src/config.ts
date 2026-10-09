export type ModelOption = {
  id: string;
  label: string;
  description: string;
  gpuRequired?: boolean;
  /** Adapter-specific options passed via SIE's `options` field. */
  options?: Record<string, unknown>;
};

export const RECOGNITION_MODELS: ModelOption[] = [
  {
    id: "PaddlePaddle/PaddleOCR-VL-1.5:transformers",
    label: "PaddleOCR-VL-1.5",
    description:
      "Paddle's VLM-OCR, 0.9B, transformers profile. Preloaded on the CPU default image with Donut and GLiNER. Six task modes. Falls back to fp32 off CUDA.",
    options: { task: "ocr" },
  },
  {
    id: "lightonai/LightOnOCR-2-1B",
    label: "LightOnOCR-2-1B",
    description:
      "Pixtral encoder + Qwen3 decoder, 2.1B. Strong Markdown output across dense layouts. GPU sglang-vision image only.",
    gpuRequired: true,
  },
  {
    id: "zai-org/GLM-OCR",
    label: "GLM-OCR",
    description:
      "CogViT + GLM-0.5B decoder, 9B in bfloat16. Premium quality, needs ~18 GB VRAM. GPU sglang-vision image only; not preloaded with LightOnOCR.",
    gpuRequired: true,
  },
];

export const STRUCTURED_MODELS: ModelOption[] = [
  {
    id: "naver-clova-ix/donut-base-finetuned-cord-v2",
    label: "Donut on CORD (receipts)",
    description:
      "Fine-tuned for the CORD receipt schema. Pixels in, nested JSON out. Preloaded by the CPU compose (default image); not on the GPU sglang vision image.",
  },
  {
    id: "naver-clova-ix/donut-base-finetuned-docvqa",
    label: "Donut on DocVQA",
    description:
      "Same Donut architecture, fine-tuned for visual question answering. Returns text answers. Preloaded by the CPU compose (default image); not on the GPU sglang vision image.",
  },
];

export const NER_MODELS: ModelOption[] = [
  {
    id: "urchade/gliner_multi-v2.1",
    label: "GLiNER multi (multilingual)",
    description:
      "280M, zero-shot NER, 100+ languages. CPU default. Served by the CPU compose (default image); not on the GPU sglang vision image.",
  },
  {
    id: "urchade/gliner_large-v2.1",
    label: "GLiNER large (English)",
    description:
      "440M, English-focused, higher quality on English text. Served by the CPU compose (default image); not on the GPU sglang vision image.",
  },
  {
    id: "urchade/gliner_multi_pii-v1",
    label: "GLiNER multi PII",
    description:
      "GLiNER fine-tuned for PII extraction. Good for redaction-style pipelines on documents. Served by the CPU compose (default image); not on the GPU sglang vision image.",
  },
  {
    id: "numind/NuNER_Zero",
    label: "NuNER Zero",
    description:
      "NuMind's zero-shot NER. Different architecture from GLiNER; useful for comparing zero-shot NER families on the same input text. Served by the CPU compose (default image); not on the GPU sglang vision image.",
  },
];

export const config = {
  sieUrl: process.env.SIE_URL ?? "http://localhost:8080",
  sieApiKey: process.env.SIE_API_KEY,

  defaults: {
    recognition: RECOGNITION_MODELS[0].id,
    structured: STRUCTURED_MODELS[0].id,
    ner: NER_MODELS[0].id,
  },

  paths: {
    samples: "data/samples/index.json",
    sampleDir: "data/samples",
  },

  port: Number(process.env.PORT ?? 3032),
} as const;
