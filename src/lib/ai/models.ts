// models.ts
import { google } from "@ai-sdk/google";
import { openrouter } from "@openrouter/ai-sdk-provider";
import { LanguageModel } from "ai";
import {
  createOpenAICompatibleModels,
  openaiCompatibleModelsSafeParse,
} from "./create-openai-compatiable";
import { ChatModel } from "app-types/chat";

const staticModels = {
  google: {
    "gemini-2.5-flash": google("gemini-2.5-flash"),
    "gemini-2.5-pro": google("gemini-2.5-pro"),
    "gemini-2.5-flash-lite": google("gemini-2.5-flash-lite"),
    "gemini-2.0-flash": google("gemini-2.0-flash"),
    "gemini-2.0-flash-preview-image-generation": google(
      "gemini-2.0-flash-preview-image-generation",
    ),
    "learnlm-2.0-flash-experimental": google("learnlm-2.0-flash-experimental"),
    "gemma-3-27b-it": google("gemma-3-27b-it"),
    "gemma-3-12b-it": google("gemma-3-12b-it"),
    "gemma-3-4b-it": google("gemma-3-4b-it"),
    "gemma-3-1b-it": google("gemma-3-1b-it"),
    "gemma-3n-e4b-it": google("gemma-3n-e4b-it"),
    "gemma-3n-e2b-it": google("gemma-3n-e2b-it"),
  },
  openRouter: {
    "google/gemini-2.0-flash-exp:free": openrouter("google/gemini-2.0-flash-exp:free"),
    "moonshotai/kimi-k2:free": openrouter("moonshotai/kimi-k2:free"),
    "cognitivecomputations/dolphin-mistral-24b-venice-edition:free": openrouter("cognitivecomputations/dolphin-mistral-24b-venice-edition:free"),
    "google/gemma-3n-e2b-it:free": openrouter("google/gemma-3n-e2b-it:free"),
    "tencent/hunyuan-a13b-instruct:free": openrouter("tencent/hunyuan-a13b-instruct:free"),
    "mistralai/mistral-small-3.2-24b-instruct:free": openrouter("mistralai/mistral-small-3.2-24b-instruct:free"),
    "moonshotai/kimi-dev-72b:free": openrouter("moonshotai/kimi-dev-72b:free"),
    "deepseek/deepseek-r1-0528:free": openrouter("deepseek/deepseek-r1-0528:free"),
    "mistralai/devstral-small-2505:free": openrouter("mistralai/devstral-small-2505:free"),
    "microsoft/mai-ds-r1:free": openrouter("microsoft/mai-ds-r1:free"),
    "moonshotai/kimi-vl-a3b-thinking:free": openrouter("moonshotai/kimi-vl-a3b-thinking:free"),
    "google/gemini-2.5-pro-exp-03-25": openrouter("google/gemini-2.5-pro-exp-03-25"),
    "google/gemma-3-27b-it:free": openrouter("google/gemma-3-27b-it:free"),
    "google/gemma-3-12b-it:free": openrouter("google/gemma-3-12b-it:free"),
    "google/gemma-3n-e4b-it:free": openrouter("google/gemma-3n-e4b-it:free"),
    "deepseek/deepseek-r1:free": openrouter("deepseek/deepseek-r1:free"),
    "deepseek/deepseek-chat-v3-0324:free": openrouter("deepseek/deepseek-chat-v3-0324:free"),
    "deepseek/deepseek-r1-0528-qwen3-8b:free": openrouter("deepseek/deepseek-r1-0528-qwen3-8b:free"),
    "deepseek/deepseek-r1-distill-llama-70b:free": openrouter("deepseek/deepseek-r1-distill-llama-70b:free"),
    "meta-llama/llama-3.3-70b-instruct:free": openrouter("meta-llama/llama-3.3-70b-instruct:free"),
    "meta-llama/llama-3.2-11b-vision-instruct:free": openrouter("meta-llama/llama-3.2-11b-vision-instruct:free"),
    "meta-llama/llama-3.2-3b-instruct:free": openrouter("meta-llama/llama-3.2-3b-instruct:free"),
    "meta-llama/llama-3.1-405b-instruct:free": openrouter("meta-llama/llama-3.1-405b-instruct:free"),
    "qwen/qwen3-235b-a22b:free": openrouter("qwen/qwen3-235b-a22b:free"),
    "qwen/qwen3-235b-a22b-07-25:free": openrouter("qwen/qwen3-235b-a22b-07-25:free"),
    "qwen/qwen3-30b-a3b:free": openrouter("qwen/qwen3-30b-a3b:free"),
    "qwen/qwen2.5-vl-32b-instruct:free": openrouter("qwen/qwen2.5-vl-32b-instruct:free"),
    "nvidia/llama-3.1-nemotron-ultra-253b-v1:free": openrouter("nvidia/llama-3.1-nemotron-ultra-253b-v1:free"),
  },
};

const staticUnsupportedModels = new Set([
  staticModels.google["gemini-2.0-flash-lite"],
  staticModels.google["gemini-2.0-flash"],
  staticModels.google["gemini-2.0-flash-preview-image-generation"],
  staticModels.google["gemini-2.5-flash-lite"],
  staticModels.google["learnlm-2.0-flash-experimental"],
  staticModels.google["gemma-3-27b-it"],
  staticModels.google["gemma-3-12b-it"],
  staticModels.google["gemma-3-4b-it"],
  staticModels.google["gemma-3-1b-it"],
  staticModels.google["gemma-3n-e4b-it"],
  staticModels.google["gemma-3n-e2b-it"],
  staticModels.openRouter["qwen3-8b:free"],
  staticModels.openRouter["qwen3-14b:free"],
]);

const openaiCompatibleProviders = openaiCompatibleModelsSafeParse(
  process.env.OPENAI_COMPATIBLE_DATA,
);

const {
  providers: openaiCompatibleModels,
  unsupportedModels: openaiCompatibleUnsupportedModels,
} = createOpenAICompatibleModels(openaiCompatibleProviders);

const allModels = { ...openaiCompatibleModels, ...staticModels };

const allUnsupportedModels = new Set([
  ...openaiCompatibleUnsupportedModels,
  ...staticUnsupportedModels,
]);

export const isToolCallUnsupportedModel = (model: LanguageModel) => {
  return allUnsupportedModels.has(model);
};

const firstProvider = Object.keys(allModels)[0];
const firstModel = Object.keys(allModels[firstProvider])[0];

const fallbackModel = allModels[firstProvider][firstModel];

export const customModelProvider = {
  modelsInfo: Object.entries(allModels).map(([provider, models]) => ({
    provider,
    models: Object.entries(models).map(([name, model]) => ({
      name,
      isToolCallUnsupported: isToolCallUnsupportedModel(model),
    })),
  })),
  getModel: (model?: ChatModel): LanguageModel => {
    if (!model) return fallbackModel;
    return allModels[model.provider]?.[model.model] || fallbackModel;
  },
};
