import { readFileSync, statSync } from "node:fs";
import { extname, resolve } from "node:path";
import { UserError } from "./errors.js";

export function documentPage(args: string[], defaultFile: string, maxBytes: number) {
  const arguments_ = args.filter((argument) => argument !== "--");
  if (arguments_.length > 1) throw new UserError("Supply at most one raster page.");
  const file = resolve(arguments_[0] ?? defaultFile);
  if (![".png", ".jpg", ".jpeg", ".webp"].includes(extname(file).toLowerCase()))
    throw new UserError(
      "Supply one raster page as PNG, JPEG or WebP; render PDF pages before calling this example.",
    );
  let size: number;
  try {
    const stat = statSync(file);
    if (!stat.isFile()) throw new Error("Not a file");
    size = stat.size;
  } catch {
    throw new UserError(
      "Cannot read the raster page. Check its path or run pnpm fetch-data for the sample page.",
    );
  }
  if (size > maxBytes) throw new UserError("Document exceeds documentBytes in config.json.");
  try {
    return { file, image: readFileSync(file) };
  } catch {
    throw new UserError("Cannot read the raster page. Check its path and file permissions.");
  }
}

export function learningRun(args: string[]): string | undefined {
  const arguments_ = args.filter((argument) => argument !== "--");
  if (arguments_.length === 0) return undefined;
  if (arguments_.length === 2 && arguments_[0] === "--verify-run" && arguments_[1]?.trim())
    return arguments_[1];
  throw new UserError("Use pnpm learn, or pnpm learn -- --verify-run PATH.");
}
