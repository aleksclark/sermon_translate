import js from "@eslint/js";
import tseslint from "typescript-eslint";

export default tseslint.config(
  { ignores: ["dist/", "src/api/types.gen.ts"] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
);
