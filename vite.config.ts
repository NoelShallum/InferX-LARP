import { defineConfig } from 'vitest/config';
import preact from '@preact/preset-vite';
import { viteSingleFile } from 'vite-plugin-singlefile';

const single = process.env.SINGLE_FILE === '1';

export default defineConfig({
  base: './',
  plugins: [preact(), ...(single ? [viteSingleFile()] : [])],
  build: { target: 'es2022', sourcemap: false },
  test: { include: ['src/**/*.test.ts', 'scripts/**/*.test.mjs'], environment: 'node' },
});
