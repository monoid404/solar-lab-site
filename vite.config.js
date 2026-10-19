import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  // Relative paths support both the repository Pages URL and a user site URL.
  base: './',
  plugins: [react()],
});
