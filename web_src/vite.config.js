// vite.config.js
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react'; // used only for APP build
import { viteStaticCopy } from 'vite-plugin-static-copy';
import { readdirSync, statSync, rmSync, mkdirSync, cpSync, existsSync } from 'node:fs';
import { resolve, basename, join, dirname } from 'node:path';

// ===== Helpers =====
const pad = (n) => String(n).padStart(2, '0');
const version = (() => {
  const d = new Date();
  return `${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}T${pad(d.getHours())}${pad(d.getMinutes())}${pad(d.getSeconds())}`;
})();

const CWD = process.cwd();
const TMP_DIR = resolve(CWD, 'tmp');
const SRC_ROOT = resolve(CWD, 'src');
const EXT_ROOT = resolve(CWD, '..');                     // extension root (final app destination)
const JS_DEST = resolve(EXT_ROOT, 'web', 'assets', 'js'); // final js destination
const NODES_DEST = resolve(JS_DEST, 'nodes');             // final nodes destination
const DOCS_DEST = resolve(JS_DEST, 'docs');               // final docs destination

function ensureDir(p) { if (!existsSync(p)) mkdirSync(p, { recursive: true }); }
function rimraf(p) { if (existsSync(p)) rmSync(p, { recursive: true, force: true }); }
function mirrorDir(src, dest) { ensureDir(dest); cpSync(src, dest, { recursive: true }); }
function mirrorIfExists(src, dest) { if (existsSync(src)) mirrorDir(src, dest); }

function listTopLevelFiles(dir, exts = ['.js']) {
  try {
    const abs = resolve(CWD, dir);
    return Object.fromEntries(
      readdirSync(abs, { withFileTypes: true })
        .filter((e) => e.isFile() && exts.some((x) => e.name.endsWith(x)))
        .map((e) => [basename(e.name, exts.find((x) => e.name.endsWith(x))), join(abs, e.name)])
    );
  } catch { return {}; }
}
function listNodeEntriesRec(dir, exts = ['.js', '.jsx']) {
  const abs = resolve(CWD, dir);
  const files = {};
  function walk(d, root = abs) {
    for (const name of readdirSync(d)) {
      const p = join(d, name);
      const st = statSync(p);
      if (st.isDirectory()) walk(p, root);
      else if (exts.some((x) => p.endsWith(x))) {
        const rel = p.substring(root.length + 1).replace(/\.[^.]+$/, '');
        files[rel] = p;
      }
    }
  }
  try { walk(abs); } catch {}
  return files;
}

// Mark relative imports that resolve OUTSIDE src/ as external (leave as-is)
function externalizeOutsideSrc(id, importer) {
  if (!id || !(id.startsWith('./') || id.startsWith('../'))) return false;
  const abs = resolve(importer ? dirname(importer) : CWD, id);
  return !abs.startsWith(SRC_ROOT + '\\') && !abs.startsWith(SRC_ROOT + '/');
}

// Keep ./nodes/... imports literal in the APP build output
const NODES_PREFIX = './nodes/';
function NodesImportShim() {
  return {
    name: 'nodes-import-shim',
    enforce: 'pre',
    resolveId(source) {
      if (!source) return null;
      if (source.startsWith(NODES_PREFIX)) return { id: source, external: true };
      const m = source.match(/^\.\/nodes([^/].*\.js)$/);
      if (m) {
        const fixed = `./nodes/${m[1].replace(/^\//, '')}`;
        return { id: fixed, external: true };
      }
      return null;
    },
  };
}

// Ensure TMP lifecycle (create before build)
function EnsureTmpDir() {
  return { name: 'ensure-tmp-dir', apply: 'build', buildStart() { ensureDir(TMP_DIR); } };
}

// ===== APP build (versioned → JS_DEST via mirror) =====
function makeAppConfig() {
  const appInputs = listTopLevelFiles('src/js', ['.js']);

  // Do final mirror/cleanup AFTER all plugins finished emitting (closeBundle)
  function AppCleanAndMirror() {
    return {
      name: 'app-clean-and-mirror',
      apply: 'build',
      enforce: 'post',
      closeBundle() {
        // Clean ms_* folders and old versioned entries
        rimraf(resolve(JS_DEST, 'ms_chunks'));
        rimraf(resolve(JS_DEST, 'ms_assets'));
        const entryNames = Object.keys(appInputs);
        const jsInRoot = existsSync(JS_DEST) ? readdirSync(JS_DEST) : [];
        for (const file of jsInRoot) {
          for (const name of entryNames) {
            if (new RegExp(`^${name}\\.\\d{8}T\\d{6}\\.js$`).test(file)) {
              rimraf(resolve(JS_DEST, file));
            }
          }
        }
        // Replace docs and mirror all outputs
        rimraf(DOCS_DEST);
        mirrorDir(TMP_DIR, JS_DEST);

        // Final cleanup: remove TMP entirely
        rimraf(TMP_DIR);
      },
    };
  }

  return {
    appType: 'custom',
    root: './',
    resolve: {
      alias: [
        { find: '@src', replacement: resolve(CWD, 'src') },
        { find: '@js', replacement: resolve(CWD, 'src/js') },
        { find: '@nodes', replacement: resolve(CWD, 'src/nodes') },
      ],
    },
    build: {
      outDir: TMP_DIR,          // build into TMP, then mirror to JS_DEST
      emptyOutDir: true,
      manifest: false,
      rollupOptions: {
        input: appInputs,
        external: (id, importer) =>
          externalizeOutsideSrc(id, importer) ||
          id.startsWith(NODES_PREFIX) ||
          /^\.\/nodes[^/].*\.js$/.test(id),
        output: {
          entryFileNames: ({ name }) => `${name}.${version}.js`,
          chunkFileNames: `ms_chunks/[name].${version}.js`,
          assetFileNames: (assetInfo) => {
            const ext = assetInfo.name?.split('.').pop();
            const stem = assetInfo.name?.replace(/\.[^.]+$/, '') ?? 'asset';
            return `ms_assets/${stem}.${version}.${ext}`;
          },
          format: 'es',
        },
        preserveEntrySignatures: 'strict',
      },
    },
    plugins: [
      EnsureTmpDir(),
      NodesImportShim(),
      react(),
      // Copy docs → TMP/docs (keep structure)
      viteStaticCopy({ targets: [{ src: 'src/docs/**/*', dest: 'docs', flatten: false }] }),
      AppCleanAndMirror(),
    ],
  };
}

// ===== NODES build (classic JSX, NOT versioned → /nodes via mirror; docs → JS_DEST/docs) =====
function makeNodesConfig() {
  const nodeInputs = listNodeEntriesRec('src/nodes', ['.js', '.jsx']);
  const NODES_TMP = resolve(TMP_DIR, 'nodes');

  // Do final mirror/cleanup AFTER all plugins (closeBundle)
  function NodesCleanAndMirror() {
    return {
      name: 'nodes-clean-and-mirror',
      apply: 'build',
      enforce: 'post',
      closeBundle() {
        rimraf(NODES_DEST);
        mirrorDir(NODES_TMP, NODES_DEST);
        rimraf(TMP_DIR);
      },
    };
  }

  return {
    appType: 'custom',
    root: './',
    resolve: { alias: [{ find: '@nodes', replacement: resolve(CWD, 'src/nodes') }] },
    esbuild: {
      jsx: 'transform',                 // classic JSX (no jsx-runtime)
      jsxFactory: 'globalThis.React.createElement',
      jsxFragment: 'globalThis.React.Fragment',
    },
    build: {
      outDir: TMP_DIR,                  // write under TMP/*
      emptyOutDir: true,
      sourcemap: false,
      target: 'es2019',
    //   minify: 'terser',
      terserOptions: {
        module: true,
        mangle: { module: true, toplevel: true, reserved: ['__esModule', 'default'] },
        compress: { passes: 2, toplevel: true, dead_code: true, drop_console: true, drop_debugger: true },
        format: { comments: false },
      },
      rollupOptions: {
        input: nodeInputs,
        // bundle react & react-dom; don't externalize them
        external: (id, importer) => {
          if (id === 'react' || id === 'react-dom' || id?.startsWith('react/')) return false;
          return externalizeOutsideSrc(id, importer);
        },
        output: {
          preserveModules: false,
          format: 'es',
          entryFileNames: `nodes/[name].js`,
          chunkFileNames: `nodes/ms_chunks/[name].js`,
          assetFileNames: (assetInfo) => {
            const ext = assetInfo.name?.split('.').pop();
            const stem = assetInfo.name?.replace(/\.[^.]+$/, '') ?? 'asset';
            return `nodes/ms_assets/${stem}.${ext}`;
          },
          manualChunks(id) {
            if (!id) return;
            if (id.includes('node_modules/react') || id.includes('node_modules/react-dom')) {
              return 'nodes/vendor';
            }
          },
        },
        preserveEntrySignatures: 'exports-only',
        treeshake: true,
      },
    },
    plugins: [
      EnsureTmpDir(),
      // copy docs → TMP/docs in nodes mode too (optional but consistent)
      NodesCleanAndMirror(),
    ],
  };
}

// ===== Select by mode =====
export default defineConfig(({ mode }) => {
  if (mode === 'nodes') return makeNodesConfig();
  return makeAppConfig();
});
