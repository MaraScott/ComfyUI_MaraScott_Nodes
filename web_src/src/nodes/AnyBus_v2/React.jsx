// ---------- React UMD loader (no imports / no JSX-runtime imports) ----------
async function ensureReactGlobals() {
    if (globalThis.React && globalThis.ReactDOM) return;
    // TIP: ship local copies instead of CDN in production:
    //   /web/assets/js/ms_assets/react.production.min.js
    //   /web/assets/js/ms_assets/react-dom.production.min.js
    await loadScript('https://unpkg.com/react@18.3.1/umd/react.production.min.js');
    await loadScript('https://unpkg.com/react-dom@18.3.1/umd/react-dom.production.min.js');
    if (!globalThis.React || !globalThis.ReactDOM) {
        console.error('[AnyBus] React/ReactDOM UMD not available');
        throw new Error('React UMD not loaded');
    }
}
function loadScript(src) {
    return new Promise((res, rej) => {
        const s = document.createElement('script');
        s.src = src; s.async = true;
        s.onload = res; s.onerror = () => rej(new Error('Failed to load ' + src));
        document.head.appendChild(s);
    });
}
function mountJSX(el, vnode) {
    if (!el.__ms_mount) {
        const host = document.createElement('div');
        (el.attachShadow ? el.attachShadow({ mode: 'open' }) : el).appendChild(host);
        el.__ms_mount = { container: host, root: globalThis.ReactDOM.createRoot(host) };
    }
    el.__ms_mount.root.render(vnode);
}

export { ensureReactGlobals, loadScript, mountJSX };