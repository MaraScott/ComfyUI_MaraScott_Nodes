/**
 * File: helper.js
 * Project: MaraScottUniversalBusNode
 * Author: David Asquiedge
 *
 * Copyright (c) 2024 David Asquiedge
 * 
 * Inspired by Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

// - crude uuid
export function makeUUID() {
    let dt = new Date().getTime()
    const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (dt + Math.random() * 16) % 16 | 0
        dt = Math.floor(dt / 16)
        return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
    })
    return uuid
}

//- local storage manager
export class LocalStorageManager {
    constructor(namespace) {
        this.namespace = namespace;
    }

    _namespacedKey(key) {
        return `${this.namespace}:${key}`;
    }

    set(key, value) {
        const serializedValue = JSON.stringify(value);
        localStorage.setItem(this._namespacedKey(key), serializedValue);
    }

    get(key, default_val = null) {
        const value = localStorage.getItem(this._namespacedKey(key));
        return value ? JSON.parse(value) : default_val;
    }

    remove(key) {
        localStorage.removeItem(this._namespacedKey(key));
    }

    clear() {
        Object.keys(localStorage)
            .filter(k => k.startsWith(this.namespace + ':'))
            .forEach(k => localStorage.removeItem(k));
    }
}

// - log utilities

function createLogger(emoji, color, consoleMethod = 'log') {
    return function (message, ...args) {
        if (window.marascott?.DEBUG) {
            console[consoleMethod](
                `%c${emoji} ${message}`,
                `color: ${color};`,
                ...args
            )
        }
    }
}

export const infoLogger = createLogger('ℹ️', 'yellow')
export const warnLogger = createLogger('⚠️', 'orange', 'warn')
export const errorLogger = createLogger('🔥', 'red', 'error')
export const successLogger = createLogger('✅', 'green')

export const log = (...args) => {
    if (window.marascott?.DEBUG) {
        console.debug(...args)
    }
}

// Nodes that allow you to tunnel connections for cleaner graphs
export const setColorAndBgColor = (type) => {
    const colorMap = {
        "MODEL": LGraphCanvas.node_colors.blue,
        "LATENT": LGraphCanvas.node_colors.purple,
        "VAE": LGraphCanvas.node_colors.red,
        "CONDITIONING": LGraphCanvas.node_colors.brown,
        "IMAGE": LGraphCanvas.node_colors.pale_blue,
        "CLIP": LGraphCanvas.node_colors.yellow,
        "FLOAT": LGraphCanvas.node_colors.green,
        "MASK": LGraphCanvas.node_colors.cyan,
        "INT": { color: "#1b4669", bgcolor: "#29699c" },

    };

    const colors = colorMap[type];
    if (colors) {
        this.color = colors.color;
        this.bgcolor = colors.bgcolor;
    } else {
        // Handle the default case if needed
    }
}

let isAlertShown = false;

export const showAlertWithThrottle = (message, delay) => {
    if (!isAlertShown) {
        isAlertShown = true;
        alert(message);
        setTimeout(() => isAlertShown = false, delay);
    }
}
