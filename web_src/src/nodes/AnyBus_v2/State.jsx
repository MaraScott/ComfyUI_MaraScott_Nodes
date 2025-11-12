// State management for AnyBus_v2
// Global profile registry to track all AnyBus nodes by profile
export const profileRegistry = new Map();

// Global slot order registry by profile
// Stores the slot mapping: { profile -> { originalIndex -> newIndex } }
export const profileSlotOrders = new Map();

// Helper: Get or create profile entry
export function getProfileEntry(profile) {
    if (!profileRegistry.has(profile)) {
        profileRegistry.set(profile, new Set());
    }
    return profileRegistry.get(profile);
}

// Helper: Get or create slot order for profile
export function getProfileSlotOrder(profile) {
    if (!profileSlotOrders.has(profile)) {
        profileSlotOrders.set(profile, {});
    }
    return profileSlotOrders.get(profile);
}
