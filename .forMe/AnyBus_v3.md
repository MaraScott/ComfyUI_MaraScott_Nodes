# AnyBus_v3 Node Generation Prompt

## Official Documentation References

### JavaScript Frontend Development
Refer to these official ComfyUI JavaScript documentation pages for implementation guidance:
- **Overview & Basics**: https://docs.comfy.org/custom-nodes/js/javascript_overview
- **Hooks System**: https://docs.comfy.org/custom-nodes/js/javascript_hooks
- **Objects & Hijacking**: https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking
- **Settings Management**: https://docs.comfy.org/custom-nodes/js/javascript_settings
- **Dialog System**: https://docs.comfy.org/custom-nodes/js/javascript_dialog
- **Toast Notifications**: https://docs.comfy.org/custom-nodes/js/javascript_toast
- **About Panel Badges**: https://docs.comfy.org/custom-nodes/js/javascript_about_panel_badges
- **Bottom Panel Tabs**: https://docs.comfy.org/custom-nodes/js/javascript_bottom_panel_tabs
- **Sidebar Tabs**: https://docs.comfy.org/custom-nodes/js/javascript_sidebar_tabs
- **Topbar Menu**: https://docs.comfy.org/custom-nodes/js/javascript_topbar_menu
- **Context Menu Migration**: https://docs.comfy.org/custom-nodes/js/context-menu-migration
- **JavaScript Examples**: https://docs.comfy.org/custom-nodes/js/javascript_examples
- **Internationalization**: https://docs.comfy.org/custom-nodes/i18n

### Python Backend Development
Refer to these official ComfyUI Python documentation pages for implementation guidance:
- **Server Overview**: https://docs.comfy.org/custom-nodes/backend/server_overview
- **Node Lifecycle**: https://docs.comfy.org/custom-nodes/backend/lifecycle
- **Data Types**: https://docs.comfy.org/custom-nodes/backend/datatypes
- **Images & Masks**: https://docs.comfy.org/custom-nodes/backend/images_and_masks
- **Advanced Inputs**: https://docs.comfy.org/custom-nodes/backend/more_on_inputs
- **Lazy Evaluation**: https://docs.comfy.org/custom-nodes/backend/lazy_evaluation
- **Node Expansion**: https://docs.comfy.org/custom-nodes/backend/expansion
- **List Handling**: https://docs.comfy.org/custom-nodes/backend/lists
- **Code Snippets**: https://docs.comfy.org/custom-nodes/backend/snippets
- **Tensor Operations**: https://docs.comfy.org/custom-nodes/backend/tensors

## Objective
Create a dynamic bus system node (AnyBus_v3) with Python backend and JavaScript frontend that supports both BUS and Get/Set modes, profile-based node synchronization, dynamic slot management, automatic type propagation, and **centralized flow management via a dedicated sidebar tab**.

## Project Context

### Existing Infrastructure
- **Build System**: Vite 7.1.3 with dual-mode configuration (app + nodes)
- **Build Output**: `web/assets/js/` (app files) and `web/assets/js/nodes/` (node modules)
- **Build Command**: `npm run build` from `web_src` directory
- **React**: Version 19.1.1 available globally via UMD bundle
- **Python Namespace**: `MaraScott` (registration format: `f"{NAMESPACE}NodeName"`)
- **No configuration changes needed**: Just add new source files and build

### Lessons from AnyBus_v2
Critical patterns learned from previous implementation:
1. **Node Registration**: Use format `MaraScottAnyBus_v3` (no `::` separator) to match Python registration
2. **Slot Numbering**: Always extract slot numbers from input names using regex pattern `/\* (\d+)/`, never use array indices for labeling
3. **Label vs Name**: Separate internal name from display label using `addInput(name, type, { label })` syntax
4. **Dynamic Management**: Use `addInput`/`removeInput` for slot visibility, not type hiding
5. **Widget Callbacks**: Must trigger both UI updates and cross-node synchronization
6. **Event System**: Implement event emitter for real-time sidebar updates

## Core Philosophy: Centralized Flow Management

The sidebar tab is the primary interface for managing AnyBus flows. Users should be able to:
- See all AnyBus_v3 nodes at a glance with their configurations
- Group and visualize nodes by profile
- Perform bulk operations (change profile, mode, slot count) across multiple nodes
- Validate flow configuration and identify issues
- Save/load flow templates for reusable patterns
- Navigate directly to nodes from the sidebar
- Understand BUS and Get/Set connection relationships

## Architecture Requirements

### Python Backend (`py/nodes/Bus/AnyBus_v3.py`)

**Node Class**:
- Class name: `Mara_AnyBus_v3`
- Category: Use `get_category("Bus")` helper function (resolves to `MaraScott/Bus`)
- Registration: Use `f"{NAMESPACE}AnyBus_v3"` format where NAMESPACE='MaraScott'
- Class Attributes:
  - `NAME = "AnyBus v3"` (display name)
  - `SHORTCUT = "b"` (keyboard shortcut identifier)

**Imports Required**:
```python
from ...utils.constants import get_category
from ...utils.helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")
```

**Input Configuration**:
- **Required Widget Inputs**:
  - `num_slots`: Integer widget controlling slot count (default: 2, range: 1-24, step: 1)
  - `profile`: String widget for profile name (default: "default")
  - `mode`: Combo widget with options ["bus", "getset"] (default: "bus")

- **Optional Data Inputs**:
  - `bus`: ANYBUS_v3 type with forceInput=True
  - `getset_source`: STRING type with forceInput=True (for Get/Set mode source selection)
  - Slot inputs: `* 01` through `* 24` (all use `any_type` with forceInput=True)

**Output Configuration**:
- `bus`: ANYBUS_v3 type
- Slot outputs: `* 01` through `* 24` (all use `any_type`)
- Total: 25 outputs (1 ANYBUS_v3 + 24 ANY slots)
- Return names: tuple starting with "bus" followed by zero-padded slot names

**Function Implementation**:
- Function name: `fn` (set via `FUNCTION = "fn"`)
- Parameters: Accept num_slots, profile, mode as required, plus **kwargs
- Extract all slot values from kwargs using zero-padded keys (`* 01`, `* 02`, etc.)
- Return tuple containing bus data dict followed by all slot values
- Handle missing/None values appropriately
- Bus data structure: `{"profile": str, "mode": str, "slots": int}`

### JavaScript Frontend Architecture

**Registration Architecture** (`web_src/src/js/AnyBus_v3.js`):

Create a registration file that imports and registers both the node extension and sidebar tab:

```javascript
import { app } from "../../scripts/app.js";
import { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab } from "./nodes/AnyBus_v3.js";

app.registerExtension(MaraScottAnyBusNodeExtension());
app.extensionManager.registerSidebarTab(MaraScottAnyBusNodeSidebarTab());
```

**Module Organization** (`web_src/src/nodes/AnyBus_v3/`):

Implement modular structure with the following components:

1. **AnyBus_v3.jsx** (Main Entry Point & Exports)
   - Import from `"../../../scripts/app.js"` (correct relative path from nodes directory)
   - Export NODE_CLASS constant matching Python registration
   - Implement beforeRegisterNodeDef hook for node type extension
   - Implement onNodeCreated callback for initialization
   - Implement onRemoved callback for cleanup
   - Register node in global registry for sidebar access
   - Setup widget callbacks with sidebar event notifications
   - Provide getAvailableGetSetSources method for dropdown population
   - **Export MaraScottAnyBusNodeExtension()**: Returns extension object for app.registerExtension()
   - **Export MaraScottAnyBusNodeSidebarTab()**: Returns sidebar tab configuration object

2. **State.jsx** (Global State Management)
   - Maintain global registries:
     * All nodes registry (nodeId -> node instance)
     * Profile registry (profile name -> Set of nodeIds)
     * Profile slot orders (profile -> slot type ordering)
   - Implement event emitter system:
     * Event types: node-created, node-removed, profile-changed, mode-changed, slots-changed
     * Subscribe/unsubscribe methods for sidebar updates
     * Emit method for broadcasting changes
   - Export register/unregister functions for profile tracking

3. **Node.jsx** (Slot Management)
   - Implement `updateNodeSlots(node, numSlots)`:
     * Scan existing inputs to find slot inputs by regex matching names
     * Extract slot numbers from names (not array positions)
     * Sort slots by their logical number
     * Remove excess slots from highest number downward
     * Add missing slots with proper zero-padded naming (`* 01`, `* 02`, etc.)
     * Ensure name and label are properly separated
     * Manage outputs: 1 bus + numSlots slots
   - Implement `resetNodeDisconnectedSlots(node)`:
     * Iterate through inputs/outputs
     * Extract slot number from name using regex
     * Reset label to default format only if no connection
   - Implement `getNodeConfiguration()` for sidebar export
   - Implement `applyNodeConfiguration(config)` for sidebar bulk operations

4. **Bus.jsx** (Connection & Synchronization Logic)
   - Implement `getBusConnectedNodes(node, graph, visitedIds)`:
     * Traverse BUS connections to find all nodes in same profile
     * Use visited set to prevent circular traversal
     * Return array of connected node instances
   - Implement `syncConnectedNodesLabelsAndTypes(node, graph)`:
     * Get all profile-connected nodes
     * Synchronize slot labels and types across all nodes
     * Handle type propagation when connections change
   - Implement `getProfileTopology()`:
     * Return connection graph structure for sidebar visualization
     * Include both BUS and Get/Set relationships
   - Handle "default" profile special behavior:
     * When default profile node connects to non-default, adopt that profile name
     * Trigger synchronization of all affected nodes

5. **Widget.jsx** (Widget Configuration)
   - Implement `setupProfileWidget(node)`:
     * Find profile widget by name
     * Wrap callback to emit profile-changed events
     * Update profile registry on change
     * Trigger synchronization of connected nodes
   - Implement `setupNumSlotsWidget(node)`:
     * Find num_slots widget by name
     * Wrap callback to call updateNodeSlots
     * Emit slots-changed event for sidebar
   - Implement `setupModeWidget(node)`:
     * Find mode widget by name
     * Wrap callback to toggle input visibility (bus vs getset_source)
     * In "getset" mode: hide bus input, show getset_source dropdown
     * In "bus" mode: show bus input, hide getset_source
     * Emit mode-changed event for sidebar

6. **React.jsx** (React Runtime Loader)
   - Load React 19.1.1 UMD bundle from CDN or local source
   - Export React global for use in JSX components
   - Handle loading errors gracefully
   - Ensure single React instance across all modules

7. **SidebarTab.jsx** (PRIMARY UI - Centralized Management)
   - Import from `/scripts/app.js` (absolute path for runtime)
   - Export registerAnyBusFlowSidebar(container) function
   - Function receives container element from sidebar tab render callback
   - Render React-based UI directly into provided container
   - Subscribe to event emitter for real-time updates
   - Implement the following UI sections:

   **a) Flow Overview Panel**
   - Display statistics: total nodes, profile count, connection summary
   - Show all AnyBus_v3 nodes grouped by profile
   - Collapsible profile sections with node counts
   - Color-coded profile indicators for visual distinction
   - Real-time updates via event subscription

   **b) Node Cards** (for each node)
   - Display node title/ID and current configuration
   - Show profile, mode, and slot count
   - Display connection summary (inputs/outputs connected)
   - Sync status indicator (synced/pending/error)
   - Checkbox for selection in bulk operations
   - Navigation button to focus node on canvas
   - Inline controls for quick setting adjustments

   **c) Search & Filter Tools**
   - Text search by node title or ID
   - Filter by profile (dropdown with all profiles)
   - Filter by mode (bus/getset/all)
   - Filter by connection status (connected/orphaned/all)
   - Filter by slot count range

   **d) Bulk Operations Toolbar**
   - Only visible when nodes are selected
   - Change profile for all selected nodes
   - Set mode (bus/getset) for all selected nodes
   - Adjust slot count for all selected nodes
   - Apply preset configurations
   - Clear selection button

   **e) Profile Management Panel**
   - List all profiles with node counts
   - Create new profile button
   - Rename profile (updates all nodes in profile)
   - Merge profiles (combine two profiles into one)
   - Delete profile (with warning if nodes exist)
   - Set default profile for newly created nodes

   **f) Connection Visualization**
   - Tree view showing BUS connection hierarchy
   - Diagram showing Get/Set relationships
   - Highlight connected nodes on hover
   - Click to navigate to node on canvas
   - Visual indicators for connection types

   **g) Validation Panel**
   - Real-time validation of flow configuration
   - Display errors and warnings:
     * Profile conflicts (incompatible profiles trying to connect)
     * Orphaned nodes (no connections)
     * Circular Get/Set references
     * Type mismatches on slot connections
     * Default profile nodes not connected to any profile
   - Click error to navigate to problematic node
   - "Fix All" button for auto-correctable issues

   **h) Flow Templates System**
   - Save current flow configuration as named template
   - Load template to apply settings to nodes
   - Template library display with descriptions
   - Export templates as JSON files
   - Import templates from JSON files
   - Template structure includes:
     * Profile structure and names
     * Node modes and slot counts
     * Connection patterns (BUS/Get/Set relationships)
     * Custom configuration metadata

   **i) Actions Menu**
   - Refresh flow data manually
   - Export entire flow configuration (JSON)
   - Import flow configuration from file
   - Reset all nodes to default settings (with confirmation)
   - Force synchronize all profiles
   - Generate flow documentation/report
   - Open advanced flow settings dialog

   **Implementation Notes**:
   - Subscribe to flowEventEmitter on mount
   - Unsubscribe on unmount to prevent memory leaks
   - Use React state for UI reactivity
   - Implement efficient re-rendering (only update changed sections)
   - Access global registries for node data
   - Use ComfyUI canvas API for node navigation
   - Implement proper error handling for all operations

## Key Features & Behaviors

### 1. Dynamic Slot Management
- **Initial State**: Node loads with 1 BUS input/output + 2 slot inputs/outputs
- **Slot Labels**: Default format `* 01`, `* 02`, ... `* 24` (zero-padded two digits)
- **Widget Control**: num_slots widget (1-24 range) controls visible slot count
- **Sidebar Control**: Bulk slot count adjustment via sidebar selection
- **Add/Remove Pattern**: Dynamically add or remove slots using LiteGraph APIs
- **Numbering Logic**: Slot numbers extracted from names, not array positions

### 2. BUS vs Get/Set Mode
- **BUS Mode**:
  - bus input visible and connectable
  - Nodes physically connect via BUS connections
  - Profile-based synchronization across connected nodes
  - Sidebar shows BUS connection tree visualization

- **Get/Set Mode**:
  - bus input hidden (virtual connection only)
  - getset_source dropdown visible for source node selection
  - Dropdown populated from compatible AnyBus_v3 nodes
  - Virtual linking: source node's outputs map to this node's inputs
  - No physical BUS connection required
  - Sidebar shows Get/Set relationship diagram

### 3. Profile-Based Synchronization
- **Profile System**: String identifier grouping related nodes
- **Same Profile Behavior**: Nodes with identical profiles synchronize labels/types
- **Different Profiles**: Cannot connect BUS unless one is "default"
- **Default Profile Special Handling**:
  - When default profile node connects to non-default node
  - Default node adopts the non-default profile name
  - All former "default" profile nodes synchronize to new profile
  - Sidebar highlights this profile inheritance event
- **Sidebar Management**: All profile operations centralized in sidebar UI

### 4. Type Propagation
- **Initial Type**: All slots start as type "*" (ANY)
- **Connection-Based**: When slot connects, inherits type from connected output
- **Label Updates**: Slot label updates to reflect connection type
- **Cross-Node Sync**: Type changes propagate to all nodes in same profile
- **Disconnection**: Label resets to default format, type returns to "*"
- **Sidebar Display**: Show current types for each slot in node cards

### 5. Centralized Flow Templates
- **Save Operation**: Export all node configurations as template object
- **Load Operation**: Apply saved configuration to existing or new nodes
- **Template Storage**: JSON format with metadata
- **Template Contents**:
  - Profile definitions with node lists
  - Mode settings per profile/node
  - Slot count configurations
  - Connection patterns (optional)
  - Custom metadata (description, author, version)
- **Sidebar Interface**: Template library with preview and management

### 6. Flow Validation
- **Real-Time Checking**: Continuously validate configuration as nodes change
- **Error Detection**:
  - Profile conflicts when incompatible profiles attempt connection
  - Orphaned nodes with no BUS or Get/Set connections
  - Circular Get/Set reference chains
  - Type mismatches on connected slots
  - Unconnected default profile nodes
- **Warning System**: Non-critical issues displayed as warnings
- **Navigation**: Click validation error to jump to problematic node
- **Auto-Fix**: Offer automatic resolution for correctable issues

## Build Process

**Directory Structure**:
```
custom_nodes/ComfyUI_MaraScott_Nodes/
├── py/nodes/Bus/
│   ├── AnyBus_v2.py          [Existing]
│   └── AnyBus_v3.py          [NEW - Create this]
├── web_src/
│   ├── vite.config.js        [Existing - No changes needed]
│   ├── package.json          [Existing - No changes needed]
│   └── src/
│       ├── js/
│       │   ├── AnyBus_v2.js  [Existing registration]
│       │   └── AnyBus_v3.js  [NEW - Create registration file]
│       └── nodes/
│           ├── AnyBus_v2.jsx     [Existing]
│           ├── AnyBus_v2/        [Existing modules]
│           ├── AnyBus_v3.jsx     [NEW - Create this]
│           └── AnyBus_v3/        [NEW - Create these modules]
│               ├── State.jsx
│               ├── Node.jsx
│               ├── Bus.jsx
│               ├── Widget.jsx
│               ├── React.jsx
│               └── SidebarTab.jsx
└── web/assets/js/
    ├── nodes/
    │   ├── AnyBus_v2.js      [Existing compiled]
    │   └── AnyBus_v3.js      [NEW - Auto-generated by build]
    └── [other app files]
```

**Build Steps**:
1. Create Python backend file in `py/nodes/Bus/AnyBus_v3.py`
2. Create JavaScript registration file in `web_src/src/js/AnyBus_v3.js`
3. Create JavaScript main module in `web_src/src/nodes/AnyBus_v3.jsx`
4. Create JavaScript sub-modules in `web_src/src/nodes/AnyBus_v3/` directory
5. Run build command from `web_src` directory:
   ```powershell
   npm run build
   ```
6. Vite will automatically discover and compile new files
7. Output appears in `web/assets/js/nodes/AnyBus_v3.js`
8. Registration file imports from compiled node modules
9. Supporting modules bundled or code-split as appropriate
10. No manual configuration changes required

**Build Configuration Notes**:
- Existing Vite config handles both app and nodes builds
- Nodes build uses classic JSX transformation
- React bundled with node modules for standalone operation
- ES2019 target for broad compatibility
- Code splitting for shared dependencies (vendor chunks)
- Direct output to `web/assets/js/` (no intermediate tmp directory)

## Implementation Guidelines

### Python Backend Pattern
Complete Python backend structure:

```python
from ...utils.constants import get_category
from ...utils.helper import AlwaysEqualProxy

any_type = AlwaysEqualProxy("*")

class Mara_AnyBus_v3:
    """
    AnyBus v3 - Dynamic bus system with profile-based synchronization and Get/Set mode
    """

    NAME = "AnyBus v3"
    SHORTCUT = "b"

    @classmethod
    def INPUT_TYPES(cls):
        # Generate slot inputs dynamically (* 01 through * 24)
        slot_inputs = {f"* {i:02d}": (any_type, {"forceInput": True}) for i in range(1, 25)}

        return {
            "required": {
                "num_slots": ("INT", {"default": 2, "min": 1, "max": 24, "step": 1}),
                "profile": ("STRING", {"default": "default"}),
                "mode": (["bus", "getset"], {"default": "bus"}),
            },
            "optional": {
                "bus": ("ANYBUS_v3", {"forceInput": True}),
                "getset_source": ("STRING", {"forceInput": True}),
                **slot_inputs
            }
        }

    RETURN_TYPES = ("ANYBUS_v3",) + (any_type,) * 24
    RETURN_NAMES = ("bus",) + tuple(f"* {i:02d}" for i in range(1, 25))
    FUNCTION = "fn"
    CATEGORY = get_category("Bus")

    def fn(self, num_slots, profile, mode, **kwargs):
        # Extract bus data
        bus_data = kwargs.get("bus", {"profile": profile, "mode": mode, "slots": num_slots})

        # Update bus data with current settings
        bus_data.update({
            "profile": profile,
            "mode": mode,
            "slots": num_slots
        })

        # Extract slot values
        slot_values = []
        for i in range(1, 25):
            key = f"* {i:02d}"
            value = kwargs.get(key, None)
            slot_values.append(value)

        # Return bus data followed by all slot values
        return (bus_data,) + tuple(slot_values)
```

### JavaScript Import Paths
Critical import path patterns:

**In `web_src/src/nodes/AnyBus_v3.jsx`** (main node module):
```javascript
import { app } from "../../../scripts/app.js";  // Relative path from src/nodes/
```

**In `web_src/src/nodes/AnyBus_v3/SidebarTab.jsx`** (sub-module):
```javascript
import { app } from '/scripts/app.js';  // Absolute path for runtime
```

**In `web_src/src/js/AnyBus_v3.js`** (registration file):
```javascript
import { app } from "../../scripts/app.js";
import { MaraScottAnyBusNodeExtension, MaraScottAnyBusNodeSidebarTab } from "./nodes/AnyBus_v3.js";
```

### Registration Pattern
- Create separate registration file in `web_src/src/js/AnyBus_v3.js`
- Import MaraScottAnyBusNodeExtension and MaraScottAnyBusNodeSidebarTab from node module
- Register extension with `app.registerExtension(MaraScottAnyBusNodeExtension())`
- Register sidebar tab with `app.extensionManager.registerSidebarTab(MaraScottAnyBusNodeSidebarTab())`
- Extension function returns object with name and beforeRegisterNodeDef hook
- Sidebar tab function returns object with id, icon, title, tooltip, and render callback

### Node Registration
- Use `beforeRegisterNodeDef` hook to extend node type
- Check `nodeData.name` matches `NODE_CLASS` exactly
- Wrap `onNodeCreated` to add initialization logic
- Wrap `onRemoved` to add cleanup logic
- Register node in global registry on creation
- Emit events for sidebar notification

### Widget Setup Pattern
- Locate widgets by name (`node.widgets.find(w => w.name === "widget_name")`)
- Preserve original callback if it exists
- Wrap callback to add custom behavior
- Emit events to notify sidebar of changes
- Trigger appropriate synchronization functions
- Update UI elements (node size, canvas redraw)

### Slot Management Pattern
- Find slot inputs by regex matching names (not by position)
- Extract slot number from name using regex capture group
- Sort slots by logical number before operations
- Remove slots from highest number downward
- Add slots with sequential numbering
- Use separate name and label parameters
- Update outputs to match slot configuration
- Trigger canvas redraw after changes

### Event System Pattern
- Define event emitter in State module
- Implement subscribe/unsubscribe methods
- Emit events on all significant changes
- Subscribe in sidebar on mount
- Unsubscribe on unmount
- Use event types as constants for consistency

### Sidebar Integration Pattern
- Export registerAnyBusFlowSidebar(container) function from SidebarTab.jsx
- Function receives container element as parameter
- Render React components directly into container
- Subscribe to flowEventEmitter on function call
- Access global registries for data
- Use ComfyUI canvas API (via app.graph) for navigation
- Implement bulk operations by iterating selected nodes
- Export/import using JSON serialization
- Handle errors gracefully with user feedback
- Main entry point exports MaraScottAnyBusNodeSidebarTab() returning tab config

### Profile Synchronization Pattern
- Maintain profile registry mapping profile names to node sets
- On profile change: update registry, notify sidebar, sync connected nodes
- Traverse BUS connections to find all profile members
- Apply changes to all nodes in profile simultaneously
- Handle default profile inheritance specially
- Emit events after synchronization completes

### Type Propagation Pattern
- Monitor connection events on inputs
- Extract type from connected output
- Update input type and label
- Propagate to all profile-connected nodes
- Reset on disconnection
- Update sidebar display

## Testing Requirements

**Functional Testing**:
1. Node creation: Verify node loads with correct initial state (1 BUS + 2 slots)
2. Slot adjustment: Test changing num_slots from 1 to 24, verify slots add/remove correctly
3. Label consistency: Verify labels always show correct slot numbers (01-24) after operations
4. Mode switching: Test switching between bus and getset modes, verify input visibility
5. Profile change: Change profile, verify registry updates and synchronization occurs
6. BUS connection: Connect BUS between same-profile nodes, verify type propagation
7. Get/Set linking: Select source in getset mode, verify virtual connection works
8. Default profile: Connect default to non-default, verify profile inheritance
9. Disconnection: Disconnect slots, verify labels reset to default format
10. Type propagation: Connect typed slot, verify type flows across profile nodes

**Sidebar Testing**:
11. Tab registration: Verify AnyBus Flow tab appears in sidebar
12. Node discovery: Create multiple nodes, verify all appear in sidebar
13. Real-time updates: Change settings in canvas, verify sidebar updates immediately
14. Profile grouping: Create nodes with different profiles, verify grouping correct
15. Search: Search for nodes by title, verify filtering works
16. Filter by profile: Filter nodes by profile, verify correct subset shown
17. Filter by mode: Filter by bus/getset, verify filtering works
18. Bulk profile change: Select multiple nodes, change profile, verify all update
19. Bulk mode change: Select nodes, switch mode, verify all change correctly
20. Bulk slot adjustment: Select nodes, change slot count, verify all adjust
21. Node navigation: Click node in sidebar, verify canvas navigates to node
22. Connection visualization: Verify BUS tree and Get/Set diagram display correctly
23. Validation: Create profile conflict, verify error appears in validation panel
24. Orphaned node: Create unconnected node, verify warning appears
25. Template save: Save flow as template, verify JSON structure correct
26. Template load: Load template, verify settings apply correctly
27. Export config: Export flow, verify JSON contains all node data
28. Import config: Import configuration, verify nodes update correctly
29. Profile rename: Rename profile in sidebar, verify all nodes update
30. Profile merge: Merge two profiles, verify nodes combine correctly

**Edge Cases**:
31. Maximum slots: Set num_slots to 24, verify all slots appear correctly
32. Minimum slots: Set num_slots to 1, verify excess slots removed
33. Rapid changes: Quickly change settings multiple times, verify stability
34. Circular Get/Set: Create circular reference, verify validation detects it
35. Mixed profiles: Create complex profile relationships, verify all sync correctly

## Success Criteria

✅ **Python Backend**: Node class properly defined with correct INPUT_TYPES and outputs
✅ **Node Registration**: JavaScript NODE_CLASS matches Python registration exactly
✅ **Initial Load**: Node appears with 1 BUS + 2 slots by default
✅ **Slot Management**: Dynamic add/remove works for 1-24 slot range
✅ **Label Numbering**: Labels always show correct slot numbers (never array indices)
✅ **Mode Switching**: BUS and Get/Set modes toggle input visibility correctly
✅ **Profile System**: Profile widget allows custom names and updates registry
✅ **BUS Connections**: Same-profile nodes can connect via BUS
✅ **Get/Set Linking**: Virtual linking works via getset_source dropdown
✅ **Type Propagation**: Types flow across connections and propagate to profile nodes
✅ **Default Profile**: Default profile inherits name when connecting to non-default
✅ **Disconnection**: Labels reset properly when slots disconnect
✅ **Sidebar Tab**: AnyBus Flow tab appears in sidebar with all features
✅ **Node Discovery**: Sidebar automatically finds and displays all AnyBus_v3 nodes
✅ **Real-Time Updates**: Sidebar reflects changes immediately via event system
✅ **Profile Grouping**: Sidebar groups nodes by profile with visual distinction
✅ **Search & Filter**: All filtering options work correctly
✅ **Bulk Operations**: Can change profile, mode, slots for multiple nodes simultaneously
✅ **Node Navigation**: Clicking node in sidebar navigates to it on canvas
✅ **Connection Visualization**: BUS tree and Get/Set diagram display correctly
✅ **Validation**: All error types detected and displayed with navigation
✅ **Templates**: Can save/load templates with full configuration
✅ **Export/Import**: Flow configurations export/import correctly as JSON
✅ **Profile Management**: Rename, merge, delete operations work correctly
✅ **Build Success**: `npm run build` completes without errors
✅ **No Weird Numbering**: Labels remain consistent through all operations

## Key Principles

1. **Follow Official Documentation**: All implementations must align with official ComfyUI patterns
2. **Sidebar-First Design**: Most management happens in sidebar, not individual nodes
3. **Event-Driven Updates**: Use event system for real-time sidebar synchronization
4. **Bulk Operations Priority**: Design for managing many nodes simultaneously
5. **Visual Clarity**: Use grouping, color coding, icons for quick understanding
6. **Validation Built-In**: Automatically detect and surface configuration issues
7. **Template-Driven Workflows**: Encourage reusable flow patterns
8. **Non-Destructive Operations**: All operations reversible or modifiable
9. **Export/Import Support**: Flow configurations portable and shareable
10. **Regex-Based Numbering**: Always extract slot numbers from names, never use array indices
11. **Proper Cleanup**: Unregister nodes, unsubscribe events, prevent memory leaks
12. **Error Handling**: Graceful degradation, user feedback, no silent failures

---

## Key Implementation Details

### Python Backend Specifics
1. **Use AlwaysEqualProxy**: Import `AlwaysEqualProxy` from utils.helper and use `any_type = AlwaysEqualProxy("*")` for dynamic typing
2. **Use get_category helper**: Import `get_category` from utils.constants and use `CATEGORY = get_category("Bus")`
3. **Add NAME and SHORTCUT**: Class attributes for display name and keyboard shortcut
4. **Function name is "fn"**: Set `FUNCTION = "fn"` and implement `def fn(self, ...)` method
5. **Zero-padded slot keys**: Always use `f"* {i:02d}"` format for slot keys (e.g., `* 01`, `* 02`)

### JavaScript Import Paths
1. **Main node module** (`src/nodes/AnyBus_v3.jsx`): Use `"../../../scripts/app.js"` (relative from nodes directory)
2. **Sub-modules** (`src/nodes/AnyBus_v3/*.jsx`): Use `'/scripts/app.js'` (absolute path for runtime)
3. **Registration file** (`src/js/AnyBus_v3.js`): Use `"../../scripts/app.js"` (relative from js directory)

### Build Configuration
- **Do NOT modify vite.config.js** - It already has proper externalization for `/scripts/` imports
- **Two-stage build**: Run `npm run build:nodes` first, then `npm run build:app`
- **Output locations**:
  - Node modules: `web/assets/js/nodes/AnyBus_v3.js` + `web/assets/js/nodes/AnyBus_v3/*.js`
  - Registration file: `web/assets/js/AnyBus_v3.[timestamp].js`

---

**Implementation Note**: This prompt intentionally avoids code snippets to allow flexibility in implementation approach. Follow the official ComfyUI documentation patterns and adapt to the specific requirements while maintaining the described functionality and architecture.
