using the following doc
https://docs.comfy.org/custom-nodes/js/javascript_overview
https://docs.comfy.org/custom-nodes/js/javascript_hooks
https://docs.comfy.org/custom-nodes/js/javascript_objects_and_hijacking
https://docs.comfy.org/custom-nodes/js/javascript_settings
https://docs.comfy.org/custom-nodes/js/javascript_dialog
https://docs.comfy.org/custom-nodes/js/javascript_toast
https://docs.comfy.org/custom-nodes/js/javascript_about_panel_badges
https://docs.comfy.org/custom-nodes/js/javascript_bottom_panel_tabs
https://docs.comfy.org/custom-nodes/js/javascript_sidebar_tabs
https://docs.comfy.org/custom-nodes/js/javascript_topbar_menu
https://docs.comfy.org/custom-nodes/js/context-menu-migration
https://docs.comfy.org/custom-nodes/js/javascript_examples
https://docs.comfy.org/custom-nodes/i18n

let's focus on AnyBusNode_v3 definition

it has on load 1 bus ANYBUS_v3 type input and 1 any type labeled "* 01"  input and it got a bus ANYBUS_v3 output and 1 any type labeled "* 01" output
the input except the BUS synchronized label if/when changed
it have a 1 profile field string which allow same profile bus node to link the bus input/output together
2 different profile can t be connected except if the connection output to input BUS connect to input of a node with profile default, in that case, the default profile become same profile name as output bus node and the former default profile node syncrhonized it's inputs/outputs
when 1 connection is added to input, the input get the type of the connection
the number of input is dynamic and depends on an integer field which defined the number of input/output on top of BUS input/output
so if I have 5 as a value I have 6 inputs/outputs 1 for BUS + 5 anytype
