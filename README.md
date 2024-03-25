# ComfyUI Node: AnyBus or UniBus for Universal Bus

AnyBus Node (AKA UniversalBus) is an attempt to provide a Universal Bus Node (as some might say) which is based on AnyType Input/Output.

the Bus Nodes can be connected from one AnyBusNode to another and we support the following Reroute Nodes as long as they are linked from an AnyBusNode before connecting it to another

Reroute Node **Supported** :

- "Reroute (rgthree)" from RgThree Custom Node

Reroute Node **Not Supported** :
- "Reroute" from ComfyUI native - reason : do not allow connection on Any Type if origin Type is not Any Type too
- "ReroutePrimitive|pysssss" from Pysssss Custom Node - reason : do not display the name of the origin slot
- "0246.CastReroute" from 0246 Custom Node - reason : undefined behavior

I would like to thanks the following person to help me out by providing there awesome code and advises :
- WASasquatch with his [Was Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)
- KJNodes with his [Get/Set nodes](https://github.com/kijai/ComfyUI-KJNodes), 
- Trung0246 with his [Highway node](https://github.com/Trung0246/ComfyUI-0246), 
- DrLtData with his [Custom nodes Manager](https://github.com/ltdrdata/ComfyUI-Manager), 
- Confyanimous with his [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- Jags with his [support](https://www.youtube.com/channel/UCLXyz7oWNKx-Dp7Ba4v5ZZg)

*I might have forgotten some others person, feel free to contact me if you want to appear here and please forgive me*

**A special thanks to [Fern](https://www.youtube.com/@ferniclestix) for his great support**.

![AnyBus Node](./docs/img/bus-node.jpeg)
![AnyBus Node with assigned profile and some inputs](./docs/img/bus-node-profile.jpeg)

![AnyBus Node - Widtget Qty](./docs/img/bus-node-widget-qty-inputs-outputs.jpeg)
![AnyBus Node - Widtget Profile](./docs/img/bus-node-widget-profile-name.jpeg)

Here a very simple workflow

![AnyBus Node WorkFlow Example](./docs/img/bus-node-workflow-example.png)

What does it intend to do ?

In the world of ComfyUI, the spaghetti workflow can quickly become a mess, I personaly at some point was confused too by the get/set approach which brought me to develop this AnyBus Node applying a profile approach to organise different Buses in the same workflow.
One Bus has a maximum of 25 inputs/outputs (feel free to make an issue request to get a bigger max)

This AnyBus is dislexia friendly :D

What are the features ?

- dynamic number of input from 3 to 25
- profile assignment

If anybody is interested to use it, I'm following on feature request via the issues.
