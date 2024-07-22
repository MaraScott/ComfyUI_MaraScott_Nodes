# ComfyUI MaraScott Nodes

**ComfyUI_MaraScott_nodes** is an extension set designed to improve readability of [ComfyUI](https://github.com/comfyanonymous/ComfyUI) workflow and improve output image for printing. It offers a **Bus** manager, an **Upscaler/Refiner** set of nodes for printing purposes and an **Inpainting** set of nodes to finetune an output.

[![](./docs/img/MaraScott.png)](https://www.marascott.ai/)

## NOTICE

**AnyBus Node**

- V2 - Dynamic Bus up to 25 input/outputs with profile management
- V1 - Static Bus

**McBoaty set**

- V5 - Improve Per Tile Prompt edition adding Image and Denoise on each tile & allow the use of McBoaty as 1st step in the refining process
- V4 - Introduce Per Tile Prompt Editor via an Upscaler, a TilePrompter and a refiner node
- V3 - Introduce Dynamic Tiling & assisted prompt generation via LLM
- V2 - Convert [@TreeShark](https://www.youtube.com/@robadams2451) initial Upscaler/Refiner workflow to a Node
- V1 - this was not a node at first but a workflow from [@TreeShark](https://www.youtube.com/@robadams2451), you can [watch where everything started](https://www.youtube.com/watch?v=eei9KAg7u48&t=0s)

**McInpainty set**

- V1 - Introducing a 2 nodes set : to generate an inpainted output and to paste it back to original image

## Installation

### Installation [method1] Installation via [ComfyUI-Manager](https://github.com/ltdrdata/ComfyUI-Manager) (recommended)

1. Click "Manager" button on main menu
2. Click "Custom Nodes Manager" Button in the Menu
3. Search for "MaraScott" and click "install" button
4. Restart ComfyUI

### Installation [method2] Installation via GIT

To install `ComfyUI_MaraScott_Nodes` in addition to an existing installation of ComfyUI, you can follow the following steps :

1. goto `ComfyUI/custom_nodes` dir in terminal (cmd)
2. `git clone https://github.com/MaraScott/ComfyUI_MaraScott_Node.git`
3. Restart ComfyUI

## Changes

* v5.0.0 - McBoaty : Improved Tile prompts editor
* v4.4.1 - McBoaty : Tile prompts editor
* v4.3.0 - McBoaty : hotfix tiling ksampling issue on non standard ratio
* v4.2.8 - McBoaty : add color match feature
* v4.2.3 - McBoaty : McBoaty Dynamic Tiling
* v4.0.0 - McBoaty : update workflow png images
* v3.3.0 - McBoaty : Add size feature, hidden iteration mecanism, log improvment
* v3.2.9 - McBoaty : patch-add_loop_system_back
* v3.2.8 - McBoaty : patch-node_versionning_plus_upscaler_v2
* v3.2.7 - McBoaty : patch-add_image_size_output
* v3.2.4 - McBoaty : no hotfix for slow node
* v3.2.3 - McBoaty : Add Sigmas_type management
* v3.2.2 - McBoaty : clean code
* v3.2.1 - McBoaty : Implement VAEEncode/DecodeTiled
* v3.1.0 - AnyBus : hotfix setNode issue
* v2.2.0 - AnyBus : add console log if 2 flows have same names
* v2.1.0 - update readme for delivery, update max input number, update web flow to hotfix initial load from dnd flow
* v2.0.0 - release of obsolete Bus_node + UniversalBusNode (py)
* v1.3.0 - AnyBus : Add Details to build Detailer Pipe
* v1.2.0 - AnyBus : Update node description and dynamic name

## How To Use

### AnyBus Node AKA UniversalBus Node

The AnyBus Node (AKA UniversalBus Node) is designed to provide a Universal Bus Node based on AnyType Input/Output.

**Native Support**

You can assume that any node input/output works like "UE Nodes / Use Everywhere" Nodes.

**BUS Input/Output Support**

Bus Nodes can be connected from one AnyBus Node to another, and we support the following Reroute Nodes as long as they are linked from an AnyBus Node before connecting it to another.

Set/Get Node **Supported**:

- "SetNode" from ComfyUI-KJNodes
- "GetNode" from ComfyUI-KJNodes

Reroute Node **Supported**:

- "Reroute (rgthree)" from RgThree Custom Node

Reroute Node **Not Supported**:

- "Reroute" from ComfyUI native - reason: does not allow connection on Any Type if the origin Type is not Any Type too
- "ReroutePrimitive|pysssss" from Pysssss Custom Node - reason: does not display the name of the origin slot
- "0246.CastReroute" from 0246 Custom Node - reason: undefined behavior

![AnyBus Node](./docs/img/bus-node.jpeg)
![AnyBus Node with assigned profile and some inputs](./docs/img/bus-node-profile.jpeg)

![AnyBus Node - Widget Qty](./docs/img/bus-node-widget-qty-inputs-outputs.jpeg)
![AnyBus Node - Widget Profile](./docs/img/bus-node-widget-profile-name.jpeg)

Here is a very simple workflow

![AnyBus Node WorkFlow Example](./docs/img/bus-node-workflow-example.png)

#### **What does it intend to do?**

Any workflow can quickly become messy and difficult to identify easily which output goes to which input.
At some point either you use a custom_node which "hide" the splines for you so you don't feel overwhelmed but you will still have some issue to identify which goes where OR you use a Bus node as AnyBus which will clarify your workflow without having to remember the origins of your splines.

The AnyBus Node allow you to apply a profile to your Buses in order to organize paths in the same workflow.
One Bus has a maximum of 25 inputs/outputs.

#### **The profile setting**

The BusNode profile is a mecanism to synchronize BusNodes in the same flow (connected by the `Bus` input/output) all nodes with the same profile name and connected together will be synchronized if one of them is modified.

When adding a node, the profile is default, if you have another flow called main for example and you connect the bus output of the flow to the input of the default busnode, the main flow will synchronize the main input to the default one and change the profile name to main. this works only for default node, if you have a node called TextFlow and try to connect main to TextFlow, the connection will be refused. this allow to have multiple flow in the same workflow and avoid conflict

This AnyBus is *dyslexia friendly* :D

### McBoaty Node Set (Upscaler, Prompter, Refiner)

McBoaty Node Set (AKA Upscaler, Prompter, Refiner Node set) is an upscaler coupled with a refiner to achieve higher rendering results on a per tile basis.

#### What does it intend to do ?

McBoaty intend to provide the most efficient way to Upscale and refine an image for printing purposes. Its core reason to exists is to provide a way to Upscale an image with minimum to null modification on the upscaled / refined image. to that purpose, using a denoise at 0.27 or maybe 0.35 is the sweet spot. Using multiple x2 upscaler model would as well provide a better result than using 1 model of x4 but that is for you to choose

#### To which StableDiffusion version is it compatible ?

it has been tested successfuly with SD1.5, SDXL, SD3, it will probably work with any SD version. Feel free to report issues in the issue section.

#### What's inside ?

McBoaty set of nodes come with 4 differents nodes :

1) An **Upscaler Node** to Upscale the input image and slice it in tiles to Refine the image on a per tile approach

![Upscaler Node](./docs/wf/McBoaty_v5/McBoaty_v5_Upscaler_Node.jpeg)

Inputs :

- image : input image
- model, clip, vae, seed : as usual
- positive, negative : this is required but not used, it will be replaced by positive and negative prompt in a near futur
- upscale model, output upscale method : as usual with an Upscale (by model) node
- tile size, feather mask : these 2 size will be used to slice your upscaled image and will define the size of the image the Ksampler will need to refine
- vae encode, tile size vae : do you want to use a tiled vae encoding method and which size
- color match method : do you want to apply a color match from your input image to the final output image after the refining process
- tile prompting active, vision llm model, llm model : this is currently used to generate the prompt for the conditioning, it is not accurate and will be improved in the futur

outputs :

- McBoaty Pipe : to connect to Refiner input only
- McPrompty Pipe : to connect to TilePrompter input only
- info : Obsolete

2) A **TilePrompter Node** to edit the prompt and denoise level on a per tile basis

![TilePrompter Node](./docs/wf/McBoaty_v5/McBoaty_v5_TilePrompter_Node.jpeg)

input

- pipe : (McPrompty) Pipe output from Upscaler, Refiner or LargeRefiner

output

- McPrompty Pipe : Pipe to connect to Refiner input pipe_prompty only

3) A **Refiner Node** to refine the image based on the settings provided, either via general settings if you don't use the tilePrompter or on a per tile basis if you do use the TilePrompter

![Refiner Node](./docs/wf/McBoaty_v5/McBoaty_v5_Refiner_Node.jpeg)

inputs:

- pipe : McBoaty Pipe output from Upscaler, Refiner or LargeRefiner
- McPrompty Pipe : to connect to TilePrompter input only
- tiles to process : you can specify an empty input if you want to process all the tiles or you can specify a tile index shown on the TilePrompter to refine only that tile. The input can specify multiple tiles at once as well either by specifying the tile indexes one by one (`1,2,3,4,...`) or using a range notation like `x-y ` (`1-4 `, `5-8 `) or a combinaison of those `1,3,4-6,8,9`
- output size type : you can choose between you Upscale size which will output an image of the size of your Upscale model or you can use custome size which will use the output size parameter
- output size : this is a coefficient to apply on your input image size as output size if you pick custom size type
- sigmas type : you can choose between BasicScheduler, SDTurboScheduler and AlignYourStepScheduler
- ays model type : you can choose between SD1, SDXL, SD3, SVD (SVD has never been tested) even if you choose a scheduler different from the AYS, you still need to choose you input model type
- sampler name, basic scheduler, steps, cfg, denoise : as usual on a Ksampler node
- control net name, strength, start percent, end percent : as usual using an apply control net node
- low threshold, high threshold : as usual using a canny node

outputs:

- McBoaty Pipe : to connect to Refiner input only
- McPrompty Pipe : to connect to TilePrompter input only
- image
- image (original)
- tiles
- prompts
- info

4) the fourth node is the **LargeRefiner Node** which is actualy the combinaison or the Upscaler Node + the Refiner Node in 1 node using the general settings

![LargeRefiner Node](./docs/wf/McBoaty_v5/McBoaty_v5_LargeRefiner_Node.jpeg)

inputs

- inputs from Upscaler + Refiner

outputs :

- outputs from refiner

#### How does it work ?

Below is an example of the LargeRefiner being connected to the TilePrompter and Refined by a second pass to a Refiner.

Using the upscale instead of the LargeRefiner will require to use the 1st pass of the refiner with tiles to process at empty value to process all tiles at least once (which the LargeRefiner does by default).

The overall idea is to Upscale (/Refine) your image and slice it to provide each tiles to McPrompty the TilePrompter which will allow you to fine tune your prompt on a per tile basis as well as the denoise level. then those parameters are passed to the Refiner which will process those tiles and rebuild your image for the output. then you can continue the refining on another refiner pass or modify the current TilePrompter parameters to finetune the result.

One use case, is to push the denoise to 0.76 (maximum denoise recommended with control net activated) in the LargeRefiner then finetune the prompt and denoise for the tiles which went too wild and process the tiles you want (specified in the tiles to process settings) again through a second refiner pass.

![img](./docs/img/McBoaty_v5_set.jpeg)

#### Where can I found a functional workflow to test it ?

Below is a functional workflow embedded in the image which allow you to test both technic.

![WorkFlow for McBoaty_V5](./docs/wf/McBoaty_v5/McBoaty_v5.png)

![Base Image for Comparaison](./docs/benchmark/Competitor/input/template.png "FromKyotoToIstanbul")

**Learn More on the Name McBoaty**

[![BoatyMcBoatface](./docs/img/BoatyMcBoatFace.png "BoatyMcBoatface")](https://en.wikipedia.org/wiki/Boaty_McBoatface "Learn more about the origin of the name McBoaty")

### McInpainty Node Set (Set & Paster Node) - *description in progress*

McInpainty Node is a set of 2 nodes

![img](./docs/img/McInpainty_v2_set.jpeg)

## Benchmark

### Upscaler

[Img Slider for result comparaison](https://imgsli.com/Mjc5NDg3/0/1)

| Custom_node             | Execution Time from server start to end of queue | output                                                    | Type     | workflow                                                                          | OS Name                   | Processor                            | Installed Physical Memory (RAM) GBGP | GPU Name                                      |
| ----------------------- | ------------------------------------------------ | --------------------------------------------------------- | -------- | --------------------------------------------------------------------------------- | ------------------------- | ------------------------------------ | ------------------------------------ | --------------------------------------------- |
| McBoaty LargeRefiner v5 | 98.37                                            | ![McBoaty](./docs/benchmark/Competitor/McBoaty/McBoaty.png) | Upscaler | [benchmark_McBoaty.json](./docs/benchmark/Competitor/McBoaty/benchmark_McBoaty.json) | Microsoft Windows 10 Home | Intel(R) Xeon(R) W-3235 CPU @ 3.30GH | 41                                   | Red Hat QXL controller NVIDIA Quadro RTX 6000 |
| Supir                   | 70.48                                            | ![SUPIR](./docs/benchmark/Competitor/SUPIR/SUPIR.png)       | Upscaler | [benchmark_SUPIR.json](./docs/benchmark/Competitor/SUPIR/benchmark_SUPIR.json)       | Microsoft Windows 10 Home | Intel(R) Xeon(R) W-3235 CPU @ 3.30GH | 41                                   | Red Hat QXL controller NVIDIA Quadro RTX 6000 |
| UltimateSDUpscaler      | 47.34                                            | ![USDU](./docs/benchmark/Competitor/USDU/USDU.png)          | Upscaler | [benchmark_USDU.json](./docs/benchmark/Competitor/USDU/benchmark_USDU.json)          | Microsoft Windows 10 Home | Intel(R) Xeon(R) W-3235 CPU @ 3.30GH | 41                                   | Red Hat QXL controller NVIDIA Quadro RTX 6000 |

**Visit this link to see [an Image Slider comparing the 3 results against the orignal](https://imgsli.com/Mjc5NDg3/0/1)**

## Special thanks

I would like to thank my [AI Classroom Discord](discord.gg/t28yZEewrp) buddies with a shoutout to:

- [@Fern](https://www.youtube.com/@ferniclestix) to whom I address **A special thanks for his support** in my ComfyUI journey who accepted me in his discord in the first place and tried to put good ideas in my head
- [Rob Adams](https://www.youtube.com/@robadams2451) AKA Treeshark who helped improve McBoaty by providing great insights and always pushing further the limits,
- [@YouFunnyGuys](discord.gg/t28yZEewrp) From the discord channel for his invaluable contribution and exceptional testing skills,

## Node related Thanks

### AnyBus

- Confyanimous with his [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- DrLtData with his [Custom nodes Manager](https://github.com/ltdrdata/ComfyUI-Manager),
- WASasquatch with his [Was Node Suite](https://github.com/WASasquatch/was-node-suite-comfyui)
- Kijai with his [Get/Set nodes](https://github.com/kijai/ComfyUI-KJNodes),
- Trung0246 with his [Highway node](https://github.com/Trung0246/ComfyUI-0246),
- Jags with his [support](https://www.youtube.com/channel/UCLXyz7oWNKx-Dp7Ba4v5ZZg)

### McBoaty

- [Rob Adams](https://www.youtube.com/@robadams2451) AKA Treeshark who provided the workflow in the first place which I converted into a ComfyUI Node, you can find his original workflow presentation in his YouTube video: [Advanced UpScaling in ComfyUI](https://www.youtube.com/watch?v=HStp7u682mE),
- DrLtData with his [ImpactPack](https://github.com/ltdrdata/ComfyUI-Impact-Pack),
- Kijai with his [ColorMatch node](https://github.com/kijai/ComfyUI-KJNodes/blob/main/nodes/image_nodes.py)

### McInpainty

- Chflame163 with his [LayerStyle](https://github.com/chflame163/ComfyUI_LayerStyle) Nodes

*I might have forgotten some other people, please contact me if you want to appear here and please forgive me.*

---

If anybody is interested in using these nodes, I'm following up on feature requests via the issues.
If you have Node ideas, feel free to make a request in the issues.
