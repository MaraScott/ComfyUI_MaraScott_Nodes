import { app } from '../../../scripts/app.js'

const repoOwner = 'davask' // 替换为仓库的所有者
const repoName = 'ComfyUI-MarasIT-Nodes' // 替换为仓库的名称

const version = '3.2.8'

fetch(`https://api.github.com/repos/${repoOwner}/${repoName}/releases/latest`)
  .then(response => response.json())
  .then(data => {
    const latestVersion = data.tag_name
    console.log('Latest release version:', latestVersion)
    if (
      latestVersion &&
      latestVersion === localStorage.getItem('_davask_ComfyUI_MarasIT_Nodes_vesion')
    )
      return
    if (latestVersion && latestVersion != version) {
      localStorage.setItem('_davask_ComfyUI_MarasIT_Nodes_vesion', latestVersion)
      app.ui.dialog.show(`<a style="color: white;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 2px;
    }"
    href="https://discord.gg/ewX2yaDZ">Welcome to MaraScott.AI nodes discord</a>
    <h4 style="font-size: 18px;">${repoName} <br>
      Latest release version: ${latestVersion}</h4>
      <p>Please proceed to the official repository to download the latest version.</p>
      <a style="color: #2196F3;
      font-size: 18px;
      font-weight: 800;
      letter-spacing: 2px;
  }"
  href="https://github.com/${repoOwner}/${repoName}/releases/">https://github.com/${repoOwner}/${repoName}/releases</a>
      `)

      // window.alert(
      //     `Please proceed to the official repository to download the latest version.https://github.com/shadowcz007/comfyui-mixlab-nodes/releases/`
      //   )
      //   window.open(
      //     'https://github.com/shadowcz007/comfyui-mixlab-nodes/releases/'
      //   )
    }
  })
  .catch(error => {
    console.error('Error fetching release information:', error)
  })
//  #MixCopilot
