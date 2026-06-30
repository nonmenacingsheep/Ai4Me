const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('electron', {
  close:    () => ipcRenderer.send('window-close'),
  minimize: () => ipcRenderer.send('window-minimize'),
  maximize: () => ipcRenderer.send('window-maximize'),
  passthroughEnter: () => ipcRenderer.send('passthrough-enter'),
  passthroughExit:  () => ipcRenderer.send('passthrough-exit'),
  dragStart: (x, y) => ipcRenderer.send('window-drag-start', { x, y }),
  dragMove:  (x, y) => ipcRenderer.send('window-drag-move', { x, y }),
  pickFolder: () => ipcRenderer.invoke('pick-folder'),
});
