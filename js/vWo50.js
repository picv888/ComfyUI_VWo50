import { app } from '../../../scripts/app.js'

app.registerExtension({
    // 名字和其它节点的不重复就行
    name: "VWo50 Extension",

    // 对每种节点类型都会调用一次该函数
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // nodeData.name是节点的名字，和后端中 NODE_CLASS_MAPPINGS 字典的键名一样
        // VWo50 Get Node Name 节点
        if (nodeData.name === "VWo50 Get Node Name") {
            // 重写onNodeCreated来添加自定义行为
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }

                // 添加上游节点信息显示
                this.addWidget("text", "上游节点", "未连接", () => { }, {
                    multiline: true,
                    height: 200
                });

                this.addWidget("text", "当前节点", `名字: ${this.type}`, () => { }, {
                    multiline: true
                });

                // 更新上游节点信息
                this.updateUpstreamInfo = function () {
                    const upstreamNodes = [];

                    // 检查所有输入连接
                    if (this.inputs) {
                        this.inputs.forEach(input => {
                            if (input.link !== null) {
                                const link = app.graph.links[input.link];
                                if (link) {
                                    const upstreamNode = app.graph.getNodeById(link.origin_id);
                                    if (upstreamNode) {
                                        upstreamNodes.push({
                                            node: upstreamNode,
                                            inputIndex: input.index
                                        });
                                    }
                                }
                            }
                        });
                    }

                    let infoText = "";
                    if (upstreamNodes.length === 0) {
                        infoText = "没有检测到上游节点连接";
                    } else {
                        infoText = "";
                        upstreamNodes.forEach(item => {
                            const node = item.node;
                            infoText += `• 节点类名: ${node.type}\n`;
                            infoText += `• 节点ID: ${node.id}\n`;
                            infoText += `• 节点标题: ${node.title || "无"}\n\n`;
                        });
                    }

                    // 更新小部件显示
                    if (this.widgets) {
                        const upstreamWidget = this.widgets.find(w => w.name === "上游节点");
                        if (upstreamWidget) {
                            upstreamWidget.value = infoText;
                        }

                        const currentWidget = this.widgets.find(w => w.name === "当前节点");
                        if (currentWidget) {
                            currentWidget.value = `名字: ${this.type}\nID: ${this.id}\n标题: ${this.title || "无"}`;
                        }
                    }
                };

                // 初始更新
                setTimeout(() => {
                    this.updateUpstreamInfo();
                }, 200);

                // 重写连接变化处理
                const originalOnConnectionsChange = this.onConnectionsChange;
                this.onConnectionsChange = function (type, index, connected, link_info) {
                    if (originalOnConnectionsChange) {
                        originalOnConnectionsChange.apply(this, arguments);
                    }
                    this.updateUpstreamInfo();
                };
            };
        }

        // 需要自动增加参数的节点 
        if (nodeData.name === "VWo50 Execute Python Code" ||
            nodeData.name === "VWo50 Make List And Get Element At Index" ||
            nodeData.name === "VWo50 Switch") {
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
                if (onConnectionsChange) onConnectionsChange.apply(this, arguments);

                // type === 1 是输入连接改变
                if (type === 1) {
                    // 1. 找到所有“input_”开头的输入放进数组里
                    let inputArr = [];

                    for (let i = 0; i < this.inputs.length; i++) {
                        const name = this.inputs[i].name;
                        if (name && name.startsWith("input_")) {
                            inputArr.push(i);
                        }
                    }

                    // 2. 如果最后一个输入已连接，则增加一个新的
                    let lastInputIndex = inputArr[inputArr.length - 1]; // 至少会有一个，所以这里不担心数组越界
                    let lastInput = this.inputs[lastInputIndex];
                    if (lastInput.link !== null) {
                        if (nodeData.name === "VWo50 Switch") {
                            this.addInput(`input_${inputArr.length}`, "*", { lazy: true });  // 标记为lazy
                        } else {
                            this.addInput(`input_${inputArr.length}`, "*");
                        }
                    }

                    // 3. 如果末尾有两个未连接的输入，删掉后面那个
                    if (inputArr.length > 1) {
                        for (let i = inputArr.length - 1; i > 0; i--) {
                            let input1 = this.inputs[inputArr[i]]
                            let input2 = this.inputs[inputArr[i - 1]]
                            if (input1.link === null && input2.link === null) {
                                this.removeInput(inputArr[i]);
                            } else {
                                break;
                            }
                        }
                    }
                }
            };
        }

        // VWo50 Preview Image 节点
        if (nodeData.name === "VWo50 Preview Image") {
            // 节点创建时：添加隐藏的 mask_file widget -----
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);
                // 找到后端 INPUT_TYPES 中定义的 mask_file widget
                const maskWidget = this.widgets?.find(w => w.name === "mask_file");
                // const has_preview_image_Widget = this.widgets?.find(w => w.name === "has_preview_image");
                if (maskWidget) {
                    // 使用官方推荐方式隐藏但保持可序列化
                    maskWidget.hidden = true;
                    maskWidget.options = maskWidget.options || {};
                    maskWidget.options.hidden = true;
                    maskWidget.computeSize = () => [0, -4]; // 折叠尺寸
                    this._maskFileWidget = maskWidget;
                }
                // if(has_preview_image_Widget){
                //     this._has_preview_image_widget = has_preview_image_Widget
                // }
            };

            // ----- 2. onExecuted 渲染预览图 (保持不变) -----
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (onExecuted) onExecuted.apply(this, arguments);
                const uiData = message?.ui || message;
                const images = uiData?.images;
                if (!images || images.length === 0) return;

                this.imgs = images.map((info) => {
                    const params = new URLSearchParams({
                        filename: info.filename,
                        type: info.type || "temp",
                        subfolder: info.subfolder || "",
                    });
                    return `/view?${params.toString()}`;
                });
                // this._has_preview_image_widget.value = true
                this.setDirtyCanvas(true, true);
            };

            // ----- 3. 核心：拦截 fetch，捕获 /upload/image 的响应 -----
            const originalFetch = window.fetch;
            window.fetch = async function (...args) {
                const [resource, config] = args;
                const url = typeof resource === 'string' ? resource : resource.url;
                
                // 调用原本的 fetch
                const response = await originalFetch.apply(this, args);

                // 只处理发往 /upload/image 的请求
                if (url && url.includes('/upload/image') && config?.method === 'POST') {
                    try {
                        // 克隆响应，避免消耗原始数据流
                        const clonedResponse = response.clone();
                        const data = await clonedResponse.json();
                        
                        // data 格式类似: { name: "clipspace-mask-xxx.png", subfolder: "clipspace", type: "input" }
                        if (data && data.name) {
                            // 找到当前选中的VWo50 Preview Image节点
                            const selected = Object.values(app.canvas?.selected_nodes || {});
                            const activeNode = selected.find(n => n.type === "VWo50 Preview Image");
                            if (activeNode && activeNode._maskFileWidget) {
                                const filePath = data.subfolder ? `${data.subfolder}/${data.name}` : data.name;
                                activeNode._maskFileWidget.value = filePath;
                                activeNode.setDirtyCanvas(true, true);
                            }
                        }
                    } catch (e) {
                        console.log(e)
                        // 静默处理，避免干扰正常的网络请求
                    }
                }
                return response;
            };
        }
    }
});