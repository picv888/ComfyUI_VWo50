import { app } from '../../../scripts/app.js'

app.registerExtension({
    // 名字和其它节点的不重复就行
    name: "VWo50GetNodeName",

    // 对每种节点类型都会调用一次该函数
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // nodeData.name是节点的名字，和后端中 NODE_CLASS_MAPPINGS 字典的键名一样
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
    }
});